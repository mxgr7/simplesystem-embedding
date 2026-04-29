"""Operational helpers around `pymilvus.MilvusClient`.

The F7 spec pins a few invariants that benefit from being applied at a
single layer rather than each call site:

  - **Bounded consistency** on every read. Milvus's default is
    `Bounded` for `search`/`query` *if no explicit value is provided
    AND the collection's per-collection default is bounded*, but the
    safer pattern (per F7 §"Milvus consistency level") is to set it
    explicitly so a future change to the cluster default doesn't
    silently shift latency-sensitive reads to `Strong`. We pin
    `Bounded` on every search/query/get unless the caller passes a
    different value explicitly.

  - **Per-call timeout** on every read. `pymilvus.MilvusClient` accepts
    `timeout=N` (seconds) on `search`/`query`/`get`; default is
    `None` = wait forever. We pin a default so a stuck Milvus surfaces
    as a 5xx within the request budget (legacy SLO p99 < 5s).

  - **Retries on transient failures**. Same policy shape as the
    legacy Java service (`management.tracing.retry.*`): max 5
    attempts, 500ms base backoff, 1.5× multiplier, capped at 5s per
    backoff. Permanent errors (validation/schema/permission) raise
    immediately. The total retry budget is capped at
    `RetryPolicy.total_budget_s` so a deeply-degraded cluster doesn't
    eat the entire request SLO.

  - Future hooks: tracing baggage propagation + RED metrics — the
    same wrapper is the natural place for them. F7 follow-ups land
    here without a second ripple of call-site edits.

Use via:

    from milvus_helpers import BoundedMilvusClient
    raw = MilvusClient(uri=...)
    client = BoundedMilvusClient(raw)
    client.search(collection_name=..., data=..., ...)
    # → consistency_level='Bounded' + timeout=DEFAULT_PER_CALL_TIMEOUT_S
    # + retries on transient errors

Pass-through: any method not on this wrapper falls through to the
underlying client (e.g. `has_collection`, `describe_collection`,
`upsert`, `flush`) — those are write paths or schema reads where
neither consistency, retry, nor timeout pins apply (writes aren't
naturally idempotent; schema reads are sub-millisecond).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pymilvus import MilvusClient

log = logging.getLogger(__name__)

# Per F7 §"Milvus consistency level" — see the spec linked from the
# module docstring for the `Bounded` rationale (latency-bounded
# staleness instead of `Strong` synchronous coordination).
DEFAULT_CONSISTENCY_LEVEL = "Bounded"

# Per F7 §"Timeouts": p99 SLO is 5s. Pick a per-call default that
# leaves room for retries (5 attempts × budget headroom). 4s is
# tight but means a single-attempt timeout doesn't immediately spend
# the whole request budget.
DEFAULT_PER_CALL_TIMEOUT_S = 4.0


# Errors we treat as PERMANENT — substring match on the message,
# case-insensitive. Mirrors the bulk_insert denylist; expand as we
# encounter new permanent failure shapes in production. A permanent
# error raises on the first attempt rather than burning the retry
# budget.
_PERMANENT_ERROR_KEYWORDS = (
    "validation",
    "schema",
    "field not found",
    "not exist",
    "not found",
    "permission",
    "unauthenticated",
    "duplicate",
    "invalid parameter",
    "syntax error",
)


def _is_permanent_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _PERMANENT_ERROR_KEYWORDS)


@dataclass(frozen=True)
class RetryPolicy:
    """Mirrors the legacy retry policy
    (`management.tracing.retry.*` in
    article/search/query/src/main/resources/application.yml:91-94):
      - max_attempts: 5 (1 initial + 4 retries)
      - initial_backoff_s: 0.5
      - multiplier: 1.5
      - max_single_backoff_s: 5.0
    Cumulative backoff for 5 attempts ≈ 0.5+0.75+1.13+1.69 = 4.07s,
    which fits inside `total_budget_s=5` when the per-call timeout
    is also bounded. Beyond `total_budget_s` we raise the last
    exception immediately rather than spending more SLO."""
    max_attempts: int = 5
    initial_backoff_s: float = 0.5
    multiplier: float = 1.5
    max_single_backoff_s: float = 5.0
    total_budget_s: float = 5.0


T = TypeVar("T")


def retry(call: Callable[[], T], *, policy: RetryPolicy, label: str) -> T:
    """Run `call()` with exponential-backoff retry per `policy`.
    Permanent errors raise immediately (no retry). Transient errors
    retry until exhaustion or until the cumulative backoff would
    exceed `policy.total_budget_s`."""
    started = time.monotonic()
    last_exc: BaseException | None = None
    for attempt in range(policy.max_attempts):
        try:
            return call()
        except BaseException as e:
            if _is_permanent_error(e):
                log.warning("%s: permanent error (%s) — not retrying", label, e)
                raise
            last_exc = e
            if attempt == policy.max_attempts - 1:
                log.warning("%s: exhausted %d attempts — last error: %s",
                            label, policy.max_attempts, e)
                raise
            elapsed = time.monotonic() - started
            wait = min(
                policy.initial_backoff_s * (policy.multiplier ** attempt),
                policy.max_single_backoff_s,
            )
            if elapsed + wait > policy.total_budget_s:
                log.warning(
                    "%s: total retry budget %.1fs exhausted at attempt %d — last error: %s",
                    label, policy.total_budget_s, attempt + 1, e,
                )
                raise
            log.info(
                "%s attempt %d/%d failed (%s) — retrying in %.2fs",
                label, attempt + 1, policy.max_attempts, e, wait,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable") from last_exc


class BoundedMilvusClient:
    """Thin wrapper that pins `consistency_level='Bounded'`, a default
    timeout, and exponential-backoff retries on every read call
    (search/query/get). Pass-through for everything else.

    `retry_policy=None` disables retries entirely — useful for tests
    or when the caller is already inside a retry loop. `timeout_s=None`
    uses pymilvus's wait-forever default."""

    __slots__ = ("_client", "_timeout_s", "_policy")

    def __init__(
        self,
        client: MilvusClient,
        *,
        timeout_s: float | None = DEFAULT_PER_CALL_TIMEOUT_S,
        retry_policy: RetryPolicy | None = RetryPolicy(),
    ) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_timeout_s", timeout_s)
        object.__setattr__(self, "_policy", retry_policy)

    # ---- consistency + timeout + retry-pinned read methods ----------

    def _call(self, method: Callable[..., Any], label: str, **kwargs: Any) -> Any:
        kwargs.setdefault("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
        if self._timeout_s is not None:
            kwargs.setdefault("timeout", self._timeout_s)
        if self._policy is None:
            return method(**kwargs)
        return retry(lambda: method(**kwargs), policy=self._policy, label=label)

    def search(self, **kwargs: Any) -> Any:
        return self._call(self._client.search, "milvus.search", **kwargs)

    def query(self, **kwargs: Any) -> Any:
        return self._call(self._client.query, "milvus.query", **kwargs)

    def get(self, **kwargs: Any) -> Any:
        return self._call(self._client.get, "milvus.get", **kwargs)

    # ---- pass-through for everything else --------------------------
    # `__getattr__` only fires when normal attribute lookup misses, so
    # `self._client` (set in __init__) doesn't recurse here.

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = [
    "BoundedMilvusClient",
    "RetryPolicy",
    "retry",
    "DEFAULT_CONSISTENCY_LEVEL",
    "DEFAULT_PER_CALL_TIMEOUT_S",
]
