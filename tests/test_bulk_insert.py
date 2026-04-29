"""Unit tests for the bulk-insert sink helpers in
`indexer.bulk_insert`. Three groups:

  1. Retry wrapper — transient vs permanent error classification,
     exponential backoff, exhaustion.
  2. Checkpoint serialization — atomic write + load round-trip,
     missing/corrupt/version-mismatched files.
  3. Chunked writer offset semantics — `starting_chunk_idx` makes file
     names continue past prior runs without collision.

Real-Milvus / real-MinIO behavior is exercised by the live smoke tests
elsewhere; this file stays unit-scoped (no infra dependencies).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.bulk_insert import (  # noqa: E402
    BulkInsertConfig,
    _do_bulk_insert_with_retry,
    _empty_checkpoint,
    _is_permanent_milvus_error,
    load_checkpoint,
    save_checkpoint,
)


# ---------- retry wrapper -------------------------------------------------

class _TransientError(RuntimeError):
    """Stand-in for a transient gRPC/network error from Milvus."""


class _PermanentError(RuntimeError):
    """Stand-in for a permanent error — message contains a permanent
    keyword (`schema`) so `_is_permanent_milvus_error` classifies it."""

    def __str__(self) -> str:
        return "schema validation failed"


def _cfg(attempts: int = 3, initial: float = 0.0) -> BulkInsertConfig:
    """Tests use 0s backoff so they run instantly. Real production
    runs use the dataclass defaults."""
    return BulkInsertConfig(
        retry_attempts=attempts,
        retry_initial_backoff_s=initial,
        retry_max_backoff_s=initial,
    )


def test_retry_succeeds_on_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_do_bulk_insert(**kwargs: Any) -> int:
        calls["n"] += 1
        return 12345

    monkeypatch.setattr("indexer.bulk_insert.utility.do_bulk_insert", fake_do_bulk_insert)
    job_id = _do_bulk_insert_with_retry(collection="c", files=["k"], cfg=_cfg())
    assert job_id == 12345
    assert calls["n"] == 1


def test_retry_recovers_after_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two transient failures, succeed on third attempt — within the
    default 3-attempt budget."""
    calls = {"n": 0}

    def fake_do_bulk_insert(**kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _TransientError("connection reset by peer")
        return 99

    monkeypatch.setattr("indexer.bulk_insert.utility.do_bulk_insert", fake_do_bulk_insert)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", lambda _: None)
    job_id = _do_bulk_insert_with_retry(collection="c", files=["k"], cfg=_cfg())
    assert job_id == 99
    assert calls["n"] == 3


def test_retry_exhausts_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_do_bulk_insert(**kwargs: Any) -> int:
        calls["n"] += 1
        raise _TransientError(f"unavailable (attempt {calls['n']})")

    monkeypatch.setattr("indexer.bulk_insert.utility.do_bulk_insert", fake_do_bulk_insert)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", lambda _: None)
    with pytest.raises(_TransientError):
        _do_bulk_insert_with_retry(collection="c", files=["k"], cfg=_cfg(attempts=4))
    assert calls["n"] == 4


def test_retry_skips_permanent_error_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permanent errors (schema, validation, perms) should not waste
    backoff cycles — they raise on the first attempt."""
    calls = {"n": 0}

    def fake_do_bulk_insert(**kwargs: Any) -> int:
        calls["n"] += 1
        raise _PermanentError()

    monkeypatch.setattr("indexer.bulk_insert.utility.do_bulk_insert", fake_do_bulk_insert)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", lambda _: None)
    with pytest.raises(_PermanentError):
        _do_bulk_insert_with_retry(collection="c", files=["k"], cfg=_cfg(attempts=10))
    assert calls["n"] == 1, "permanent error should not retry"


def test_retry_backoff_is_exponential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three transient failures + success on attempt 4 should sleep
    [initial, initial*2, initial*4] (capped at retry_max_backoff_s)."""
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    calls = {"n": 0}

    def fake_do_bulk_insert(**kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] < 4:
            raise _TransientError(f"transient {calls['n']}")
        return 7

    monkeypatch.setattr("indexer.bulk_insert.utility.do_bulk_insert", fake_do_bulk_insert)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", fake_sleep)
    cfg = BulkInsertConfig(
        retry_attempts=5, retry_initial_backoff_s=1.0, retry_max_backoff_s=4.0,
    )
    _do_bulk_insert_with_retry(collection="c", files=["k"], cfg=cfg)
    # attempts 1..3 fail (sleeps 1, 2, 4); attempt 4 succeeds.
    assert sleeps == [1.0, 2.0, 4.0], f"unexpected backoff sequence {sleeps}"


def test_permanent_error_classification() -> None:
    """The substring match is case-insensitive and covers the documented
    permanent keywords — guard against accidental loosening of the
    denylist that would silently retry validation errors."""
    for msg in (
        "Validation failed",
        "schema mismatch",
        "Field not found",
        "permission denied",
        "collection not exist",
        "Unauthenticated request",
        "duplicate primary keys",
    ):
        assert _is_permanent_milvus_error(RuntimeError(msg)), f"{msg!r} should be permanent"

    for msg in (
        "connection reset by peer",
        "deadline exceeded",
        "service unavailable",
        "timeout while waiting for response",
    ):
        assert not _is_permanent_milvus_error(RuntimeError(msg)), (
            f"{msg!r} should be transient"
        )


# ---------- checkpoint ----------------------------------------------------

def test_load_checkpoint_returns_empty_when_path_is_none() -> None:
    state = load_checkpoint(None)
    assert state == _empty_checkpoint()


def test_load_checkpoint_returns_empty_when_file_missing(tmp_path: Path) -> None:
    state = load_checkpoint(tmp_path / "nope.json")
    assert state["articles"] == {"rows_done": 0, "chunks_done": 0}
    assert state["offers"] == {"rows_done": 0, "chunks_done": 0}


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    state = _empty_checkpoint()
    state["articles"]["rows_done"] = 5_000_000
    state["articles"]["chunks_done"] = 5
    state["offers"]["rows_done"] = 100
    state["offers"]["chunks_done"] = 1
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, state)

    loaded = load_checkpoint(cp)
    assert loaded["articles"]["rows_done"] == 5_000_000
    assert loaded["articles"]["chunks_done"] == 5
    assert loaded["offers"]["rows_done"] == 100
    assert loaded["offers"]["chunks_done"] == 1


def test_save_checkpoint_is_atomic(tmp_path: Path) -> None:
    """The temp file pattern protects against half-written checkpoints
    on crash. Verify by writing then opening the temp file should
    NOT exist (it gets renamed away)."""
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, _empty_checkpoint())
    assert cp.exists()
    assert not (tmp_path / "checkpoint.json.tmp").exists()


def test_save_checkpoint_creates_parent_dirs(tmp_path: Path) -> None:
    cp = tmp_path / "deeply" / "nested" / "dir" / "checkpoint.json"
    save_checkpoint(cp, _empty_checkpoint())
    assert cp.exists()


def test_save_checkpoint_overwrites_existing(tmp_path: Path) -> None:
    cp = tmp_path / "checkpoint.json"
    state1 = _empty_checkpoint()
    state1["articles"]["rows_done"] = 100
    save_checkpoint(cp, state1)
    state2 = _empty_checkpoint()
    state2["articles"]["rows_done"] = 200
    save_checkpoint(cp, state2)
    loaded = load_checkpoint(cp)
    assert loaded["articles"]["rows_done"] == 200


def test_save_checkpoint_noop_when_path_is_none(tmp_path: Path) -> None:
    """The orchestrator passes path=None when resume is disabled.
    Calling save_checkpoint(None, ...) must not raise."""
    save_checkpoint(None, _empty_checkpoint())   # no exception


def test_load_checkpoint_rejects_unknown_version(tmp_path: Path) -> None:
    cp = tmp_path / "checkpoint.json"
    cp.write_text(json.dumps({"version": 99, "articles": {}, "offers": {}}))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(cp)


def test_load_checkpoint_tolerates_partial_state(tmp_path: Path) -> None:
    """A user could hand-edit a checkpoint to only have one stream's
    state. Loading should fill in defaults rather than KeyError later."""
    cp = tmp_path / "checkpoint.json"
    cp.write_text(json.dumps({
        "version": 1,
        "articles": {"rows_done": 50, "chunks_done": 1},
        # offers omitted
    }))
    state = load_checkpoint(cp)
    assert state["articles"]["rows_done"] == 50
    assert state["offers"]["rows_done"] == 0


# ---------- stream_chunks_to_milvus polling semantics --------------------

class _FakeBulkInsertState:
    """Stand-in for `pymilvus.utility.BulkInsertState`. We only read
    `state_name`, `row_count`, `infos`, `progress`."""
    def __init__(self, *, state_name: str, row_count: int = 0, progress: int = 100, infos: str = ""):
        self.state_name = state_name
        self.row_count = row_count
        self.progress = progress
        self.infos = infos


def _patch_milvus(monkeypatch: pytest.MonkeyPatch, *,
                  do_bulk_insert: Any, get_state: Any) -> None:
    monkeypatch.setattr("indexer.bulk_insert._ensure_milvus_connection", lambda u: None)
    monkeypatch.setattr("indexer.bulk_insert.utility.do_bulk_insert", do_bulk_insert)
    monkeypatch.setattr("indexer.bulk_insert.utility.get_bulk_insert_state", get_state)
    monkeypatch.setattr("indexer.bulk_insert.upload_to_s3", lambda local, **kw: None)


def _make_chunks(stage_dir: Path, n: int, rows_per_chunk: int = 100):
    """Build N empty placeholder parquet files so the orchestrator's
    `path.unlink(missing_ok=True)` doesn't error. Yield in (chunk_idx,
    path, rows, bytes) shape, starting from chunk_idx=0."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = stage_dir / f"chunk.{i:04d}.parquet"
        p.write_bytes(b"")
        yield i, p, rows_per_chunk, 0


def test_callback_fires_in_chunk_idx_order_even_when_jobs_complete_out_of_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If chunks 0/1/2 submit and chunk 2 completes server-side first
    (chunk 1 still importing), the checkpoint must NOT advance to
    chunks_done=3 yet — that would make a resume skip chunk 1's
    half-imported state on rerun."""
    from indexer.bulk_insert import stream_chunks_to_milvus  # noqa: PLC0415

    chunk_to_job = {}  # chunk_idx → job_id
    next_job_id = [100]

    def fake_do_bulk_insert(*, collection_name, files):
        # Recover chunk_idx from the file name (chunk.0007.parquet → 7).
        name = files[0].rsplit("/", 1)[-1]
        chunk_idx = int(name.split(".")[1])
        next_job_id[0] += 1
        chunk_to_job[chunk_idx] = next_job_id[0]
        return next_job_id[0]

    # Chunk 2 completes immediately; chunks 0 and 1 take 2 polls each.
    poll_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    def fake_get_state(job_id):
        for chunk_idx, j in chunk_to_job.items():
            if j == job_id:
                poll_counts[chunk_idx] += 1
                if chunk_idx == 2:
                    return _FakeBulkInsertState(state_name="Completed", row_count=100)
                if poll_counts[chunk_idx] >= 2:
                    return _FakeBulkInsertState(state_name="Completed", row_count=100)
                return _FakeBulkInsertState(state_name="Importing", progress=50)
        raise AssertionError(f"unknown job {job_id}")

    _patch_milvus(monkeypatch, do_bulk_insert=fake_do_bulk_insert, get_state=fake_get_state)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", lambda _: None)

    cfg = BulkInsertConfig(stage_dir=tmp_path, upload_workers=1, poll_interval_s=0.0)
    callback_log: list[tuple[int, int]] = []
    chunks = _make_chunks(tmp_path, n=3)

    total_rows, stats = stream_chunks_to_milvus(
        chunks, milvus_uri="http://stub", collection="test", cfg=cfg,
        on_chunk_completed=lambda idx, rows: callback_log.append((idx, rows)),
    )

    # Callbacks must fire in order 0, 1, 2 — even though chunk 2 was
    # ready first.
    assert callback_log == [(0, 100), (1, 100), (2, 100)]
    assert total_rows == 300


def test_submit_failure_still_polls_and_persists_earlier_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If chunk 0 + 1 submit successfully but chunk 2's submit raises,
    the orchestrator must still poll chunks 0 + 1 to completion and
    fire their callbacks BEFORE re-raising. This is what makes resume
    correct after a transient bulk_insert failure mid-run."""
    from indexer.bulk_insert import stream_chunks_to_milvus  # noqa: PLC0415

    chunk_to_job: dict[int, int] = {}
    next_job_id = [100]

    def fake_do_bulk_insert(*, collection_name, files):
        name = files[0].rsplit("/", 1)[-1]
        chunk_idx = int(name.split(".")[1])
        if chunk_idx == 2:
            raise RuntimeError("connection reset by peer (injected)")
        next_job_id[0] += 1
        chunk_to_job[chunk_idx] = next_job_id[0]
        return next_job_id[0]

    def fake_get_state(job_id):
        return _FakeBulkInsertState(state_name="Completed", row_count=100)

    _patch_milvus(monkeypatch, do_bulk_insert=fake_do_bulk_insert, get_state=fake_get_state)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", lambda _: None)

    # retry_attempts=1 means no retry; first failure raises.
    cfg = BulkInsertConfig(
        stage_dir=tmp_path, upload_workers=1, poll_interval_s=0.0,
        retry_attempts=1, retry_initial_backoff_s=0.0, retry_max_backoff_s=0.0,
    )
    callback_log: list[tuple[int, int]] = []
    chunks = _make_chunks(tmp_path, n=3)

    with pytest.raises(RuntimeError, match="injected"):
        stream_chunks_to_milvus(
            chunks, milvus_uri="http://stub", collection="test", cfg=cfg,
            on_chunk_completed=lambda idx, rows: callback_log.append((idx, rows)),
        )

    # Even though chunk 2 raised, chunks 0 and 1 successfully submitted +
    # were polled to Completed + their callbacks fired. The orchestrator
    # then re-raised. A resume reads chunks_done=2 and starts at chunk 2.
    assert callback_log == [(0, 100), (1, 100)]


def test_failed_chunk_state_does_not_skip_earlier_completed_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If chunks 0+1 Complete and chunk 2 reports Failed server-side,
    we still want chunks 0+1's callbacks fired before the orchestrator
    raises about the failure. Same shape as the submit-failure case but
    failure is on the server side, not in the wrapper."""
    from indexer.bulk_insert import stream_chunks_to_milvus  # noqa: PLC0415

    chunk_to_job: dict[int, int] = {}
    next_job_id = [100]

    def fake_do_bulk_insert(*, collection_name, files):
        name = files[0].rsplit("/", 1)[-1]
        chunk_idx = int(name.split(".")[1])
        next_job_id[0] += 1
        chunk_to_job[chunk_idx] = next_job_id[0]
        return next_job_id[0]

    def fake_get_state(job_id):
        for chunk_idx, j in chunk_to_job.items():
            if j == job_id:
                if chunk_idx == 2:
                    return _FakeBulkInsertState(
                        state_name="Failed", infos="parquet schema mismatch",
                    )
                return _FakeBulkInsertState(state_name="Completed", row_count=100)
        raise AssertionError(f"unknown job {job_id}")

    _patch_milvus(monkeypatch, do_bulk_insert=fake_do_bulk_insert, get_state=fake_get_state)
    monkeypatch.setattr("indexer.bulk_insert.time.sleep", lambda _: None)

    cfg = BulkInsertConfig(stage_dir=tmp_path, upload_workers=1, poll_interval_s=0.0)
    callback_log: list[tuple[int, int]] = []
    chunks = _make_chunks(tmp_path, n=3)

    with pytest.raises(RuntimeError, match="schema mismatch"):
        stream_chunks_to_milvus(
            chunks, milvus_uri="http://stub", collection="test", cfg=cfg,
            on_chunk_completed=lambda idx, rows: callback_log.append((idx, rows)),
        )

    # 0 and 1 must have been recorded before the chunk-2 Failed status
    # propagates.
    assert callback_log == [(0, 100), (1, 100)]


# ---------- chunked writer offset ----------------------------------------

def test_chunk_idx_continues_past_starting_index(tmp_path: Path) -> None:
    """`starting_chunk_idx=N` makes the first emitted chunk be N, so a
    resume's parquet file names don't collide with already-uploaded
    chunks from the prior run."""
    from indexer.bulk_insert import _BatchConverter, _write_chunked  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    schema = pa.schema([("v", pa.int64())])

    def convert(batch: list[dict]) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([r["v"] for r in batch], type=pa.int64())],
            schema=schema,
        )

    batches = [
        [{"v": 1}, {"v": 2}],
        [{"v": 3}, {"v": 4}],
        [{"v": 5}],
    ]

    out: list[tuple[int, str, int]] = []
    for chunk_idx, path, rows, _bytes in _write_chunked(
        iter(batches),
        stage_dir=tmp_path,
        name_template="t.{idx:04d}.parquet",
        schema=schema,
        convert=convert,
        chunk_rows=2,
        compression="zstd",
        compression_level=1,
        starting_chunk_idx=42,
    ):
        out.append((chunk_idx, path.name, rows))

    # 5 input rows / 2 per chunk → 3 chunks (last is partial).
    assert out == [
        (42, "t.0042.parquet", 2),
        (43, "t.0043.parquet", 2),
        (44, "t.0044.parquet", 1),
    ]
