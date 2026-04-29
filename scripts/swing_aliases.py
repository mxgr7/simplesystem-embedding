"""F9 paired alias-swing CLI — atomic-as-possible swap of `articles`
and `offers` to a freshly-built collection pair.

Replaces the manual procedure in `scripts/MILVUS_ALIAS_WORKFLOW.md`
"Paired alias swing" section. The swing is sequenced articles-then-offers
so a partial failure leaves the prior offers reading against the new
articles (instead of the other way round, which would orphan offer
rows pointing at no-longer-aliased article hashes).

Pre-flight validation (skip with --no-validate):
  - Both target collections exist + are loaded.
  - Row counts ≥ --min-rows-articles / --min-rows-offers (catch a
    half-populated bulk run before swinging).
  - Sample N random offers, verify their article_hash exists in the
    target articles collection (catches a join-key drift between
    the two streams).

Rollback:
  - First swing succeeds, second fails → first is rolled back
    automatically + the original failure is re-raised.
  - Both swings succeeded but operator wants to revert (data quality
    issue surfaces post-swing) → re-run with --rollback-to to point
    aliases back at a known-good prior pair.

Typical invocation after `indexer_bulk.py` populates a fresh pair:

    uv run python scripts/swing_aliases.py \\
        --articles-target articles_v4 \\
        --offers-target offers_v5 \\
        --milvus-uri http://localhost:19530

Dry-run prints what would happen without contacting Milvus to mutate.

Per F9: this script does NOT drop the previous collections — the
operator drops them after a soak window (default 7 days) once
confident no in-flight requests still reference them.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass

from pymilvus import MilvusClient

log = logging.getLogger(__name__)


@dataclass
class AliasState:
    """Snapshot of an alias target, captured before mutation so we have
    a precise rollback target."""
    alias: str
    target: str | None  # None = alias didn't exist before


def _current_alias(client: MilvusClient, alias: str) -> AliasState:
    try:
        info = client.describe_alias(alias=alias)
        target = info.get("collection_name") if isinstance(info, dict) else None
    except Exception:
        target = None
    return AliasState(alias=alias, target=target)


def _swing(client: MilvusClient, alias: str, target: str, dry_run: bool) -> None:
    """Atomic alias-target switch. Either creates the alias (if absent)
    or alter_alias's it. Both Milvus 2.6 operations are single
    metaserver writes — atomic from the client's perspective."""
    state = _current_alias(client, alias)
    if state.target == target:
        log.info("  alias %r already → %r (no-op)", alias, target)
        return
    if dry_run:
        log.info("  [dry-run] would swing alias %r: %r → %r",
                 alias, state.target, target)
        return
    if state.target is None:
        log.info("  creating alias %r → %r", alias, target)
        client.create_alias(collection_name=target, alias=alias)
    else:
        log.info("  swinging alias %r: %r → %r", alias, state.target, target)
        client.alter_alias(collection_name=target, alias=alias)


def _row_count(client: MilvusClient, collection: str) -> int:
    """Cheap-ish row count via stats. Newly-bulk-inserted rows are
    visible only after server-side flush — operators should run flush
    before swinging. Stats reflect sealed segments only."""
    stats = client.get_collection_stats(collection)
    if isinstance(stats, dict):
        rc = stats.get("row_count", 0)
    else:
        rc = stats
    return int(rc)


def _validate_join_key_consistency(
    client: MilvusClient,
    *,
    articles_collection: str,
    offers_collection: str,
    sample_size: int,
) -> None:
    """Pick `sample_size` random offers, verify each carries an
    `article_hash` that resolves in the target articles collection.
    Catches the F9 worst-case bug — bulk run wrote articles + offers
    from different DuckDB stream invocations and the hashes drifted —
    cheaper than scanning every row."""
    if sample_size <= 0:
        log.info("  skipping join-key check (sample_size=0)")
        return

    # Random sampling: query a small page of offers ordered by article_hash
    # at a random offset. Milvus query() supports OFFSET, so this is one
    # round-trip; no client-side sample.
    o_count = _row_count(client, offers_collection)
    if o_count == 0:
        log.warning("  offers collection is empty — skipping join-key check")
        return
    take = min(sample_size, o_count)
    offset = random.randint(0, max(o_count - take, 0))
    rows = client.query(
        collection_name=offers_collection,
        filter='id != ""',
        output_fields=["id", "article_hash"],
        limit=take,
        offset=offset,
    )
    hashes = sorted({r["article_hash"] for r in rows if r.get("article_hash")})
    if not hashes:
        log.warning("  sampled %d offers have no article_hash — skipping check", len(rows))
        return

    # Resolve those hashes against articles. `IN` clauses on Milvus
    # VARCHAR PKs are well-supported; the F9 PATH_B_HASH_LIMIT is much
    # larger than our sample so this fits in one query.
    quoted = ", ".join(f'"{h}"' for h in hashes)
    found = client.query(
        collection_name=articles_collection,
        filter=f"article_hash in [{quoted}]",
        output_fields=["article_hash"],
        limit=len(hashes),
    )
    found_hashes = {r["article_hash"] for r in found}
    missing = sorted(set(hashes) - found_hashes)
    if missing:
        sample = missing[:5]
        raise RuntimeError(
            f"join-key drift: {len(missing)}/{len(hashes)} sampled offer "
            f"article_hashes not found in {articles_collection!r} "
            f"(sample: {sample!r}). Re-run the bulk indexer to regenerate "
            "a consistent pair before swinging."
        )
    log.info("  join-key check OK: %d/%d sampled article_hashes resolve",
             len(hashes), len(hashes))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--milvus-uri", default="http://localhost:19530")
    p.add_argument("--articles-target", required=True,
                   help="Versioned name to point the articles alias at (e.g. articles_v4).")
    p.add_argument("--offers-target", required=True,
                   help="Versioned name to point the offers alias at (e.g. offers_v5).")
    p.add_argument("--articles-alias", default="articles",
                   help="Public articles alias name (default 'articles').")
    p.add_argument("--offers-alias", default="offers",
                   help="Public offers alias name (default 'offers').")

    p.add_argument("--no-validate", action="store_true",
                   help="Skip pre-flight validation. Only use if you've validated "
                        "by hand already — the validation catches half-populated "
                        "bulk runs and join-key drift.")
    p.add_argument("--min-rows-articles", type=int, default=1,
                   help="Minimum row count required in the articles target. "
                        "Operators should set this to expected production "
                        "scale (e.g. 100M+) once cutover is past the smoke phase.")
    p.add_argument("--min-rows-offers", type=int, default=1,
                   help="Minimum row count required in the offers target.")
    p.add_argument("--join-key-sample", type=int, default=200,
                   help="Random offers to sample for article_hash → articles "
                        "round-trip verification (default 200). 0 = skip.")

    p.add_argument("--rollback-to", default="",
                   help="Pair specifier 'articles_vM,offers_vN' to swing both "
                        "aliases back to. Skips validation + treats as recovery. "
                        "Use after a post-cutover incident; for a same-run "
                        "rollback the orchestrator does it automatically on "
                        "second-swing failure.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan + run validations without mutating aliases.")
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = MilvusClient(uri=args.milvus_uri)

    # --- rollback path: swing aliases back, no validation
    if args.rollback_to:
        try:
            articles_back, offers_back = [
                s.strip() for s in args.rollback_to.split(",")
            ]
        except ValueError:
            sys.exit("--rollback-to must be 'articles_vM,offers_vN'")
        log.warning("ROLLBACK: swinging articles → %r and offers → %r",
                    articles_back, offers_back)
        _swing(client, args.articles_alias, articles_back, args.dry_run)
        _swing(client, args.offers_alias, offers_back, args.dry_run)
        log.info("rollback done.")
        return

    log.info("Target pair: articles=%r, offers=%r", args.articles_target, args.offers_target)

    # --- existence checks (always — even with --no-validate)
    for name in (args.articles_target, args.offers_target):
        if not client.has_collection(name):
            sys.exit(
                f"Target collection {name!r} does not exist. "
                "Run scripts/create_*_collection.py + indexer_bulk.py first."
            )

    # --- validation
    if not args.no_validate:
        log.info("Pre-flight validation:")
        a_rows = _row_count(client, args.articles_target)
        o_rows = _row_count(client, args.offers_target)
        log.info("  rows: articles=%d, offers=%d", a_rows, o_rows)
        if a_rows < args.min_rows_articles:
            sys.exit(
                f"articles target {args.articles_target!r} has {a_rows} rows, "
                f"below --min-rows-articles={args.min_rows_articles}. "
                "Did the bulk indexer crash mid-run? Check the run logs + "
                "the checkpoint file."
            )
        if o_rows < args.min_rows_offers:
            sys.exit(
                f"offers target {args.offers_target!r} has {o_rows} rows, "
                f"below --min-rows-offers={args.min_rows_offers}."
            )
        _validate_join_key_consistency(
            client,
            articles_collection=args.articles_target,
            offers_collection=args.offers_target,
            sample_size=args.join_key_sample,
        )

    # --- capture rollback state BEFORE first swing
    prior_articles = _current_alias(client, args.articles_alias)
    prior_offers = _current_alias(client, args.offers_alias)
    log.info("Prior alias state (for rollback):")
    log.info("  %r → %r", prior_articles.alias, prior_articles.target or "(unset)")
    log.info("  %r → %r", prior_offers.alias, prior_offers.target or "(unset)")

    if args.dry_run:
        log.info("(dry-run — no swings performed)")
        return

    # --- swing articles first; offers last so a partial-failure window
    # leaves prior offers readable against new articles (orphan offer
    # rows would be the bigger user-facing problem).
    log.info("Swing 1/2: articles alias")
    _swing(client, args.articles_alias, args.articles_target, dry_run=False)

    log.info("Swing 2/2: offers alias")
    try:
        _swing(client, args.offers_alias, args.offers_target, dry_run=False)
    except Exception as e:
        log.error("Second swing FAILED: %s — rolling back first swing", e)
        if prior_articles.target is None:
            log.error("  rollback impossible: %r had no prior target. "
                      "You'll need to drop the alias manually.",
                      prior_articles.alias)
        else:
            try:
                _swing(client, prior_articles.alias, prior_articles.target, dry_run=False)
                log.warning("Rollback complete. System is back on the prior pair.")
            except Exception as rb_err:
                log.error(
                    "ROLLBACK ALSO FAILED (%s). Run manually: "
                    "MilvusClient.alter_alias(collection_name=%r, alias=%r)",
                    rb_err, prior_articles.target, prior_articles.alias,
                )
        raise

    log.info("Both aliases swung. Final state:")
    log.info("  %r → %r", args.articles_alias, args.articles_target)
    log.info("  %r → %r", args.offers_alias, args.offers_target)
    if prior_articles.target or prior_offers.target:
        log.info(
            "Rollback hint: re-run with `--rollback-to %s,%s` to revert.",
            prior_articles.target or args.articles_target,
            prior_offers.target or args.offers_target,
        )


if __name__ == "__main__":
    main()
