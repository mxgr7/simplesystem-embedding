"""Raw-JSONL JOIN pipeline smoke test.

Validates the production path: read 4 raw collections from local-cached
S3 shards, JOIN on `(vendorId, articleNumber)`, project + aggregate. The
wrapper-JSON projection parity check (`test_duckdb_projection_parity.py`)
proves the projection SQL is correct; this test proves the raw-JSONL
JOIN that feeds it produces a sane shape.

Skipped if `~/s3-cache/` doesn't have the expected shards; run
`scripts/dump_s3_sample.py` (or the matching `aws s3 cp` recipe) to
populate it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.duckdb_projection import (  # noqa: E402
    aggregate_articles_from_collections,
    init_macros,
    load_raw_collections,
    project_offer_rows_from_collections,
)

CACHE_ROOT = Path.home() / "s3-cache"
OFFERS_GLOB    = str(CACHE_ROOT / "offers/atlas-fkxrb3-shard-0.0.json.gz")
PRICINGS_GLOB  = str(CACHE_ROOT / "pricings/atlas-*.json.gz")
MARKERS_GLOB   = str(CACHE_ROOT / "coreArticleMarkers/atlas-*.json.gz")
CANS_GLOB      = str(CACHE_ROOT / "customerArticleNumbers/atlas-*.json.gz")


def _required_shards_present() -> bool:
    return (
        (CACHE_ROOT / "offers/atlas-fkxrb3-shard-0.0.json.gz").exists()
        and any((CACHE_ROOT / "pricings").glob("atlas-*.json.gz"))
        and any((CACHE_ROOT / "coreArticleMarkers").glob("atlas-*.json.gz"))
        and any((CACHE_ROOT / "customerArticleNumbers").glob("atlas-*.json.gz"))
    )


pytestmark = pytest.mark.skipif(
    not _required_shards_present(),
    reason="s3-cache fixtures not present (run scripts/dump_s3_sample.py first)",
)


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    init_macros(c)
    load_raw_collections(
        c,
        offers_glob=OFFERS_GLOB,
        pricings_glob=PRICINGS_GLOB,
        markers_glob=MARKERS_GLOB,
        cans_glob=CANS_GLOB,
    )
    return c


def test_raw_collections_loaded(con: duckdb.DuckDBPyConnection) -> None:
    """Each collection should land non-empty rows. Sanity floor — these
    should comfortably exceed 1k each on shard 0.0 + the sample of
    pricings/markers/cans shards."""
    counts = {
        t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("raw_offers", "raw_pricings", "raw_markers", "raw_cans")
    }
    for t, n in counts.items():
        assert n > 1000, f"{t} has {n} rows (expected >1000)"


def test_articles_have_envelope_columns(con: duckdb.DuckDBPyConnection) -> None:
    """One article row per unique hash; per-currency envelope columns
    populated for at least eur (every shard has EUR-priced offers)."""
    arts = aggregate_articles_from_collections(con)
    con.execute(f"CREATE OR REPLACE TABLE articles AS {arts.sql_query()}")
    n_articles = con.execute("SELECT count(*) FROM articles").fetchone()[0]
    assert n_articles > 1000, f"only {n_articles} articles produced"

    # Articles dedup: should be < offers row count (raw_offers has joins
    # that fan out, but distinct (vendor, article) tuples are smaller).
    n_offers = con.execute("SELECT count(*) FROM raw_offers").fetchone()[0]
    assert n_articles <= n_offers, "more articles than offers — dedup broken"

    # At least some real EUR envelopes (not just sentinels everywhere).
    n_real_eur = con.execute(
        "SELECT count(*) FROM articles WHERE eur_price_min < 1e30"
    ).fetchone()[0]
    assert n_real_eur > 100, f"only {n_real_eur} articles with real EUR envelope"


def test_offer_rows_have_join_key_and_envelope(con: duckdb.DuckDBPyConnection) -> None:
    """One offer row per source offer, each with `article_hash` and the
    F8 per-offer envelope columns."""
    offs = project_offer_rows_from_collections(con)
    con.execute(f"CREATE OR REPLACE TABLE offer_rows AS {offs.sql_query()}")
    n_rows = con.execute("SELECT count(*) FROM offer_rows").fetchone()[0]
    n_with_hash = con.execute(
        "SELECT count(*) FROM offer_rows WHERE article_hash IS NOT NULL AND length(article_hash) = 32"
    ).fetchone()[0]
    assert n_with_hash == n_rows, (
        f"{n_rows - n_with_hash} offer rows missing a 32-char article_hash"
    )

    # Most offers should have at least one resolvable price (a small
    # sliver of expired/null entries is fine, but not the majority).
    n_with_prices = con.execute(
        "SELECT count(*) FROM offer_rows WHERE len(prices) > 0"
    ).fetchone()[0]
    assert n_with_prices / n_rows > 0.5, (
        f"only {n_with_prices}/{n_rows} offers have resolvable prices"
    )
