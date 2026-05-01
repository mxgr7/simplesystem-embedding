"""DuckDB SQL implementation of the F9 indexer projection + aggregation.

Pure SQL (with macros) — no Python UDFs. Three public entry points layer
on top of the same projection CTE:

    project_records(json_path) -> list[flat_row]
        — parity check vs `indexer.projection.project()`. Wrapper-JSON
          fixture format. See `tests/test_duckdb_projection_parity.py`.

    aggregate_articles(json_path) -> list[article_row]
        — one row per unique `article_hash`; matches
          `indexer.projection.aggregate_article(...)`.

    project_offer_rows(json_path) -> list[offer_row]
        — one row per offer; matches `indexer.projection.to_offer_row(...)`.

The two-stream variant — `project_two_streams` — runs the projection +
GROUP BY once and returns both result sets, the shape `indexer.bulk` uses
in production. Aggregate / offer parity lives in
`tests/test_duckdb_aggregate_parity.py`.

Input shape: the joined-record JSON file produced by
`scripts/dump_mongo_sample.js` (and `tests/fixtures/mongo_sample/sample_*.json`)
— top-level `{records: [{offer, pricings, markers, customerArticleNumbers}]}`.
The production raw-JSONL S3 path lives in `indexer.duckdb_raw_join`; it
adapts the 4 raw collections into the same `flat` shape this module
expects, then calls `_build_two_stream_sql()` directly.

Why DuckDB native instead of Python:
  - Built-in S3 + gzipped JSON reader.
  - Vectorized hash join → ~30-60% faster than Python bucketize.
  - Automatic disk-spill on aggregates, no manual bucket management.
  - ~400 lines SQL replaces ~600-800 lines of Python orchestration.

Cost: SQL of moderate complexity; team needs to be comfortable reading
DuckDB SQL idioms (LIST_TRANSFORM, FILTER aggregates, struct projection).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


# DuckDB SQL macros — reusable transforms that mirror helpers in
# `indexer.projection`. Loaded once per connection by `_init_macros`.
_MACROS_SQL = """
-- UUID decode from EJSON binary envelope: STRUCT("$binary" STRUCT(base64 VARCHAR, ...))
-- Returns canonical hyphenated UUID string. Mirrors `_decode_uuid`.
CREATE OR REPLACE MACRO uuid_str(b) AS
    CAST(CAST(from_base64(b['$binary']['base64']) AS UUID) AS VARCHAR);

-- Base64URL with no padding — used for the article_number portion of the PK.
-- Mirrors `_b64url_no_pad`. `encode()` returns the UTF-8 byte representation
-- as BLOB (vs `s::BLOB` which requires ASCII or hex-escaped); production
-- article_numbers contain non-ASCII characters like Ø in some catalogs.
CREATE OR REPLACE MACRO b64url_no_pad(s) AS
    rtrim(replace(replace(to_base64(encode(s)), '+', '-'), '/', '_'), '=');

-- Canonical category-path encoding with `¦` separator and `|` escape.
-- Mirrors `_encode_path` + `CategoryPath.java`.
CREATE OR REPLACE MACRO encode_category_path(elements) AS
    list_aggregate(
        list_transform(elements, x -> replace(x, '¦', '|')),
        'string_agg', '¦'
    );

-- Order-preserving distinct on a list. DuckDB's `list_distinct` /
-- `array_distinct` reorders (hash-based), which breaks parity with
-- Python's `if x not in bins: bins.append(x)` semantic. We zip with the
-- 1-based position, keep only elements whose first occurrence equals
-- their position, then unwrap. O(n²) per call — fine for the small lists
-- (≤5 paths, ≤16 eclass codes) where it's used.
CREATE OR REPLACE MACRO ordered_distinct(xs) AS (
    list_transform(
        list_filter(
            list_zip(xs, range(1, len(xs) + 1)),
            e -> list_indexof(xs, e[1]) = e[2]
        ),
        e -> e[1]
    )
);

-- Single-unit price: lowest minQuantity staggered.price / priceQuantity.
-- Returns DECIMAL(18,6). NULL when no staggered prices or all .price NULL.
-- Mirrors `_single_unit_price`. Implemented as list ops because DuckDB
-- macros cannot use `FROM unnest(macro_param.field)` — struct-field access
-- on a macro parameter is allowed inside list_filter/list_transform but
-- not as a table source.
CREATE OR REPLACE MACRO single_unit_price(pd) AS (
    list_extract(
        array_sort(
            list_transform(
                list_filter(
                    COALESCE(pd.prices.staggeredPrices,
                             []::STRUCT(minQuantity VARCHAR, price VARCHAR)[]),
                    s -> s.price IS NOT NULL
                ),
                -- Field order is load-bearing: array_sort uses lexicographic
                -- ordering, so `mq` first puts the smallest-minQuantity entry
                -- at index 1.
                s -> {
                    'mq': COALESCE(TRY_CAST(s.minQuantity AS DECIMAL(18,6)), 0::DECIMAL(18,6)),
                    'p':  TRY_CAST(s.price AS DECIMAL(18,6))
                }
            )
        ),
        1
    ).p
    /
    CASE
        WHEN pd.priceQuantity IS NULL
          OR TRY_CAST(pd.priceQuantity AS DECIMAL(18,6)) = 0
            THEN 1::DECIMAL(18,6)
        ELSE TRY_CAST(pd.priceQuantity AS DECIMAL(18,6))
    END
);

-- PricingType priority (legacy `commons/.../PricingType.java`). Lower = higher
-- query-time priority. Default OPEN(1) for unknown types.
CREATE OR REPLACE MACRO pricing_priority(type_name) AS
    CASE COALESCE(type_name, 'OPEN')
        WHEN 'OPEN' THEN 1
        WHEN 'CLOSED' THEN 2
        WHEN 'GROUP' THEN 3
        WHEN 'DEDICATED' THEN 4
        ELSE 1
    END;

-- Project one PricingDetails into the row's prices entry. Returns NULL
-- when no resolvable price. Mirrors `_project_one_pricing`.
CREATE OR REPLACE MACRO project_one_pricing(pd) AS
    CASE
        WHEN pd IS NULL THEN NULL
        WHEN single_unit_price(pd) IS NULL THEN NULL
        ELSE {
            'price': single_unit_price(pd)::DOUBLE,
            'currency': COALESCE(pd.prices.currencyCode, ''),
            'priority': pricing_priority(pd."type"),
            'sourcePriceListId': CASE
                WHEN pd.sourcePriceListId IS NULL THEN ''
                ELSE uuid_str(pd.sourcePriceListId)
            END
        }
    END;

-- Canonical-EJSON int unwrap. Handles both forms uniformly so the same
-- projection SQL feeds the wrapper-JSON path (numbers pre-stripped by
-- `dump_s3_sample.py:_to_relaxed_ejson`) and the raw-JSONL S3 path
-- (numbers wrapped as `{"$numberInt": "..."}` / `{"$numberLong": "..."}`).
-- Returns BIGINT or NULL — caller COALESCEs the default. NULL on any
-- input that's neither a plain JSON number nor a recognised wrapper
-- (e.g. JSON object without `$numberInt`); rare in prod data and
-- treated the same as the field being absent.
CREATE OR REPLACE MACRO unwrap_int(v) AS COALESCE(
    TRY_CAST(json_extract_string(v::JSON, '$."$numberInt"') AS BIGINT),
    TRY_CAST(json_extract_string(v::JSON, '$."$numberLong"') AS BIGINT),
    TRY_CAST(v::JSON AS BIGINT)
);

-- F9 article-hash. Mirrors `compute_article_hash` in indexer.projection.
-- sha256 of the canonical embedded-field tuple (name, manufacturerName,
-- 5 category levels, 3 eclass arrays), separated by US/RS control chars
-- that never appear in legitimate user text. Arrays sorted to make hash
-- order-independent; `substr(.., 1, 32)` gives the first 16 bytes hex.
-- DuckDB's `sha256()` returns a 64-char lowercase hex VARCHAR — no
-- to_hex() needed (an extra to_hex() would double-encode).
CREATE OR REPLACE MACRO compute_article_hash(
    a_name, a_mfg, c1, c2, c3, c4, c5, e5, e7, s2
) AS
    substr(
        sha256(
            COALESCE(a_name, '') || E'\\x1f' ||
            COALESCE(a_mfg, '') || E'\\x1f' ||
            array_to_string(array_sort(COALESCE(c1, []::VARCHAR[])), E'\\x1e') || E'\\x1f' ||
            array_to_string(array_sort(COALESCE(c2, []::VARCHAR[])), E'\\x1e') || E'\\x1f' ||
            array_to_string(array_sort(COALESCE(c3, []::VARCHAR[])), E'\\x1e') || E'\\x1f' ||
            array_to_string(array_sort(COALESCE(c4, []::VARCHAR[])), E'\\x1e') || E'\\x1f' ||
            array_to_string(array_sort(COALESCE(c5, []::VARCHAR[])), E'\\x1e') || E'\\x1f' ||
            array_to_string(list_transform(array_sort(COALESCE(e5, []::INTEGER[])), x -> x::VARCHAR), E'\\x1e') || E'\\x1f' ||
            array_to_string(list_transform(array_sort(COALESCE(e7, []::INTEGER[])), x -> x::VARCHAR), E'\\x1e') || E'\\x1f' ||
            array_to_string(list_transform(array_sort(COALESCE(s2, []::INTEGER[])), x -> x::VARCHAR), E'\\x1e')
        ),
        1, 32
    );
"""

# Mirrors `indexer.projection.MAX_PRICE_SENTINEL` — Milvus 2.6 rejects
# NaN and ±Inf on FLOAT scalars, so a large finite value substitutes for
# "no price in this currency on this offer/article". Range predicates
# (`{ccy}_price_min <= X`) and `ORDER BY {ccy}_price_min ASC` both
# behave correctly on the sentinel.
MAX_PRICE_SENTINEL = 3.4028234e38

# Catalog currencies that get per-currency envelope columns. Mirror of
# `indexer.projection.CATALOG_CURRENCIES` — the same constant in three
# places (projection.py + this file + the create_*_collection.py
# scripts), kept in sync by `test_catalog_currencies_match_script`.
CATALOG_CURRENCIES = ("eur", "chf", "huf", "pln", "gbp", "czk", "cny")


def _init_macros(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_MACROS_SQL)


# Explicit `records` element schema. Auto-detect picks the right shape on
# rich samples but degrades to `JSON[]` on sparse fields (an empty
# `features: []` across every sampled row → DuckDB has no struct shape to
# infer, falls back to JSON, and the SQL fails to bind `f.name`). Pinning
# the schema makes both the test fixtures and the production S3 reads
# behave identically. Mirrors the joined-record shape produced by
# `dump_mongo_sample.js` / `scripts/dump_s3_sample.py`.
# Building-block sub-schemas. Reused by both the wrapper-JSON record
# type below and the per-collection schemas in `_collection_schemas` for
# raw-JSONL reads.
_BINARY_TYPE = 'STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR))'
_OID_TYPE = 'STRUCT("$oid" VARCHAR)'
_PRICING_DETAILS_TYPE = (
    f'STRUCT(sourcePriceListId {_BINARY_TYPE}, "type" VARCHAR, '
    'prices STRUCT(staggeredPrices STRUCT(minQuantity VARCHAR, price VARCHAR)[], currencyCode VARCHAR), '
    'dailyPrice BOOLEAN, priceQuantity VARCHAR)'
)
_OFFER_PARAMS_TYPE = (
    'STRUCT('
    '"name" VARCHAR, '
    'features STRUCT("name" VARCHAR, "values" VARCHAR[], unit VARCHAR, description VARCHAR)[], '
    'eclassGroups JSON, '
    'categoryPaths STRUCT(elements VARCHAR[])[], '
    'ean VARCHAR, '
    'manufacturerName VARCHAR, '
    'customerArticleNumber VARCHAR, '
    # `deliveryTime` is JSON to absorb both raw EJSON-canonical
    # `{"$numberInt": "5"}` and the wrapper-JSON pre-stripped plain int.
    # The `unwrap_int` SQL macro normalises both into BIGINT downstream.
    'deliveryTime JSON'
    ')'
)
_INNER_OFFER_TYPE = (
    'STRUCT('
    f'_id {_BINARY_TYPE}, '
    f'catalogVersionId {_BINARY_TYPE}, '
    f'offerParams {_OFFER_PARAMS_TYPE}, '
    f'pricings STRUCT(open {_PRICING_DETAILS_TYPE}, closed {_PRICING_DETAILS_TYPE}), '
    'relatedArticleNumbers STRUCT(sparePartFor VARCHAR[], accessoryFor VARCHAR[], similarTo VARCHAR[])'
    ')'
)
_OUTER_OFFER_TYPE = (
    # `_id` (`$oid`) and `importEpoch` are present in the source data
    # (raw EJSON wraps `importEpoch` as `{"$numberLong": "..."}`) but
    # not used by the projection. Omitting them from the schema makes
    # DuckDB skip the fields entirely on read — same effect as parsing
    # them and discarding, with one less column to materialise.
    'STRUCT('
    'articleNumber VARCHAR, '
    f'vendorId {_BINARY_TYPE}, '
    f'catalogVersionId {_BINARY_TYPE}, '
    f'offer {_INNER_OFFER_TYPE}'
    ')'
)
_PRICING_ROW_TYPE = (
    'STRUCT('
    'articleNumber VARCHAR, '
    f'vendorId {_BINARY_TYPE}, '
    f'pricingDetails {_PRICING_DETAILS_TYPE}'
    ')'
)
_MARKER_ROW_TYPE = (
    'STRUCT('
    'articleNumber VARCHAR, '
    f'vendorId {_BINARY_TYPE}, '
    f'coreArticleListSourceId {_BINARY_TYPE}, '
    'coreArticleMarker BOOLEAN'
    ')'
)
_CAN_ROW_TYPE = (
    'STRUCT('
    'articleNumber VARCHAR, '
    f'vendorId {_BINARY_TYPE}, '
    f'customerArticleNumbersListVersionId {_BINARY_TYPE}, '
    'customerArticleNumber VARCHAR'
    ')'
)

# Wrapper-JSON record schema — one element of `records[]`.
RECORDS_ELEMENT_TYPE = (
    'STRUCT('
    f'offer {_OUTER_OFFER_TYPE}, '
    f'pricings {_PRICING_ROW_TYPE}[], '
    f'markers {_MARKER_ROW_TYPE}[], '
    f'customerArticleNumbers {_CAN_ROW_TYPE}[]'
    ')'
)


# Raw-JSONL per-collection schemas. Mongo Atlas snapshot exports each
# collection as one JSON document per line (`atlas-fkxrb3-shard-N.M.json.gz`);
# the schema below pins the column shape for `read_json(..., format='newline_delimited')`
# so DuckDB doesn't auto-degrade rare/sparse fields to `JSON[]` (same fix
# applied to the wrapper-JSON path; see `RECORDS_ELEMENT_TYPE` comment).
# `_id` and `importEpoch` (`$numberLong`-wrapped in raw EJSON) are
# absent from these schemas because the projection doesn't read them —
# DuckDB skips unknown JSON fields on read.
RAW_OFFER_COLUMNS = {
    "articleNumber": "VARCHAR",
    "vendorId": _BINARY_TYPE,
    "catalogVersionId": _BINARY_TYPE,
    "offer": _INNER_OFFER_TYPE,
}
RAW_PRICING_COLUMNS = {
    "articleNumber": "VARCHAR",
    "priceListId": _BINARY_TYPE,
    "vendorId": _BINARY_TYPE,
    "pricingDetails": _PRICING_DETAILS_TYPE,
}
RAW_MARKER_COLUMNS = {
    "articleNumber": "VARCHAR",
    "coreArticleListSourceId": _BINARY_TYPE,
    "vendorId": _BINARY_TYPE,
    "coreArticleMarker": "BOOLEAN",
}
RAW_CAN_COLUMNS = {
    "articleNumber": "VARCHAR",
    "customerArticleNumbersListVersionId": _BINARY_TYPE,
    "vendorId": _BINARY_TYPE,
    "customerArticleNumber": "VARCHAR",
}


# SQL fragment that turns the wrapper-JSON `raw` table (one row, with a
# `records` array column) into the `flat` working set. The raw-JSONL S3
# path uses `_FLAT_FROM_RAW_COLLECTIONS_SQL` instead.
_FLAT_FROM_WRAPPER_SQL = """
unnested AS (
    SELECT unnest(records) AS rec FROM raw
),
flat AS (
    SELECT
        rec.offer AS outer_offer,
        rec.offer.offer AS inner_offer,
        rec.offer.offer.offerParams AS params,
        rec.pricings AS joined_pricings,
        rec.markers AS joined_markers,
        rec.customerArticleNumbers AS joined_cans
    FROM unnested
)"""


# Raw-JSONL path: JOIN the 4 source tables (`raw_offers`, `raw_pricings`,
# `raw_markers`, `raw_cans`) on `(vendorId.$binary.base64, articleNumber)`
# to produce the same `flat` shape as the wrapper-JSON path. Each side
# pre-groups its rows into a list keyed by the join tuple so the LEFT
# JOINs against `raw_offers` stay 1:1 — the fan-out goes into the list
# columns (`joined_pricings`, etc.), exactly mirroring the wrapper-JSON
# record shape. The base64 form of `vendorId.$binary.base64` is used
# verbatim as the join key (bit-equal to the raw UUID bytes; avoids
# per-row UUID decode for the hash).
#
# `articleNumber` collation: VARCHAR equality. Production data has
# article numbers with embedded slashes / spaces / unicode — DuckDB's
# default VARCHAR equality is byte-exact, which matches the legacy
# Mongo string equality.
# Standalone SELECT statements for pre-materialising the grouped tables.
# Each runs as its own CREATE TABLE so the aggregator state from one
# doesn't pile onto the next — DuckDB releases the per-CTAS memory
# between calls. With 1.2B pricings GROUP BY (vendor, article), holding
# the aggregator state in-memory inside the same query as the downstream
# JOIN was OOM'ing at 100+ GB; splitting it lets each step use its full
# memory budget independently and frees `raw_pricings` between them.
_PRICINGS_GROUPED_SELECT_SQL = """
SELECT
    vendorId."$binary".base64 AS vk,
    articleNumber AS ak,
    list(struct_pack(
        articleNumber := articleNumber,
        vendorId := vendorId,
        pricingDetails := pricingDetails
    )) AS pricings_list
FROM raw_pricings
GROUP BY vendorId."$binary".base64, articleNumber
"""

_MARKERS_GROUPED_SELECT_SQL = """
SELECT
    vendorId."$binary".base64 AS vk,
    articleNumber AS ak,
    list(struct_pack(
        articleNumber := articleNumber,
        vendorId := vendorId,
        coreArticleListSourceId := coreArticleListSourceId,
        coreArticleMarker := coreArticleMarker
    )) AS markers_list
FROM raw_markers
GROUP BY vendorId."$binary".base64, articleNumber
"""

_CANS_GROUPED_SELECT_SQL = """
SELECT
    vendorId."$binary".base64 AS vk,
    articleNumber AS ak,
    list(struct_pack(
        articleNumber := articleNumber,
        vendorId := vendorId,
        customerArticleNumbersListVersionId := customerArticleNumbersListVersionId,
        customerArticleNumber := customerArticleNumber
    )) AS cans_list
FROM raw_cans
GROUP BY vendorId."$binary".base64, articleNumber
"""

# `flat` CTE that JOINs raw_offers against the pre-materialised
# {pricings,markers,cans}_grouped tables. No inline aggregation —
# downstream queries that use this assume `materialise_grouped_tables(con)`
# has already run.
_FLAT_FROM_GROUPED_TABLES_SQL = """
flat AS (
    SELECT
        struct_pack(
            articleNumber := o.articleNumber,
            vendorId := o.vendorId,
            catalogVersionId := o.catalogVersionId,
            "offer" := o."offer"
        ) AS outer_offer,
        o."offer" AS inner_offer,
        o."offer".offerParams AS params,
        COALESCE(p.pricings_list, []) AS joined_pricings,
        COALESCE(m.markers_list, []) AS joined_markers,
        COALESCE(c.cans_list, []) AS joined_cans
    FROM raw_offers o
    LEFT JOIN pricings_grouped p
        ON p.vk = o.vendorId."$binary".base64 AND p.ak = o.articleNumber
    LEFT JOIN markers_grouped m
        ON m.vk = o.vendorId."$binary".base64 AND m.ak = o.articleNumber
    LEFT JOIN cans_grouped c
        ON c.vk = o.vendorId."$binary".base64 AND c.ak = o.articleNumber
)"""

# Original inline-CTE form. Used only by the legacy single-CTAS path
# (kept for parity tests + the wrapper-JSON path). Production
# (`indexer.bulk._materialise_streams`) calls `materialise_grouped_tables`
# first and then uses `_FLAT_FROM_GROUPED_TABLES_SQL` instead.
_FLAT_FROM_RAW_COLLECTIONS_SQL = f"""
pricings_grouped AS ({_PRICINGS_GROUPED_SELECT_SQL.strip()}),
markers_grouped AS ({_MARKERS_GROUPED_SELECT_SQL.strip()}),
cans_grouped AS ({_CANS_GROUPED_SELECT_SQL.strip()}),
{_FLAT_FROM_GROUPED_TABLES_SQL.lstrip()}"""


# Per-offer projection CTE (`projected` + `finalized`). Reads from a
# `flat` table — see the wrapper-JSON / raw-JSONL composers below. One
# row per offer; column shape matches `indexer.projection.project()`.
_PROJECTION_CTE_SQL = """
projected AS (
    SELECT
        -- ---- PK + identifiers ----
        uuid_str(outer_offer.vendorId) AS vendor_id,
        outer_offer.articleNumber AS article_number,
        uuid_str(outer_offer.vendorId) || ':' || b64url_no_pad(outer_offer.articleNumber) AS id,

        -- ---- article-level retrieval / display ----
        COALESCE(params."name", '') AS "name",
        COALESCE(params.manufacturerName, '') AS manufacturerName,
        COALESCE(params.ean, '') AS ean,

        -- ---- catalog version ----
        CASE
            WHEN outer_offer.catalogVersionId IS NULL THEN []::VARCHAR[]
            ELSE [uuid_str(outer_offer.catalogVersionId)]
        END AS catalog_version_ids,

        -- ---- delivery ----
        -- `unwrap_int` handles both raw EJSON `{"$numberInt": "5"}` and
        -- the wrapper-JSON pre-stripped plain int.
        CAST(COALESCE(unwrap_int(params.deliveryTime), 0) AS INTEGER) AS delivery_time_days_max,

        -- ---- eClass / S2Class hierarchies ----
        -- Use JSON-path access on the struct cast: DuckDB's auto-detected
        -- schema only includes fields seen in the sample, so a sample with
        -- no S2CLASS rows fails compile if we use `.S2CLASS`. Round-tripping
        -- through JSON makes missing keys NULL-safe at the cost of one
        -- cast per row (negligible for this column count). Each element
        -- goes through `unwrap_int` because raw S3 wraps the array
        -- elements (`[{"$numberInt": "27270911"}, ...]`) while the
        -- wrapper-JSON pre-strips them (`[27270911, ...]`).
        COALESCE(
            list_transform(
                json_extract(params.eclassGroups::JSON, '$.ECLASS_5_1')::JSON[],
                j -> CAST(unwrap_int(j) AS INTEGER)
            ),
            []::INTEGER[]
        ) AS eclass5_code,
        COALESCE(
            list_transform(
                json_extract(params.eclassGroups::JSON, '$.ECLASS_7_1')::JSON[],
                j -> CAST(unwrap_int(j) AS INTEGER)
            ),
            []::INTEGER[]
        ) AS eclass7_code,
        COALESCE(
            list_transform(
                json_extract(params.eclassGroups::JSON, '$.S2CLASS')::JSON[],
                j -> CAST(unwrap_int(j) AS INTEGER)
            ),
            []::INTEGER[]
        ) AS s2class_code,

        -- ---- relationships ----
        -- Schema is pinned to VARCHAR[] for all three (see RECORDS_ELEMENT_TYPE).
        -- Truncate at 4096 = Milvus 2.6's hard Array max_capacity ceiling.
        -- Empirically `accessoryFor` hits 3175 in a 10-shard prod sample
        -- and almost certainly exceeds 4096 at full catalog scope on the
        -- pathological tail. Truncation here keeps the bulk_insert path
        -- crash-free; relationship-search recall on those rows is
        -- bounded but the schema cap is the binding constraint anyway.
        list_slice(COALESCE(inner_offer.relatedArticleNumbers.accessoryFor, []::VARCHAR[]), 1, 4096) AS relationship_accessory_for,
        list_slice(COALESCE(inner_offer.relatedArticleNumbers.sparePartFor, []::VARCHAR[]), 1, 4096) AS relationship_spare_part_for,
        list_slice(COALESCE(inner_offer.relatedArticleNumbers.similarTo,    []::VARCHAR[]), 1, 4096) AS relationship_similar_to,

        -- ---- categories: emit one entry per depth (1..5), de-duped ----
        -- For each path, produce its prefix paths up to min(len, 5).
        -- Group by depth, distinct.
        -- Per-depth prefix list, dedupe-preserving-source-order to match
        -- Python's `if encoded not in bins[depth-1]: bins[depth-1].append(...)`
        -- semantic. `list_distinct` keeps the first occurrence; default to
        -- `[]` not NULL when no path reaches the depth.
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(params.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 1
            ),
            cp -> encode_category_path(cp.elements[1:1])
        )), []::VARCHAR[]) AS category_l1,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(params.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 2
            ),
            cp -> encode_category_path(cp.elements[1:2])
        )), []::VARCHAR[]) AS category_l2,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(params.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 3
            ),
            cp -> encode_category_path(cp.elements[1:3])
        )), []::VARCHAR[]) AS category_l3,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(params.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 4
            ),
            cp -> encode_category_path(cp.elements[1:4])
        )), []::VARCHAR[]) AS category_l4,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(params.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 5
            ),
            cp -> encode_category_path(cp.elements[1:5])
        )), []::VARCHAR[]) AS category_l5,

        -- ---- features: name=value tokens, drop entries with `=` in value ----
        -- Order: outer feature × inner value, source order preserved.
        -- `flatten(list_transform(features, f -> list_transform(...)))` is
        -- the order-preserving idiom; cross-joining unnest() rearranges.
        flatten(list_transform(
            COALESCE(params.features,
                     []::STRUCT("name" VARCHAR, "values" VARCHAR[], unit VARCHAR, description VARCHAR)[]),
            f -> list_transform(
                list_filter(COALESCE(f."values", []::VARCHAR[]), v -> NOT contains(v, '=')),
                v -> f."name" || '=' || v
            )
        )) AS features,

        -- ---- prices: embedded (open + closed) + joined pricings collection ----
        -- Returns list of {price, currency, priority, sourcePriceListId} structs;
        -- NULL entries (no resolvable price) are filtered.
        list_concat(
            COALESCE([project_one_pricing(inner_offer.pricings.open)], []),
            COALESCE([project_one_pricing(inner_offer.pricings.closed)], []),
            COALESCE(
                (SELECT list(project_one_pricing(jp.pricingDetails))
                 FROM unnest(COALESCE(joined_pricings, []::STRUCT(_id STRUCT("$oid" VARCHAR), articleNumber VARCHAR, priceListId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), importEpoch BIGINT, pricingDetails STRUCT(sourcePriceListId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), "type" VARCHAR, prices STRUCT(staggeredPrices STRUCT(minQuantity VARCHAR, price VARCHAR)[], currencyCode VARCHAR), dailyPrice BOOLEAN, priceQuantity VARCHAR))[])) AS t(jp)),
                []
            )
        ) AS prices_raw,

        -- ---- markers: split into enabled/disabled, de-duped, source order preserved ----
        (
            SELECT list(distinct uuid_str(m.coreArticleListSourceId))
            FROM unnest(COALESCE(joined_markers, []::STRUCT(_id STRUCT("$oid" VARCHAR), articleNumber VARCHAR, coreArticleListSourceId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), coreArticleMarker BOOLEAN)[])) AS t(m)
            WHERE m.coreArticleListSourceId IS NOT NULL AND m.coreArticleMarker = true
        ) AS core_marker_enabled_sources_raw,
        (
            SELECT list(distinct uuid_str(m.coreArticleListSourceId))
            FROM unnest(COALESCE(joined_markers, []::STRUCT(_id STRUCT("$oid" VARCHAR), articleNumber VARCHAR, coreArticleListSourceId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), coreArticleMarker BOOLEAN)[])) AS t(m)
            WHERE m.coreArticleListSourceId IS NOT NULL AND m.coreArticleMarker = false
        ) AS core_marker_disabled_sources_raw,

        -- ---- customer_article_numbers: UNION joined + catalog-supplied, group by value ----
        -- Two sources fold together: joined customerArticleNumbers rows (with
        -- their listVersionId) + the offer's catalog-supplied customerArticleNumber
        -- (paired with outer.catalogVersionId). Group by value, sort version_ids.
        (
            WITH all_pairs AS (
                SELECT can.customerArticleNumber AS value, uuid_str(can.customerArticleNumbersListVersionId) AS version_id
                FROM unnest(COALESCE(joined_cans, []::STRUCT(_id STRUCT("$oid" VARCHAR), articleNumber VARCHAR, customerArticleNumbersListVersionId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), customerArticleNumber VARCHAR)[])) AS t(can)
                WHERE can.customerArticleNumber IS NOT NULL
                  AND can.customerArticleNumber <> ''
                  AND can.customerArticleNumbersListVersionId IS NOT NULL
                UNION ALL
                SELECT params.customerArticleNumber, uuid_str(outer_offer.catalogVersionId)
                WHERE params.customerArticleNumber IS NOT NULL
                  AND params.customerArticleNumber <> ''
                  AND outer_offer.catalogVersionId IS NOT NULL
            )
            SELECT list({'value': value, 'version_ids': version_ids} ORDER BY value)
            FROM (
                SELECT value, list(distinct version_id ORDER BY version_id) AS version_ids
                FROM all_pairs
                GROUP BY value
            )
        ) AS customer_article_numbers
    FROM flat
)"""


# Finalized CTE — coalesces nullables + filters NULL price entries. Reads
# from any `projected` CTE that emits the canonical column shape (works
# with `_PROJECTION_CTE_SQL` and `_PROJECTION_FROM_OP_CTE_SQL` alike).
_FINALIZED_CTE_SQL = """
finalized AS (
    SELECT
        id, "name", manufacturerName, ean, article_number, vendor_id,
        catalog_version_ids,
        category_l1, category_l2, category_l3, category_l4, category_l5,
        -- Drop NULL price entries (no resolvable single_unit_price).
        list_filter(prices_raw, p -> p IS NOT NULL) AS prices,
        delivery_time_days_max,
        COALESCE(core_marker_enabled_sources_raw, []::VARCHAR[]) AS core_marker_enabled_sources,
        COALESCE(core_marker_disabled_sources_raw, []::VARCHAR[]) AS core_marker_disabled_sources,
        eclass5_code, eclass7_code, s2class_code,
        COALESCE(features, []::VARCHAR[]) AS features,
        relationship_accessory_for, relationship_spare_part_for, relationship_similar_to,
        COALESCE(customer_article_numbers, []::STRUCT(value VARCHAR, version_ids VARCHAR[])[]) AS customer_article_numbers
    FROM projected
)"""


# F8 / F9 article aggregation CTE. Reads from `with_hash` (= finalized
# rows + computed `article_hash` column). Emits one row per unique hash:
# article-level scalars (invariant across the dedup group) + text_codes
# (BM25 corpus) + customer_article_numbers UNION + per-currency envelope.
# Mirrors `indexer.projection.aggregate_article(...)`.
_ARTICLES_CTE_SQL = f"""
-- Per-row prices exploded for the article-level envelope. Filter to
-- catalog currencies only (lowercased) — non-catalog currencies have no
-- envelope column to land in. Done outside the main aggregation so the
-- per-row offer count is preserved for the article-level any_value()
-- columns.
article_prices_exploded AS (
    SELECT
        article_hash,
        lower(p.currency) AS ccy,
        CAST(p.price AS DOUBLE) AS price
    FROM with_hash, unnest(prices) AS t(p)
    WHERE p.currency IS NOT NULL
      AND lower(p.currency) IN ({", ".join(f"'{c}'" for c in CATALOG_CURRENCIES)})
),
article_ccy_envelope AS (
    SELECT
        article_hash,
        {",\n        ".join(
            f"MIN(price) FILTER (WHERE ccy = '{c}') AS {c}_price_min,\n        "
            f"MAX(price) FILTER (WHERE ccy = '{c}') AS {c}_price_max"
            for c in CATALOG_CURRENCIES
        )}
    FROM article_prices_exploded
    GROUP BY article_hash
),
-- customer_article_numbers UNION across the dedup group, keyed by value.
-- Two offers under the same hash that expose "BOLT-001" under different
-- list versions land as one entry whose version_ids covers both.
article_cans_pairs AS (
    SELECT
        article_hash,
        can.value AS value,
        version_id
    FROM with_hash,
         unnest(customer_article_numbers) AS t(can),
         unnest(can.version_ids) AS u(version_id)
    WHERE can.value IS NOT NULL AND can.value <> ''
),
article_cans_per_value AS (
    SELECT
        article_hash,
        value,
        list(distinct version_id ORDER BY version_id) AS version_ids
    FROM article_cans_pairs
    GROUP BY article_hash, value
),
article_cans AS (
    SELECT
        article_hash,
        list({{'value': value, 'version_ids': version_ids}} ORDER BY value) AS customer_article_numbers
    FROM article_cans_per_value
    GROUP BY article_hash
),
-- Article-level invariants + text_codes (BM25 corpus). Pulled from any
-- single offer per hash (the embedded fields are the hash inputs, so
-- they're identical within the group). EANs and article_numbers are
-- per-offer — collected as sorted-distinct unions for the corpus.
article_base AS (
    SELECT
        article_hash,
        any_value("name") AS "name",
        any_value(manufacturerName) AS manufacturerName,
        any_value(category_l1) AS category_l1,
        any_value(category_l2) AS category_l2,
        any_value(category_l3) AS category_l3,
        any_value(category_l4) AS category_l4,
        any_value(category_l5) AS category_l5,
        any_value(eclass5_code) AS eclass5_code,
        any_value(eclass7_code) AS eclass7_code,
        any_value(s2class_code) AS s2class_code,
        -- text_codes: name + manufacturerName + sorted distinct EANs +
        -- sorted distinct article_numbers, joined by single space, empty
        -- segments dropped (matches `_canon` and the Python aggregator's
        -- `" ".join(p for p in [...] if p)`).
        --
        -- Truncated to 8192 chars to fit Milvus collection schema's
        -- VARCHAR(8192) cap. Production data has rare articles whose
        -- aggregated EAN/article_number unions blow past 8 KB; Milvus
        -- bulk_insert rejects the whole batch on first overflow row.
        -- Truncate to safe length; BM25 already de-prioritises long
        -- token salads so the tail has near-zero IDF weight anyway.
        substr(
            array_to_string(
                list_filter(
                    list_concat(
                        CASE WHEN any_value("name") IS NOT NULL AND any_value("name") <> ''
                             THEN [any_value("name")] ELSE []::VARCHAR[] END,
                        CASE WHEN any_value(manufacturerName) IS NOT NULL AND any_value(manufacturerName) <> ''
                             THEN [any_value(manufacturerName)] ELSE []::VARCHAR[] END,
                        COALESCE(array_sort(list(distinct ean) FILTER (WHERE ean IS NOT NULL AND ean <> '')), []::VARCHAR[]),
                        COALESCE(array_sort(list(distinct article_number) FILTER (WHERE article_number IS NOT NULL AND article_number <> '')), []::VARCHAR[])
                    ),
                    x -> x IS NOT NULL AND x <> ''
                ),
                ' '
            ),
            1, 8192
        ) AS text_codes
    FROM with_hash
    GROUP BY article_hash
),
articles AS (
    SELECT
        ab.article_hash,
        ab."name", ab.manufacturerName,
        ab.category_l1, ab.category_l2, ab.category_l3, ab.category_l4, ab.category_l5,
        ab.eclass5_code, ab.eclass7_code, ab.s2class_code,
        ab.text_codes,
        COALESCE(ac.customer_article_numbers,
                 []::STRUCT(value VARCHAR, version_ids VARCHAR[])[]) AS customer_article_numbers,
        {",\n        ".join(
            f"COALESCE(ae.{c}_price_min, {MAX_PRICE_SENTINEL!r}::DOUBLE) AS {c}_price_min,\n        "
            f"COALESCE(ae.{c}_price_max, {-MAX_PRICE_SENTINEL!r}::DOUBLE) AS {c}_price_max"
            for c in CATALOG_CURRENCIES
        )}
    FROM article_base ab
    LEFT JOIN article_cans ac USING (article_hash)
    LEFT JOIN article_ccy_envelope ae USING (article_hash)
)"""


# F8 per-offer envelope CTE. Reads from `with_hash` (= finalized rows +
# article_hash). Emits one row per offer matching `to_offer_row(...)`.
_OFFERS_CTE_SQL = f"""
offer_prices_exploded AS (
    SELECT
        id,
        lower(p.currency) AS ccy,
        CAST(p.price AS DOUBLE) AS price
    FROM with_hash, unnest(prices) AS t(p)
    WHERE p.currency IS NOT NULL
      AND lower(p.currency) IN ({", ".join(f"'{c}'" for c in CATALOG_CURRENCIES)})
),
offer_ccy_envelope AS (
    SELECT
        id,
        {",\n        ".join(
            f"MIN(price) FILTER (WHERE ccy = '{c}') AS {c}_price_min,\n        "
            f"MAX(price) FILTER (WHERE ccy = '{c}') AS {c}_price_max"
            for c in CATALOG_CURRENCIES
        )}
    FROM offer_prices_exploded
    GROUP BY id
),
offers_pre_dedup AS (
    SELECT
        wh.id,
        wh.article_hash,
        -- Milvus 2.6 requires every collection to declare at least one
        -- vector field. Path B never searches `offers_v{{N}}` — only
        -- `query()` on filter expressions — so the offer collection
        -- carries a 2-dim FLOAT placeholder (see
        -- `scripts/create_offers_collection.py`).
        [0.0, 0.0]::FLOAT[] AS _placeholder_vector,
        wh.ean, wh.article_number, wh.vendor_id, wh.catalog_version_ids,
        wh.prices, wh.delivery_time_days_max,
        wh.core_marker_enabled_sources, wh.core_marker_disabled_sources,
        -- eclass / s2class / categories / customer_article_numbers are
        -- article-level (they're the inputs to `compute_article_hash`,
        -- so they're invariant within the dedup group). Live on
        -- `articles_v{{N}}` only. Mirrors `_ARTICLE_LEVEL_KEYS` filter
        -- in `indexer.projection.to_offer_row`.
        wh.features,
        wh.relationship_accessory_for, wh.relationship_spare_part_for, wh.relationship_similar_to,
        -- F8 per-offer envelope inputs.
        -- price_list_ids = sorted distinct sourcePriceListId across this
        -- offer's prices (drop NULLs and empty strings).
        COALESCE(
            list_sort(list_distinct(list_filter(
                list_transform(wh.prices, p -> p.sourcePriceListId),
                x -> x IS NOT NULL AND x <> ''
            ))),
            []::VARCHAR[]
        ) AS price_list_ids,
        -- currencies = sorted distinct lowercased currency across all
        -- prices (NOT restricted to the catalog set — narrow filters on
        -- rare currencies still need this column to drop the row).
        COALESCE(
            list_sort(list_distinct(list_filter(
                list_transform(wh.prices, p -> lower(p.currency)),
                x -> x IS NOT NULL AND x <> ''
            ))),
            []::VARCHAR[]
        ) AS currencies,
        {",\n        ".join(
            f"COALESCE(oe.{c}_price_min, {MAX_PRICE_SENTINEL!r}::DOUBLE) AS {c}_price_min,\n        "
            f"COALESCE(oe.{c}_price_max, {-MAX_PRICE_SENTINEL!r}::DOUBLE) AS {c}_price_max"
            for c in CATALOG_CURRENCIES
        )}
    FROM with_hash wh
    LEFT JOIN offer_ccy_envelope oe USING (id)
),
-- Atlas snapshots can carry duplicate (vendorId, articleNumber) tuples
-- for the same offer when multiple shards or staging variants overlap.
-- Milvus rejects upserts that contain duplicate primary keys within a
-- single batch, so we collapse to one row per `id` here. The choice of
-- representative is non-deterministic (no `updated_at` column on the
-- offer schema); for two snapshots taken simultaneously this is
-- well-defined enough — operators wanting "latest" semantics need to
-- carry an explicit timestamp through. Per-row parity (sample_10k)
-- explicitly tested via multi-set comparison since the dedup happens
-- at this CTE boundary, not in the upstream projection.
offers AS (
    SELECT * EXCLUDE (_rn) FROM (
        SELECT *, row_number() OVER (PARTITION BY id) AS _rn
        FROM offers_pre_dedup
    )
    WHERE _rn = 1
)"""


# Hash CTE — adds `article_hash` to each finalized row. Sits between
# `finalized` and the article/offer CTEs. Defined here so all consumers
# pick up the same column shape.
_WITH_HASH_CTE_SQL = """
with_hash AS (
    SELECT
        f.*,
        compute_article_hash(
            "name", manufacturerName,
            category_l1, category_l2, category_l3, category_l4, category_l5,
            eclass5_code, eclass7_code, s2class_code
        ) AS article_hash
    FROM finalized f
)"""


# Offer-derived precomputation. Reads `raw_offers` directly (no JOINs to
# pricings/markers/cans), produces a wide per-offer row that contains
# every column that does NOT depend on the JOIN. Includes:
#   - join keys (vk, ak) for downstream chunk-time JOINs
#   - article_hash (computable from offer-only fields)
#   - the raw inline `pricings.{open,closed}` STRUCTs (carry-through; the
#     chunk-time projection passes them through `project_one_pricing`
#     alongside the joined pricings)
#   - inline_can_pair: the offer-inline customerArticleNumber + its
#     catalogVersionId (carry-through; UNIONed at chunk-time with joined
#     cans)
#
# Build once via `scripts/build_offer_projected.py`, write to
# `offer_projected.parquet/chunk_KKKK.parquet`. Per-chunk indexer reads
# the chunk's parquet and skips the heavy `_PROJECTION_CTE_SQL` work.
_OFFER_PROJECTED_BUILD_SELECT_SQL = """
WITH offer_derived AS (
    SELECT
        o.vendorId."$binary".base64 AS vk,
        o.articleNumber AS ak,

        -- ---- PK + identifiers ----
        uuid_str(o.vendorId) AS vendor_id,
        o.articleNumber AS article_number,
        uuid_str(o.vendorId) || ':' || b64url_no_pad(o.articleNumber) AS id,

        -- ---- article-level retrieval / display ----
        COALESCE(o."offer".offerParams."name", '') AS "name",
        COALESCE(o."offer".offerParams.manufacturerName, '') AS manufacturerName,
        COALESCE(o."offer".offerParams.ean, '') AS ean,

        -- ---- catalog version ----
        CASE
            WHEN o.catalogVersionId IS NULL THEN []::VARCHAR[]
            ELSE [uuid_str(o.catalogVersionId)]
        END AS catalog_version_ids,

        -- ---- delivery ----
        CAST(COALESCE(unwrap_int(o."offer".offerParams.deliveryTime), 0) AS INTEGER) AS delivery_time_days_max,

        -- ---- eClass / S2Class hierarchies ----
        COALESCE(
            list_transform(
                json_extract(o."offer".offerParams.eclassGroups::JSON, '$.ECLASS_5_1')::JSON[],
                j -> CAST(unwrap_int(j) AS INTEGER)
            ),
            []::INTEGER[]
        ) AS eclass5_code,
        COALESCE(
            list_transform(
                json_extract(o."offer".offerParams.eclassGroups::JSON, '$.ECLASS_7_1')::JSON[],
                j -> CAST(unwrap_int(j) AS INTEGER)
            ),
            []::INTEGER[]
        ) AS eclass7_code,
        COALESCE(
            list_transform(
                json_extract(o."offer".offerParams.eclassGroups::JSON, '$.S2CLASS')::JSON[],
                j -> CAST(unwrap_int(j) AS INTEGER)
            ),
            []::INTEGER[]
        ) AS s2class_code,

        -- ---- relationships ----
        list_slice(COALESCE(o."offer".relatedArticleNumbers.accessoryFor, []::VARCHAR[]), 1, 4096) AS relationship_accessory_for,
        list_slice(COALESCE(o."offer".relatedArticleNumbers.sparePartFor, []::VARCHAR[]), 1, 4096) AS relationship_spare_part_for,
        list_slice(COALESCE(o."offer".relatedArticleNumbers.similarTo,    []::VARCHAR[]), 1, 4096) AS relationship_similar_to,

        -- ---- categories: per-depth dedup ----
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(o."offer".offerParams.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 1
            ),
            cp -> encode_category_path(cp.elements[1:1])
        )), []::VARCHAR[]) AS category_l1,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(o."offer".offerParams.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 2
            ),
            cp -> encode_category_path(cp.elements[1:2])
        )), []::VARCHAR[]) AS category_l2,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(o."offer".offerParams.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 3
            ),
            cp -> encode_category_path(cp.elements[1:3])
        )), []::VARCHAR[]) AS category_l3,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(o."offer".offerParams.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 4
            ),
            cp -> encode_category_path(cp.elements[1:4])
        )), []::VARCHAR[]) AS category_l4,
        COALESCE(ordered_distinct(list_transform(
            list_filter(
                COALESCE(o."offer".offerParams.categoryPaths, []::STRUCT(elements VARCHAR[])[]),
                cp -> len(cp.elements) >= 5
            ),
            cp -> encode_category_path(cp.elements[1:5])
        )), []::VARCHAR[]) AS category_l5,

        -- ---- features ----
        COALESCE(flatten(list_transform(
            COALESCE(o."offer".offerParams.features,
                     []::STRUCT("name" VARCHAR, "values" VARCHAR[], unit VARCHAR, description VARCHAR)[]),
            f -> list_transform(
                list_filter(COALESCE(f."values", []::VARCHAR[]), v -> NOT contains(v, '=')),
                v -> f."name" || '=' || v
            )
        )), []::VARCHAR[]) AS features,

        -- ---- carry-through: raw inline pricings (used at chunk-time) ----
        o."offer".pricings.open AS inline_pricings_open,
        o."offer".pricings.closed AS inline_pricings_closed,

        -- ---- carry-through: offer-inline can pair (UNIONed at chunk-time
        -- with joined cans) ----
        CASE
            WHEN o."offer".offerParams.customerArticleNumber IS NOT NULL
             AND o."offer".offerParams.customerArticleNumber <> ''
             AND o.catalogVersionId IS NOT NULL
            THEN {
                'value': o."offer".offerParams.customerArticleNumber,
                'version_id': uuid_str(o.catalogVersionId)
            }
            ELSE NULL
        END AS inline_can_pair
    FROM raw_offers o
)
SELECT
    *,
    -- article_hash references the per-row computed columns above. Computed
    -- in a second SELECT layer so the column references are valid.
    compute_article_hash(
        "name", manufacturerName,
        category_l1, category_l2, category_l3, category_l4, category_l5,
        eclass5_code, eclass7_code, s2class_code
    ) AS article_hash
FROM offer_derived
"""


# Chunk-time projection that reuses precomputed offer_projected. Drop-in
# replacement for `_PROJECTION_CTE_SQL` when offer_projected is available.
# Reads from a table named `offer_projected` (operator's responsibility to
# create) and JOINs to chunk-local pricings_grouped / markers_grouped /
# cans_grouped to compute the 4 JOIN-dependent columns:
#     prices_raw, core_marker_enabled_sources_raw,
#     core_marker_disabled_sources_raw, customer_article_numbers
# Output column shape matches `_PROJECTION_CTE_SQL` exactly so downstream
# `finalized` + `with_hash` + article/offer CTEs are unchanged.
_PROJECTION_FROM_OP_CTE_SQL = """
projected AS (
    SELECT
        op.vendor_id, op.article_number, op.id,
        op."name", op.manufacturerName, op.ean,
        op.catalog_version_ids, op.delivery_time_days_max,
        op.eclass5_code, op.eclass7_code, op.s2class_code,
        op.relationship_accessory_for, op.relationship_spare_part_for, op.relationship_similar_to,
        op.category_l1, op.category_l2, op.category_l3, op.category_l4, op.category_l5,
        op.features,

        -- prices_raw: concat offer-inline (open + closed) + joined pricings
        list_concat(
            COALESCE([project_one_pricing(op.inline_pricings_open)], []),
            COALESCE([project_one_pricing(op.inline_pricings_closed)], []),
            COALESCE(
                (SELECT list(project_one_pricing(jp.pricingDetails))
                 FROM unnest(COALESCE(p.pricings_list, []::STRUCT(articleNumber VARCHAR, vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), pricingDetails STRUCT(sourcePriceListId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), "type" VARCHAR, prices STRUCT(staggeredPrices STRUCT(minQuantity VARCHAR, price VARCHAR)[], currencyCode VARCHAR), dailyPrice BOOLEAN, priceQuantity VARCHAR))[])) AS t(jp)),
                []
            )
        ) AS prices_raw,

        -- core_marker_enabled / disabled sources from joined markers
        (SELECT list(distinct uuid_str(mk.coreArticleListSourceId))
         FROM unnest(COALESCE(m.markers_list, []::STRUCT(articleNumber VARCHAR, vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), coreArticleListSourceId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), coreArticleMarker BOOLEAN)[])) AS t(mk)
         WHERE mk.coreArticleListSourceId IS NOT NULL AND mk.coreArticleMarker = true
        ) AS core_marker_enabled_sources_raw,
        (SELECT list(distinct uuid_str(mk.coreArticleListSourceId))
         FROM unnest(COALESCE(m.markers_list, []::STRUCT(articleNumber VARCHAR, vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), coreArticleListSourceId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), coreArticleMarker BOOLEAN)[])) AS t(mk)
         WHERE mk.coreArticleListSourceId IS NOT NULL AND mk.coreArticleMarker = false
        ) AS core_marker_disabled_sources_raw,

        -- customer_article_numbers: UNION offer-inline (carried as
        -- op.inline_can_pair) with joined cans, group by value.
        (
            WITH all_pairs AS (
                SELECT can.customerArticleNumber AS value, uuid_str(can.customerArticleNumbersListVersionId) AS version_id
                FROM unnest(COALESCE(c.cans_list, []::STRUCT(articleNumber VARCHAR, vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), customerArticleNumbersListVersionId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), customerArticleNumber VARCHAR)[])) AS t(can)
                WHERE can.customerArticleNumber IS NOT NULL
                  AND can.customerArticleNumber <> ''
                  AND can.customerArticleNumbersListVersionId IS NOT NULL
                UNION ALL
                SELECT op.inline_can_pair.value, op.inline_can_pair.version_id
                WHERE op.inline_can_pair IS NOT NULL
            )
            SELECT list({'value': value, 'version_ids': version_ids} ORDER BY value)
            FROM (
                SELECT value, list(distinct version_id ORDER BY version_id) AS version_ids
                FROM all_pairs
                GROUP BY value
            )
        ) AS customer_article_numbers
    FROM offer_projected op
    LEFT JOIN pricings_grouped p ON p.vk = op.vk AND p.ak = op.ak
    LEFT JOIN markers_grouped m ON m.vk = op.vk AND m.ak = op.ak
    LEFT JOIN cans_grouped c ON c.vk = op.vk AND c.ak = op.ak
)"""


def offer_projected_build_sql(*, source_table_or_glob: str) -> str:
    """Render the SELECT that produces offer_projected output.

    `source_table_or_glob` is either:
      - a DuckDB table name (e.g. 'raw_offers'), or
      - a parquet path/glob (e.g. '/path/to/offers.parquet/chunk_0000.parquet'),
        in which case it's wrapped in `read_parquet(...)`.
    """
    if source_table_or_glob.endswith(".parquet") or "*.parquet" in source_table_or_glob:
        from_clause = f"read_parquet('{source_table_or_glob}') o"
    else:
        from_clause = f"{source_table_or_glob} o"
    # Substitute `raw_offers o` with the chosen FROM clause.
    return _OFFER_PROJECTED_BUILD_SELECT_SQL.replace("FROM raw_offers o", f"FROM {from_clause}")


def _resolve_flat_cte(source: str) -> str:
    """`source` ∈ {'wrapper', 'raw', 'raw_grouped', 'op_grouped'}.

    - 'wrapper': single-record fixture path.
    - 'raw': inline aggregator + JOIN — heavy memory, kept for parity tests.
    - 'raw_grouped': assumes pricings_grouped/markers_grouped/cans_grouped
      already exist as tables (call `materialise_grouped_tables(con)` first).
      Production path — drops the inline aggregator's hash-table memory.
    - 'op_grouped': assumes a precomputed `offer_projected` table exists
      (loaded from `offer_projected.parquet/chunk_K.parquet`) plus the
      chunk-local pricings_grouped/markers_grouped/cans_grouped. Skips
      the entire `flat` JOIN + heavy `_PROJECTION_CTE_SQL` work."""
    if source == "wrapper":
        return _FLAT_FROM_WRAPPER_SQL
    if source == "raw":
        return _FLAT_FROM_RAW_COLLECTIONS_SQL
    if source == "raw_grouped":
        return _FLAT_FROM_GROUPED_TABLES_SQL
    if source == "op_grouped":
        return ""  # no flat CTE needed; projected reads from offer_projected directly
    raise ValueError(f"unknown source: {source}")


def _resolve_projection_cte(source: str) -> str:
    """Pick the right `projected` CTE for the given source mode."""
    if source == "op_grouped":
        return _PROJECTION_FROM_OP_CTE_SQL
    return _PROJECTION_CTE_SQL


def _resolve_with_hash_cte(source: str) -> str:
    """Pick the right `with_hash` CTE. We always recompute article_hash
    from finalized's columns rather than JOINing back to offer_projected
    by id — the per-row sha256 (~500 MB/s/core) is much cheaper than a
    3M×3M hash join, and finalized already carries every input the
    macro needs (name, mfg, categories, eclass codes)."""
    return _WITH_HASH_CTE_SQL


def _build_flat_rows_sql(*, source: str = "wrapper") -> str:
    """Emit projected flat rows. See `_resolve_flat_cte` for `source` modes."""
    flat_cte = _resolve_flat_cte(source)
    projection_cte = _resolve_projection_cte(source)
    return f"""
WITH {(flat_cte + ',') if flat_cte else ''}
{projection_cte.lstrip()}
SELECT * FROM finalized;
"""


def _build_articles_sql(*, source: str = "wrapper") -> str:
    """Emit one article row per unique hash."""
    flat_cte = _resolve_flat_cte(source)
    projection_cte = _resolve_projection_cte(source)
    with_hash_cte = _resolve_with_hash_cte(source)
    return f"""
WITH {(flat_cte + ',') if flat_cte else ''}
{projection_cte.lstrip()},
{_FINALIZED_CTE_SQL.lstrip()},
{with_hash_cte.lstrip()},
{_ARTICLES_CTE_SQL.lstrip()}
SELECT * FROM articles;
"""


def _build_offers_sql(*, source: str = "wrapper") -> str:
    """Emit one offer row per finalized projected row."""
    flat_cte = _resolve_flat_cte(source)
    projection_cte = _resolve_projection_cte(source)
    with_hash_cte = _resolve_with_hash_cte(source)
    return f"""
WITH {(flat_cte + ',') if flat_cte else ''}
{projection_cte.lstrip()},
{_FINALIZED_CTE_SQL.lstrip()},
{with_hash_cte.lstrip()},
{_OFFERS_CTE_SQL.lstrip()}
SELECT * FROM offers;
"""


def materialise_grouped_tables(con) -> tuple[int, int, int]:
    """Pre-materialise pricings_grouped, markers_grouped, cans_grouped as
    actual tables (not inline CTEs). Returns the row counts.

    Each CTAS runs separately so the per-aggregator hash-table state is
    freed before the next runs. Drops the corresponding raw_* table after
    its grouped table is built — for pricings this saves ~30 GB of
    memory + spill since raw_pricings (1.2B rows) is no longer needed
    once pricings_grouped (159M rows) exists."""
    import logging
    log = logging.getLogger("indexer.duckdb")

    log.info("Materialising pricings_grouped from raw_pricings (1.2B rows → ~159M)…")
    con.execute(f"CREATE OR REPLACE TABLE pricings_grouped AS {_PRICINGS_GROUPED_SELECT_SQL.strip()}")
    p_n = con.execute("SELECT count(*) FROM pricings_grouped").fetchone()[0]
    log.info("  pricings_grouped: %d rows", p_n)
    con.execute("DROP TABLE raw_pricings")

    log.info("Materialising markers_grouped from raw_markers…")
    con.execute(f"CREATE OR REPLACE TABLE markers_grouped AS {_MARKERS_GROUPED_SELECT_SQL.strip()}")
    m_n = con.execute("SELECT count(*) FROM markers_grouped").fetchone()[0]
    log.info("  markers_grouped: %d rows", m_n)
    con.execute("DROP TABLE raw_markers")

    log.info("Materialising cans_grouped from raw_cans…")
    con.execute(f"CREATE OR REPLACE TABLE cans_grouped AS {_CANS_GROUPED_SELECT_SQL.strip()}")
    c_n = con.execute("SELECT count(*) FROM cans_grouped").fetchone()[0]
    log.info("  cans_grouped: %d rows", c_n)
    con.execute("DROP TABLE raw_cans")

    return p_n, m_n, c_n


def load_raw_collections(
    con: duckdb.DuckDBPyConnection,
    *,
    offers_glob: str,
    pricings_glob: str,
    markers_glob: str,
    cans_glob: str,
    maximum_object_size: int = 256 * 1024 * 1024,
) -> None:
    """Read the 4 source collections into DuckDB tables. Each `*_glob`
    can be a local path glob (`/path/atlas-*.json.gz`) or an S3 URL
    pattern (`s3://bucket/.../atlas-*.json.gz`); DuckDB's `read_json`
    handles both transparently when `httpfs` is loaded with a
    `credential_chain` secret (set up by the caller). Files are gzipped
    JSONL — one document per line.

    `maximum_object_size` bumps DuckDB's per-line buffer limit for the
    rare offer documents with very long descriptions or feature lists.
    256 MB is well above any plausible Mongo doc size."""
    schemas = [
        ("raw_offers",   offers_glob,   RAW_OFFER_COLUMNS),
        ("raw_pricings", pricings_glob, RAW_PRICING_COLUMNS),
        ("raw_markers",  markers_glob,  RAW_MARKER_COLUMNS),
        ("raw_cans",     cans_glob,     RAW_CAN_COLUMNS),
    ]
    for table_name, glob, columns in schemas:
        # Auto-dispatch on file extension. Parquet is preferred for
        # production runs — DuckDB parses parquet column-parallel and
        # zero-copy, vs JSON which is per-file single-threaded. With 4116
        # pricings shards on a shared NVMe, parquet cuts raw-load wall
        # from ~25 min (gzip + JSON parse, disk-I/O bound) to ~3 min.
        # Parity: column shape is preserved across both formats because
        # the parquet was originally written from the JSON source via
        # `scripts/convert_source_to_parquet.py` with the same column set.
        if ".parquet" in glob:
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                "SELECT * FROM read_parquet(?)",
                [glob],
            )
        else:
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                "SELECT * FROM read_json(?, format='newline_delimited', "
                "maximum_object_size=?, columns=?)",
                [glob, maximum_object_size, columns],
            )


def _load_wrapper_fixture(con: duckdb.DuckDBPyConnection, json_path: Path | str) -> None:
    """Load a wrapper-JSON fixture into the `raw` table. Explicit schema
    (vs auto-detect) keeps sparse/empty fields from degrading to `JSON[]`.
    `maximum_object_size` bumps the 16 MB default to handle sample_10k
    (~33 MB on one line); production raw-JSONL reads stream line-by-line
    where the default suffices."""
    con.execute(
        "CREATE OR REPLACE TABLE raw AS SELECT * FROM read_json(?, format='auto', maximum_object_size=?, columns=?)",
        [
            str(json_path),
            256 * 1024 * 1024,
            {"generated_at": "VARCHAR", "sample_size": "BIGINT", "records": f"{RECORDS_ELEMENT_TYPE}[]"},
        ],
    )


def _fetchall_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    rows = con.execute(sql).fetchall()
    columns = [d[0] for d in con.description]
    return [dict(zip(columns, row)) for row in rows]


def project_records(json_path: Path | str) -> list[dict[str, Any]]:
    """Run the DuckDB SQL projection over the joined-records JSON file at
    `json_path`. Returns one dict per record, keys/values matching
    `indexer.projection.project(record).row` (modulo dict-vs-list ordering
    on JSON-typed fields, which the parity test canonicalises before
    comparing)."""
    con = duckdb.connect()
    _init_macros(con)
    _load_wrapper_fixture(con, json_path)
    return _fetchall_dicts(con, _build_flat_rows_sql())


def aggregate_articles(json_path: Path | str) -> list[dict[str, Any]]:
    """One article row per unique `article_hash` from the joined-records
    JSON fixture. Output dicts match
    `indexer.projection.aggregate_article(group)` for the corresponding
    Python-grouped flat rows (modulo numeric / sentinel rounding handled
    by the parity test canonicalisation)."""
    con = duckdb.connect()
    _init_macros(con)
    _load_wrapper_fixture(con, json_path)
    return _fetchall_dicts(con, _build_articles_sql())


def project_offer_rows(json_path: Path | str) -> list[dict[str, Any]]:
    """One offer row per finalized projected row, with F8 envelope
    columns. Output dicts match
    `indexer.projection.to_offer_row(row, article_hash=…)` for the
    corresponding Python row (`_placeholder_vector` included)."""
    con = duckdb.connect()
    _init_macros(con)
    _load_wrapper_fixture(con, json_path)
    return _fetchall_dicts(con, _build_offers_sql())


def aggregate_articles_from_collections(
    con: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyRelation:
    """Production article-row stream from the 4 raw collections that
    `load_raw_collections` populated. Returns a DuckDB relation — caller
    iterates with `.fetchmany()` or `.fetch_arrow_reader()` to keep
    memory bounded at S3 scale.

    Caller must invoke `_init_macros(con)` and `load_raw_collections(...)`
    first."""
    return con.sql(_build_articles_sql(source="raw"))


def project_offer_rows_from_collections(
    con: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyRelation:
    """Production offer-row stream from the 4 raw collections."""
    return con.sql(_build_offers_sql(source="raw"))


def init_macros(con: duckdb.DuckDBPyConnection) -> None:
    """Public alias of the macro initialiser — callers that drive the
    DuckDB connection themselves (e.g. `indexer.bulk`) need to install
    macros before running any of the SQL builders."""
    _init_macros(con)


__all__ = [
    "project_records",
    "aggregate_articles",
    "project_offer_rows",
    "load_raw_collections",
    "aggregate_articles_from_collections",
    "project_offer_rows_from_collections",
    "init_macros",
    "CATALOG_CURRENCIES",
    "MAX_PRICE_SENTINEL",
    "RAW_OFFER_COLUMNS",
    "RAW_PRICING_COLUMNS",
    "RAW_MARKER_COLUMNS",
    "RAW_CAN_COLUMNS",
]
