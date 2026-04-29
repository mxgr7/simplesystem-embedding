"""DuckDB SQL implementation of `indexer.projection.project()`.

Pure SQL (with macros) — no Python UDFs. Validates the F9 design
question "can the bulk indexer be DuckDB-native instead of a Python
bucketize+join orchestrator?" with a byte-for-byte parity check vs the
existing Python projection (`tests/test_duckdb_projection_parity.py`).

Input shape: the joined-record JSON file produced by
`scripts/dump_mongo_sample.js` (and `tests/fixtures/mongo_sample/sample_*.json`)
— top-level `{records: [{offer, pricings, markers, customerArticleNumbers}]}`.

Output shape: list of dicts whose keys + values match
`indexer.projection.project(record).row` exactly. Field-level diffs in the
parity test point at parity bugs in this module.

Why DuckDB native instead of Python:
  - Built-in S3 + gzipped JSON reader.
  - Vectorized hash join → ~30-60% faster than Python bucketize.
  - Automatic disk-spill on aggregates, no manual bucket management.
  - ~150 lines SQL replaces ~600-800 lines of Python orchestration.

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
-- Mirrors `_b64url_no_pad`.
CREATE OR REPLACE MACRO b64url_no_pad(s) AS
    rtrim(replace(replace(to_base64(s::BLOB), '+', '-'), '/', '_'), '=');

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
"""


def _init_macros(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_MACROS_SQL)


# Explicit `records` element schema. Auto-detect picks the right shape on
# rich samples but degrades to `JSON[]` on sparse fields (an empty
# `features: []` across every sampled row → DuckDB has no struct shape to
# infer, falls back to JSON, and the SQL fails to bind `f.name`). Pinning
# the schema makes both the test fixtures and the production S3 reads
# behave identically. Mirrors the joined-record shape produced by
# `dump_mongo_sample.js` / `scripts/dump_s3_sample.py`.
RECORDS_ELEMENT_TYPE = """STRUCT(
    offer STRUCT(
        _id STRUCT("$oid" VARCHAR),
        articleNumber VARCHAR,
        vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        catalogVersionId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        importEpoch BIGINT,
        offer STRUCT(
            _id STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
            catalogVersionId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
            offerParams STRUCT(
                "name" VARCHAR,
                features STRUCT("name" VARCHAR, "values" VARCHAR[], unit VARCHAR, description VARCHAR)[],
                eclassGroups JSON,
                categoryPaths STRUCT(elements VARCHAR[])[],
                ean VARCHAR,
                manufacturerName VARCHAR,
                customerArticleNumber VARCHAR,
                deliveryTime BIGINT
            ),
            pricings STRUCT(
                open    STRUCT(sourcePriceListId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), "type" VARCHAR, prices STRUCT(staggeredPrices STRUCT(minQuantity VARCHAR, price VARCHAR)[], currencyCode VARCHAR), dailyPrice BOOLEAN, priceQuantity VARCHAR),
                closed  STRUCT(sourcePriceListId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)), "type" VARCHAR, prices STRUCT(staggeredPrices STRUCT(minQuantity VARCHAR, price VARCHAR)[], currencyCode VARCHAR), dailyPrice BOOLEAN, priceQuantity VARCHAR)
            ),
            relatedArticleNumbers STRUCT(
                sparePartFor   VARCHAR[],
                accessoryFor   VARCHAR[],
                similarTo      VARCHAR[]
            )
        )
    ),
    pricings STRUCT(
        articleNumber VARCHAR,
        vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        pricingDetails STRUCT(
            sourcePriceListId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
            "type" VARCHAR,
            prices STRUCT(staggeredPrices STRUCT(minQuantity VARCHAR, price VARCHAR)[], currencyCode VARCHAR),
            dailyPrice BOOLEAN,
            priceQuantity VARCHAR
        )
    )[],
    markers STRUCT(
        articleNumber VARCHAR,
        vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        coreArticleListSourceId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        coreArticleMarker BOOLEAN
    )[],
    customerArticleNumbers STRUCT(
        articleNumber VARCHAR,
        vendorId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        customerArticleNumbersListVersionId STRUCT("$binary" STRUCT(base64 VARCHAR, subType VARCHAR)),
        customerArticleNumber VARCHAR
    )[]
)"""


def _build_projection_sql() -> str:
    """The full per-record projection. One row per joined record (UNNEST
    of the `records` array). Mirrors the keys + values produced by
    `indexer.projection.project(record).row`. The F8 per-offer envelope
    columns + per-article aggregates land later (in `to_offer_row` /
    `aggregate_article` SQL — not part of the parity check at this layer)."""

    return """
WITH unnested AS (
    SELECT
        unnest(records) AS rec
    FROM raw
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
),
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
        CAST(COALESCE(params.deliveryTime, 0) AS INTEGER) AS delivery_time_days_max,

        -- ---- eClass / S2Class hierarchies ----
        -- Use JSON-path access on the struct cast: DuckDB's auto-detected
        -- schema only includes fields seen in the sample, so a sample with
        -- no S2CLASS rows fails compile if we use `.S2CLASS`. Round-tripping
        -- through JSON makes missing keys NULL-safe at the cost of one
        -- cast per row (negligible for this column count).
        list_transform(
            COALESCE(json_extract(params.eclassGroups::JSON, '$.ECLASS_5_1')::BIGINT[], []::BIGINT[]),
            x -> CAST(x AS INTEGER)
        ) AS eclass5_code,
        list_transform(
            COALESCE(json_extract(params.eclassGroups::JSON, '$.ECLASS_7_1')::BIGINT[], []::BIGINT[]),
            x -> CAST(x AS INTEGER)
        ) AS eclass7_code,
        list_transform(
            COALESCE(json_extract(params.eclassGroups::JSON, '$.S2CLASS')::BIGINT[], []::BIGINT[]),
            x -> CAST(x AS INTEGER)
        ) AS s2class_code,

        -- ---- relationships ----
        -- Schema is pinned to VARCHAR[] for all three (see RECORDS_ELEMENT_TYPE).
        COALESCE(inner_offer.relatedArticleNumbers.accessoryFor, []::VARCHAR[]) AS relationship_accessory_for,
        COALESCE(inner_offer.relatedArticleNumbers.sparePartFor, []::VARCHAR[]) AS relationship_spare_part_for,
        COALESCE(inner_offer.relatedArticleNumbers.similarTo, []::VARCHAR[]) AS relationship_similar_to,

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
),
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
)
SELECT * FROM finalized;
"""


def project_records(json_path: Path | str) -> list[dict[str, Any]]:
    """Run the DuckDB SQL projection over the joined-records JSON file at
    `json_path`. Returns one dict per record, keys/values matching
    `indexer.projection.project(record).row` (modulo dict-vs-list ordering
    on JSON-typed fields, which the parity test canonicalises before
    comparing)."""
    con = duckdb.connect()
    _init_macros(con)
    # Explicit `records` schema (vs auto-detect) so empty/sparse fields
    # don't degrade to JSON[]. `maximum_object_size` bumps the per-line
    # wrapper {"records": [...]} limit (16 MB default) for sample_10k
    # (~33 MB); production reads stream JSONL line-by-line where the
    # default suffices.
    con.execute(
        "CREATE TABLE raw AS SELECT * FROM read_json(?, format='auto', maximum_object_size=?, columns=?)",
        [
            str(json_path),
            256 * 1024 * 1024,
            {"generated_at": "VARCHAR", "sample_size": "BIGINT", "records": f"{RECORDS_ELEMENT_TYPE}[]"},
        ],
    )
    sql = _build_projection_sql()
    rows = con.execute(sql).fetchall()
    columns = [d[0] for d in con.description]
    return [dict(zip(columns, row)) for row in rows]


__all__ = ["project_records"]
