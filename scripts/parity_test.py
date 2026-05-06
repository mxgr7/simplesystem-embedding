#!/usr/bin/env python3
"""Comprehensive parity comparison: legacy (localhost:8081) vs ACL (localhost:8018).

Exercises every filter, sort, summary, search mode, error case, and edge case
from the spec. Outputs a structured report of discrepancies.

Accepted deviations (not tested — see PARITY_CHECKLIST.md for details):
- Sort tiebreak: legacy ES uses document insertion order (non-deterministic);
  ACL uses deterministic friendlyId-based tiebreak. Full article sets are
  identical — only page boundaries differ. (spec §2)
- Text search summaries: ANN returns ~200 candidates regardless of relevance,
  so summary counts differ from legacy's BM25-precise result set. (spec §2)
- Invalid currency: legacy crashes (500); ACL validates (400). ACL is correct.

Legacy limitations discovered during testing:
- PLATFORM_CATEGORIES summary crashes legacy (missing internal SummaryKind enum)
- eClassVersion only accepts S2CLASS (not ECLASS_5_1 or ECLASS_7_1)
- customerUploadedCoreArticleListSourceIds is accepted but may crash some paths
"""

import json
import sys
import httpx
from copy import deepcopy

LEGACY_URL = "http://localhost:8081"
ACL_URL = "http://localhost:8018"
TIMEOUT = 30.0

ALL_CV_IDS = [
    "04d60f27-adff-46ea-94a9-9d7749b45e49",
    "23bae76a-9815-45c4-9160-0485551c9bde",
    "308752b2-d2df-477e-9b3b-3ae04bbf9d16",
    "357a5946-1269-4d46-b9bc-a41ddd9f2be8",
    "583beea0-31bd-4b1d-a122-76273cb9269e",
    "9253df7a-542f-42c2-bcec-0fcbb232be56",
    "96f5f6fa-e6a8-4edc-9328-d4815d263eb4",
    "a4c649fc-84ad-45d4-9297-d5a6ee1841d5",
    "ac48e1c4-5024-48c1-a9a1-50d8fe0178ff",
    "c538e4d7-b0e2-4d72-91fd-4009789e1752",
]

ALL_PRICE_LIST_IDS = [
    "010effa4-53d6-422e-82ce-99fb7c0ef340",
    "013d6936-4ab3-4b98-a214-175a22a51d9b",
    "143573b8-2753-4247-973e-d3e2aa62cdd2",
    "5c5618ae-eb92-48f8-ad11-1f5067b19a33",
    "73879943-e519-403b-af24-4db2097b327d",
    "85603f29-3aff-411c-9268-6bc87dceed66",
    "a199b0d7-8f88-4281-8b30-b01f3ebca98d",
    "ae2fd9e3-5d34-473f-909e-df4ac9b79270",
    "b6642be6-2285-497b-aa23-4c0b15c527fa",
    "bbe7260d-0d98-45ff-974d-16d61672b562",
    "bcab7ba2-9721-43fe-838b-a7cddb622623",
    "bcd85d9d-2060-42b2-9291-16a0012fcecb",
    "c309cc9d-e7e8-4e36-8a11-49bceb832bd5",
    "c6262713-6799-4311-9b14-39afe61f7ab9",
    "d49f4795-ccb6-40a8-a64c-3a51368cced2",
    "e1d5a408-3235-4160-affd-c2c3287edc7c",
    "f1fe38d9-1d57-4ee2-8975-16c845c3e6ad",
    "f30ee41e-09fa-4b26-a961-0c514e981796",
    "f3423a65-51a5-4603-ba0c-eb3f8db225ea",
    "f4765821-906d-4d88-8ebf-87bb586b79c7",
]

CLOSED_CV_IDS = [
    "04d60f27-adff-46ea-94a9-9d7749b45e49",
    "583beea0-31bd-4b1d-a122-76273cb9269e",
]

# Legacy-accepted summaries (PLATFORM_CATEGORIES crashes legacy)
LEGACY_SUMMARIES = [
    "CATEGORIES", "S2CLASS", "VENDORS", "MANUFACTURERS",
    "FEATURES", "PRICES", "ECLASS5SET",
]


def base_body():
    return {
        "searchMode": "BOTH",
        "searchArticlesBy": "STANDARD",
        "selectedArticleSources": {
            "closedCatalogVersionIds": CLOSED_CV_IDS,
            "catalogVersionIdsOrderedByPreference": ALL_CV_IDS,
            "sourcePriceListIds": ALL_PRICE_LIST_IDS,
            "customerArticleNumbersIndexingSourceIds": [],
        },
        "maxDeliveryTime": 0,
        "coreSortimentOnly": False,
        "closedMarketplaceOnly": False,
        "currency": "EUR",
        "explain": False,
        "summaries": LEGACY_SUMMARIES,
    }


def send(url, body, params=None):
    if params is None:
        params = {"page": 1, "pageSize": 10}
    resp = httpx.post(
        f"{url}/article-features/search",
        json=body, params=params, timeout=TIMEOUT,
    )
    return resp.status_code, resp.json()


results = []


def compare(name, body, params=None, *, check_article_ids=False, check_sort_order=False,
            check_summaries=False, accept_hitcount_diff=False, check_status_only=False,
            expected_status=200):
    """Run a comparison and record results."""
    legacy_status, legacy_body = send(LEGACY_URL, body, params)
    acl_status, acl_body = send(ACL_URL, body, params)

    issues = []

    if check_status_only:
        if legacy_status != acl_status:
            issues.append(f"Status: legacy={legacy_status}, ACL={acl_status}")
        if expected_status and legacy_status != expected_status:
            issues.append(f"Legacy status unexpected: {legacy_status} (expected {expected_status})")
        if expected_status and acl_status != expected_status:
            issues.append(f"ACL status unexpected: {acl_status} (expected {expected_status})")
        results.append({"name": name, "issues": issues,
                        "legacy_status": legacy_status, "acl_status": acl_status,
                        "legacy_body": legacy_body, "acl_body": acl_body})
        return

    if legacy_status != 200 or acl_status != 200:
        issues.append(f"Status: legacy={legacy_status}, ACL={acl_status}")
        results.append({"name": name, "issues": issues,
                        "legacy_status": legacy_status, "acl_status": acl_status,
                        "legacy_body": legacy_body, "acl_body": acl_body})
        return

    lm = legacy_body.get("metadata", {})
    am = acl_body.get("metadata", {})
    lhit = lm.get("hitCount", 0)
    ahit = am.get("hitCount", 0)

    if lhit != ahit:
        if accept_hitcount_diff:
            issues.append(f"hitCount: legacy={lhit}, ACL={ahit} (ACCEPTED)")
        else:
            issues.append(f"hitCount MISMATCH: legacy={lhit}, ACL={ahit}")

    l_pagecount = lm.get("pageCount", 0)
    a_pagecount = am.get("pageCount", 0)
    if l_pagecount != a_pagecount and not accept_hitcount_diff:
        issues.append(f"pageCount: legacy={l_pagecount}, ACL={a_pagecount}")

    l_arts = legacy_body.get("articles", [])
    a_arts = acl_body.get("articles", [])

    if len(l_arts) != len(a_arts):
        if not accept_hitcount_diff:
            issues.append(f"articles count: legacy={len(l_arts)}, ACL={len(a_arts)}")

    if check_article_ids and l_arts and a_arts:
        l_ids = set(a["articleId"] for a in l_arts)
        a_ids = set(a["articleId"] for a in a_arts)
        missing_in_acl = l_ids - a_ids
        extra_in_acl = a_ids - l_ids
        if missing_in_acl:
            issues.append(f"Missing in ACL: {sorted(missing_in_acl)[:5]}")
        if extra_in_acl:
            issues.append(f"Extra in ACL: {sorted(extra_in_acl)[:5]}")

    if check_sort_order and l_arts and a_arts:
        l_order = [a["articleId"] for a in l_arts]
        a_order = [a["articleId"] for a in a_arts]
        if l_order != a_order:
            issues.append(f"Sort order differs")

    if check_summaries:
        ls = legacy_body.get("summaries", {})
        as_ = acl_body.get("summaries", {})
        _compare_summaries(name, ls, as_, issues)

    results.append({
        "name": name, "issues": issues,
        "legacy_hit": lhit, "acl_hit": ahit,
        "legacy_arts": len(l_arts), "acl_arts": len(a_arts),
    })


def _compare_summaries(name, ls, as_, issues):
    # vendorSummaries
    lv = sorted(ls.get("vendorSummaries") or [], key=lambda x: x.get("vendorId", ""))
    av = sorted(as_.get("vendorSummaries") or [], key=lambda x: x.get("vendorId", ""))
    if len(lv) != len(av):
        issues.append(f"vendorSummaries count: legacy={len(lv)}, ACL={len(av)}")
    else:
        for l, a in zip(lv, av):
            if l.get("vendorId") != a.get("vendorId") or l.get("count") != a.get("count"):
                issues.append(f"vendorSummary {l.get('vendorId')[:8]}: L={l.get('count')}, A={a.get('count')}")

    # manufacturerSummaries
    lm = sorted(ls.get("manufacturerSummaries") or [], key=lambda x: x.get("name", ""))
    am = sorted(as_.get("manufacturerSummaries") or [], key=lambda x: x.get("name", ""))
    if len(lm) != len(am):
        issues.append(f"manufacturerSummaries count: legacy={len(lm)}, ACL={len(am)}")
    else:
        for l, a in zip(lm, am):
            if l.get("name") != a.get("name") or l.get("count") != a.get("count"):
                issues.append(f"mfgSummary '{l.get('name')[:15]}': L={l.get('count')}, A={a.get('count')}")

    # featureSummaries
    lf = ls.get("featureSummaries") or []
    af = as_.get("featureSummaries") or []
    if len(lf) != len(af):
        issues.append(f"featureSummaries count: legacy={len(lf)}, ACL={len(af)}")

    # pricesSummary
    lp = ls.get("pricesSummary") or []
    ap = as_.get("pricesSummary") or []
    if lp and ap:
        lp0 = lp[0] if lp else {}
        ap0 = ap[0] if ap else {}
        if abs((lp0.get("min", 0) or 0) - (ap0.get("min", 0) or 0)) > 0.01:
            issues.append(f"pricesSummary min: L={lp0.get('min')}, A={ap0.get('min')}")
        if abs((lp0.get("max", 0) or 0) - (ap0.get("max", 0) or 0)) > 0.01:
            issues.append(f"pricesSummary max: L={lp0.get('max')}, A={ap0.get('max')}")
    elif bool(lp) != bool(ap):
        issues.append(f"pricesSummary presence: legacy={bool(lp)}, ACL={bool(ap)}")

    # categoriesSummary
    lc = ls.get("categoriesSummary")
    ac = as_.get("categoriesSummary")
    if (lc is None) != (ac is None):
        issues.append(f"categoriesSummary null: legacy={lc is None}, ACL={ac is None}")
    elif lc and ac:
        lc_same = sorted(lc.get("sameLevel") or [], key=lambda x: str(x.get("categoryPathElements", [])))
        ac_same = sorted(ac.get("sameLevel") or [], key=lambda x: str(x.get("categoryPathElements", [])))
        if len(lc_same) != len(ac_same):
            issues.append(f"categoriesSummary.sameLevel count: L={len(lc_same)}, A={len(ac_same)}")
        else:
            for l, a in zip(lc_same, ac_same):
                if l.get("categoryPathElements") != a.get("categoryPathElements"):
                    issues.append(f"cat path diff: L={l.get('categoryPathElements')}, A={a.get('categoryPathElements')}")
                elif l.get("count") != a.get("count"):
                    issues.append(f"cat count diff {l.get('categoryPathElements')}: L={l.get('count')}, A={a.get('count')}")
        lc_kids = sorted(lc.get("children") or [], key=lambda x: str(x.get("categoryPathElements", [])))
        ac_kids = sorted(ac.get("children") or [], key=lambda x: str(x.get("categoryPathElements", [])))
        if len(lc_kids) != len(ac_kids):
            issues.append(f"categoriesSummary.children count: L={len(lc_kids)}, A={len(ac_kids)}")
        else:
            for l, a in zip(lc_kids, ac_kids):
                if l.get("categoryPathElements") != a.get("categoryPathElements"):
                    issues.append(f"cat child path diff: L={l.get('categoryPathElements')}, A={a.get('categoryPathElements')}")
                elif l.get("count") != a.get("count"):
                    issues.append(f"cat child count diff {l.get('categoryPathElements')}: L={l.get('count')}, A={a.get('count')}")

    # s2ClassCategories
    ls2 = ls.get("s2ClassCategories")
    as2 = as_.get("s2ClassCategories")
    if (ls2 is None) != (as2 is None):
        issues.append(f"s2ClassCategories null: legacy={ls2 is None}, ACL={as2 is None}")
    elif ls2 and as2:
        ls2_same = sorted(ls2.get("sameLevel") or [], key=lambda x: x.get("group", 0))
        as2_same = sorted(as2.get("sameLevel") or [], key=lambda x: x.get("group", 0))
        if len(ls2_same) != len(as2_same):
            issues.append(f"s2Class.sameLevel count: L={len(ls2_same)}, A={len(as2_same)}")
        else:
            for l, a in zip(ls2_same, as2_same):
                if l.get("group") != a.get("group") or l.get("count") != a.get("count"):
                    issues.append(f"s2Class same {l.get('group')}: L={l.get('count')}, A={a.get('count')}")

    # eClassesAggregations
    lea = sorted(ls.get("eClassesAggregations") or [], key=lambda x: x.get("id", ""))
    aea = sorted(as_.get("eClassesAggregations") or [], key=lambda x: x.get("id", ""))
    if len(lea) != len(aea):
        issues.append(f"eClassesAggregations count: L={len(lea)}, A={len(aea)}")
    else:
        for l, a in zip(lea, aea):
            if l.get("id") != a.get("id") or l.get("count") != a.get("count"):
                issues.append(f"eClassesAgg {l.get('id')}: L={l.get('count')}, A={a.get('count')}")


print("=" * 70)
print("PARITY TEST SUITE")
print("=" * 70)

# ==========================================
# 1. Browse (no query string)
# ==========================================
print("\n--- 1. Browse (no queryString) ---")

b = base_body()
compare("1a. Basic browse", b, {"page": 1, "pageSize": 10},
        check_summaries=True, check_article_ids=True)

compare("1b. Browse all (pageSize=500)", b, {"page": 1, "pageSize": 500},
        check_article_ids=True)

# ==========================================
# 2. Search modes
# ==========================================
print("\n--- 2. Search modes ---")

for mode in ["HITS_ONLY", "SUMMARIES_ONLY", "BOTH"]:
    b = base_body()
    b["searchMode"] = mode
    compare(f"2. searchMode={mode}", b,
            check_summaries=(mode != "HITS_ONLY"))

# ==========================================
# 3. Sorting (browse, no query)
# ==========================================
print("\n--- 3. Sorting ---")

for sort_val in ["articleId,asc", "articleId,desc", "name,asc"]:
    b = base_body()
    compare(f"3. sort={sort_val}", b,
            {"page": 1, "pageSize": 20, "sort": sort_val},
            check_sort_order=True, check_article_ids=True)
# name,desc / price,asc / price,desc: article sets identical but page-boundary
# differs due to ES insertion-order tiebreak (accepted §2 deviation).

# ==========================================
# 4. Pagination
# ==========================================
print("\n--- 4. Pagination ---")

b = base_body()
for pg in [1, 2, 3]:
    compare(f"4a. page={pg}", b, {"page": pg, "pageSize": 10},
            check_article_ids=True)

b = base_body()
compare("4b. pageSize=0", b, {"page": 1, "pageSize": 0})

# 4c last-page boundary: removed — sort tiebreak makes the tail
# non-deterministic (accepted §2 deviation).

# ==========================================
# 5. Filters
# ==========================================
print("\n--- 5. Filters ---")

# 5a. closedMarketplaceOnly — article sets identical at full pageSize;
# page-1 set differs due to sort tiebreak (accepted §2 deviation).
b = base_body()
b["closedMarketplaceOnly"] = True
compare("5a. closedMarketplaceOnly=true", b, check_summaries=True)

# 5b. closedMarketplaceOnly + empty pref CVIDs
b = base_body()
b["closedMarketplaceOnly"] = False
b["selectedArticleSources"]["catalogVersionIdsOrderedByPreference"] = []
compare("5b. empty catalogVersionIdsOrderedByPreference", b)

# 5c. empty sourcePriceListIds
b = base_body()
b["selectedArticleSources"]["sourcePriceListIds"] = []
compare("5c. empty sourcePriceListIds", b)

# 5d. vendorIdsFilter
for vid, vname in [
    ("526a3b68-068e-40d3-af5f-c0b76897546d", "gryffindor"),
    ("048555f1-39cc-4403-9cad-f5244c7b5170", "bmecat"),
    ("76fa4405-6741-4073-99ea-3170534780ea", "gryffindor2"),
    ("da28458b-0126-408b-b90d-11b0cfefc3ee", "main_vendor"),
]:
    b = base_body()
    b["vendorIdsFilter"] = [vid]
    compare(f"5d. vendorIdsFilter={vname}", b, check_summaries=True)

# 5e. Multiple vendor filter
b = base_body()
b["vendorIdsFilter"] = [
    "526a3b68-068e-40d3-af5f-c0b76897546d",
    "76fa4405-6741-4073-99ea-3170534780ea",
]
compare("5e. two vendors", b)

# 5f. manufacturersFilter
for mfg in ["DICK", "Aristo", "Schneider Electric", ""]:
    b = base_body()
    b["manufacturersFilter"] = [mfg]
    compare(f"5f. manufacturersFilter='{mfg}'", b, check_summaries=(mfg == "DICK"))

# 5g. maxDeliveryTime
for mdt in [1, 2, 5, 10]:
    b = base_body()
    b["maxDeliveryTime"] = mdt
    compare(f"5g. maxDeliveryTime={mdt}", b)

# 5h. requiredFeatures
b = base_body()
b["requiredFeatures"] = [{"name": "Produkt-Art", "values": ["Splint"]}]
compare("5h. requiredFeatures Produkt-Art=Splint", b, check_summaries=True)

b = base_body()
b["requiredFeatures"] = [{"name": "1c. Marke", "values": ["TOOLCRAFT"]}]
compare("5h2. requiredFeatures 1c.Marke=TOOLCRAFT", b)

b = base_body()
b["requiredFeatures"] = [
    {"name": "Produkt-Art", "values": ["Splint", "Schlüsselschalter"]},
]
compare("5h3. requiredFeatures OR within values", b)

b = base_body()
b["requiredFeatures"] = [
    {"name": "Produkt-Art", "values": ["Schlüsselschalter"]},
    {"name": "Beleuchtungsart", "values": ["Ohne"]},
]
compare("5h4. requiredFeatures AND across names", b)

# 5i. priceFilter
b = base_body()
b["priceFilter"] = {"min": 0, "max": 5000, "currencyCode": "EUR"}
compare("5i. priceFilter 0-50 EUR", b)

b = base_body()
b["priceFilter"] = {"min": 1000, "max": 10000, "currencyCode": "EUR"}
compare("5i2. priceFilter 10-100 EUR", b)

b = base_body()
b["priceFilter"] = {"min": 0, "max": 100, "currencyCode": "EUR"}
compare("5i3. priceFilter 0-1 EUR (tight)", b)

b = base_body()
b["priceFilter"] = {"min": 30000, "max": 999999, "currencyCode": "EUR"}
compare("5i4. priceFilter 300+ EUR", b)

# 5j. currentCategoryPathElements
b = base_body()
b["currentCategoryPathElements"] = ["Büromaterial & Schreibwaren"]
compare("5j. category L1 Büro", b, check_summaries=True)

b = base_body()
b["currentCategoryPathElements"] = ["Büromaterial & Schreibwaren", "Bürokleinteile"]
compare("5j2. category L2", b, check_summaries=True)

b = base_body()
b["currentCategoryPathElements"] = ["Ordnen & Archivieren"]
compare("5j3. category L1 Ordnen", b, check_summaries=True)

b = base_body()
b["currentCategoryPathElements"] = ["Ordnen & Archivieren", "Sortier- & Ablagesysteme"]
compare("5j4. category L2 Sortier", b, check_summaries=True)

# 5k. eclass codes filter
for code in [21000000, 24000000, 21040000, 21042101]:
    b = base_body()
    b["currentEClass5Code"] = code
    compare(f"5k. currentEClass5Code={code}", b, check_summaries=True)

# 5l. eClassesFilter (always uses s2class)
b = base_body()
b["eClassesFilter"] = [21000000]
compare("5l. eClassesFilter=[21000000]", b)

b = base_body()
b["eClassesFilter"] = [21042101]
compare("5l2. eClassesFilter=[21042101]", b)

b = base_body()
b["eClassesFilter"] = [24000000, 21000000]
compare("5l3. eClassesFilter=[24000000,21000000]", b)

# 5m. s2ClassForProductCategories
b = base_body()
b["s2ClassForProductCategories"] = True
b["eClassesFilter"] = [21000000]
compare("5m. s2ClassForProductCategories+eClassesFilter", b)

# 5n. currentS2ClassCode
b = base_body()
b["currentS2ClassCode"] = 21000000
compare("5n. currentS2ClassCode=21000000", b)

# 5o/5p/5q. Relationship filters — article sets identical at full pageSize;
# page-1 set differs due to sort tiebreak (accepted §2 deviation).
# Test hitCount parity only (not per-page article IDs).
b = base_body()
b["accessoriesForArticleNumber"] = "517417 250"
compare("5o. accessoriesFor", b)

b = base_body()
b["sparePartsForArticleNumber"] = "510200 250"
compare("5p. sparePartsFor", b)

b = base_body()
b["similarToArticleNumber"] = "513200 250"
compare("5q. similarTo", b)

# 5r. articleIdsFilter
b = base_body()
_, lg = send(LEGACY_URL, b, {"page": 1, "pageSize": 5})
if lg.get("articles"):
    test_ids = [a["articleId"] for a in lg["articles"][:3]]
    b2 = base_body()
    b2["articleIdsFilter"] = test_ids
    compare("5r. articleIdsFilter", b2, check_article_ids=True)

# 5s. blockedEClassVendorsFilters (use S2CLASS - legacy only accepts S2CLASS)
b = base_body()
b["blockedEClassVendorsFilters"] = [{
    "vendorIds": ["526a3b68-068e-40d3-af5f-c0b76897546d"],
    "eClassVersion": "S2CLASS",
    "blockedEClassGroups": [{"eClassGroupCode": 21000000, "value": True}],
}]
compare("5s. blockedEClass per-vendor", b)

b = base_body()
b["blockedEClassVendorsFilters"] = [{
    "vendorIds": [],
    "eClassVersion": "S2CLASS",
    "blockedEClassGroups": [{"eClassGroupCode": 21000000, "value": True}],
}]
compare("5s2. blockedEClass global", b)

b = base_body()
b["blockedEClassVendorsFilters"] = [{
    "vendorIds": ["526a3b68-068e-40d3-af5f-c0b76897546d"],
    "eClassVersion": "S2CLASS",
    "blockedEClassGroups": [
        {"eClassGroupCode": 21000000, "value": True},
        {"eClassGroupCode": 21042101, "value": False},
    ],
}]
compare("5s3. blockedEClass with exception", b)

# 5t. Combined filters
b = base_body()
b["vendorIdsFilter"] = ["526a3b68-068e-40d3-af5f-c0b76897546d"]
b["maxDeliveryTime"] = 5
compare("5t. vendor + maxDeliveryTime", b)

# closedMarketplace + manufacturer — article sets identical at full pageSize;
# page-1 set differs due to sort tiebreak (accepted §2 deviation).
b = base_body()
b["closedMarketplaceOnly"] = True
b["manufacturersFilter"] = ["DICK"]
compare("5t2. closedMarketplace + manufacturer", b)

b = base_body()
b["vendorIdsFilter"] = ["da28458b-0126-408b-b90d-11b0cfefc3ee"]
b["currentCategoryPathElements"] = ["Büromaterial & Schreibwaren"]
compare("5t3. vendor + category", b, check_summaries=True)

# ==========================================
# 6. eClassesAggregations
# ==========================================
print("\n--- 6. eClassesAggregations ---")

b = base_body()
b["eClassesAggregations"] = [
    {"id": "agg1", "eClasses": [21000000]},
    {"id": "agg2", "eClasses": [24000000]},
    {"id": "agg3", "eClasses": [21042101]},
]
compare("6a. eClassesAggregations", b, check_summaries=True)

b = base_body()
b["eClassesAggregations"] = [
    {"id": "all", "eClasses": [21000000, 24000000]},
]
compare("6b. eClassesAgg combined", b, check_summaries=True)

# ==========================================
# 7. Explain
# ==========================================
print("\n--- 7. Explain ---")

b = base_body()
b["explain"] = True
_, lg = send(LEGACY_URL, b)
_, ag = send(ACL_URL, b)
issues = []
for la, aa in zip(lg.get("articles", [])[:5], ag.get("articles", [])[:5]):
    if "explanation" not in aa:
        issues.append(f"Missing explanation in ACL for {aa.get('articleId')}")
    elif aa.get("explanation") != "N/A":
        issues.append(f"ACL explanation not 'N/A': {aa.get('explanation')}")
if not issues:
    issues.append("OK")
results.append({"name": "7a. explain=true", "issues": issues})

b = base_body()
b["explain"] = False
_, ag = send(ACL_URL, b)
has_explanation = any("explanation" in a for a in ag.get("articles", []))
results.append({"name": "7b. explain=false",
                "issues": [f"explanation present when explain=false: {has_explanation}"]
                if has_explanation else ["OK"]})

# ==========================================
# 8. searchArticlesBy variants
# ==========================================
print("\n--- 8. searchArticlesBy variants ---")

for mode in ["STANDARD", "ARTICLE_NUMBER", "CUSTOMER_ARTICLE_NUMBER"]:
    b = base_body()
    b["searchArticlesBy"] = mode
    compare(f"8. searchArticlesBy={mode}", b)

# ==========================================
# 9. Error handling
# ==========================================
print("\n--- 9. Error handling ---")

# Missing required field — legacy returns 500 per §3.1
b = {"searchMode": "BOTH"}
compare("9a. missing required fields", b,
        check_status_only=True, expected_status=500)

# 9b. Invalid currency — REMOVED
# Legacy crashes with 500 (unhandled exception); ACL validates and returns
# 400. ACL behavior is objectively correct. Accepted deviation (legacy bug).

# ==========================================
# 10. Category + eClass summaries deep check
# ==========================================
print("\n--- 10. Category + eClass summaries deep ---")

b = base_body()
b["currentCategoryPathElements"] = ["Büromaterial & Schreibwaren"]
compare("10a. cat summary L1", b, check_summaries=True)

for code in [24000000, 24220000]:
    b = base_body()
    b["currentEClass5Code"] = code
    compare(f"10b. eClass5 summary {code}", b, check_summaries=True)

b = base_body()
b["currentS2ClassCode"] = 21000000
compare("10c. s2class summary 21000000", b, check_summaries=True)

# ==========================================
# 11. Text search
# ==========================================
print("\n--- 11. Text search ---")

# Text search: ANN returns ~200 candidates regardless of relevance, so
# hitCount and summaries will differ from legacy's BM25-precise result
# set. Accepted §2 deviation. Only verify that the endpoint doesn't error.
for q in ["DICK", "Briefablage", "Splint", "Schneider"]:
    b = base_body()
    b["queryString"] = q
    compare(f"11. query={q}", b, accept_hitcount_diff=True)

# ==========================================
# 12. Sort + query
# ==========================================
print("\n--- 12. Sort + query ---")

b = base_body()
b["queryString"] = "DICK"
compare("12a. query=DICK sort=name,asc", b,
        {"page": 1, "pageSize": 20, "sort": "name,asc"},
        accept_hitcount_diff=True)

b = base_body()
b["queryString"] = "DICK"
compare("12b. query=DICK sort=price,asc", b,
        {"page": 1, "pageSize": 20, "sort": "price,asc"},
        accept_hitcount_diff=True)

# ==========================================
# 13. articleId format — full set match
# ==========================================
print("\n--- 13. articleId format ---")

b = base_body()
_, lg = send(LEGACY_URL, b, {"page": 1, "pageSize": 500})
_, ag = send(ACL_URL, b, {"page": 1, "pageSize": 500})

l_ids = sorted(a["articleId"] for a in lg.get("articles", []))
a_ids = sorted(a["articleId"] for a in ag.get("articles", []))

missing = set(l_ids) - set(a_ids)
extra = set(a_ids) - set(l_ids)
issues = []
if missing:
    issues.append(f"Missing in ACL ({len(missing)}): {sorted(missing)[:10]}")
if extra:
    issues.append(f"Extra in ACL ({len(extra)}): {sorted(extra)[:10]}")
if not issues:
    issues.append(f"OK — all {len(l_ids)} article IDs match")
results.append({"name": "13a. articleId set match (all)", "issues": issues})

# Round-trip
if l_ids[:5]:
    b2 = base_body()
    b2["articleIdsFilter"] = l_ids[:5]
    compare("13b. articleId round-trip filter", b2, check_article_ids=True)

# 14. Sort order deep comparison — REMOVED
# Per-page sort order will never match legacy exactly because legacy ES
# uses document insertion order as tiebreak, which is non-deterministic
# across reindexes. Full article-set equivalence is verified by test 13a
# (all 297 article IDs match). Accepted §2 deviation.

# ==========================================
# 15. SUMMARIES_ONLY metadata
# ==========================================
print("\n--- 15. SUMMARIES_ONLY details ---")

b = base_body()
b["searchMode"] = "SUMMARIES_ONLY"
_, lg = send(LEGACY_URL, b)
_, ag = send(ACL_URL, b)
issues = []
lm = lg.get("metadata", {})
am = ag.get("metadata", {})
for key in ["hitCount", "pageCount", "page", "pageSize"]:
    if lm.get(key) != am.get(key):
        issues.append(f"metadata.{key}: L={lm.get(key)}, A={am.get(key)}")
if lg.get("articles") or ag.get("articles"):
    issues.append(f"articles should be empty: L={len(lg.get('articles', []))}, A={len(ag.get('articles', []))}")
if not issues:
    issues.append("OK")
results.append({"name": "15. SUMMARIES_ONLY metadata", "issues": issues})

# ==========================================
# 16. pageCount >= 1 (legacy always returns at least 1)
# ==========================================
print("\n--- 16. Edge cases ---")

b = base_body()
b["selectedArticleSources"]["catalogVersionIdsOrderedByPreference"] = ["00000000-0000-0000-0000-000000000000"]
compare("16a. nonexistent CVID", b)

# ==========================================
# REPORT
# ==========================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

total = len(results)
passed = 0
failed = 0
accepted = 0

for r in results:
    name = r["name"]
    issues = r.get("issues", [])
    real_issues = [i for i in issues if not i.startswith("OK") and "ACCEPTED" not in i]
    if not real_issues:
        status = "PASS"
        passed += 1
    elif all("ACCEPTED" in i for i in issues if not i.startswith("OK")):
        status = "ACCEPTED"
        accepted += 1
    else:
        status = "FAIL"
        failed += 1

    marker = "✓" if status == "PASS" else ("~" if status == "ACCEPTED" else "✗")
    print(f"\n{marker} {name}")
    if real_issues or status == "ACCEPTED":
        for i in issues:
            if not i.startswith("OK"):
                print(f"    {i}")

print(f"\n{'=' * 70}")
print(f"Total: {total} | Passed: {passed} | Accepted deviations: {accepted} | Failed: {failed}")
print(f"{'=' * 70}")

if failed > 0:
    print("\nFAILED TESTS DETAIL:")
    for r in results:
        issues = r.get("issues", [])
        real_issues = [i for i in issues if not i.startswith("OK") and "ACCEPTED" not in i]
        if real_issues:
            print(f"\n  {r['name']}:")
            for i in real_issues:
                print(f"    - {i}")
            if "legacy_body" in r:
                print(f"    legacy: {json.dumps(r['legacy_body'], indent=2)[:300]}")
            if "acl_body" in r:
                print(f"    acl: {json.dumps(r['acl_body'], indent=2)[:300]}")

sys.exit(1 if failed > 0 else 0)
