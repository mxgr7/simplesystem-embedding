"""Wire DTOs for the new ftsearch search contract (F2).

Mirrors the legacy contract in `issues/article-search-replacement-spec.md`
§3 with two intentional deltas:

  * `searchArticlesBy` — dropped (only `STANDARD` is supported per §2.1;
    the ACL rejects other values, ftsearch never sees them).
  * `explain` — dropped (the ACL stubs `explanation = "N/A"` per §2.2,
    ftsearch returns `score` per article instead).

Python attributes are snake_case; aliases match the legacy wire format
exactly. `extra='forbid'` makes unknown fields a 422, matching legacy's
strict validation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Base config for every DTO in this module.

    `populate_by_name=True` — accept either alias or Python field name on
    input; aliases are still emitted on output (set per route via
    `response_model_by_alias=True`).
    `extra='forbid'` — unknown keys raise (matches legacy bean-validation
    strictness; surfaces typos/contract drift fast).
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ---------- enums ----------------------------------------------------------

class SearchMode(str, Enum):
    HITS_ONLY = "HITS_ONLY"
    SUMMARIES_ONLY = "SUMMARIES_ONLY"
    BOTH = "BOTH"


class SummaryKind(str, Enum):
    CATEGORIES = "CATEGORIES"
    ECLASS5 = "ECLASS5"
    ECLASS7 = "ECLASS7"
    S2CLASS = "S2CLASS"
    VENDORS = "VENDORS"
    MANUFACTURERS = "MANUFACTURERS"
    FEATURES = "FEATURES"
    PRICES = "PRICES"
    PLATFORM_CATEGORIES = "PLATFORM_CATEGORIES"
    ECLASS5SET = "ECLASS5SET"


class EClassVersion(str, Enum):
    ECLASS_5_1 = "ECLASS_5_1"
    ECLASS_7_1 = "ECLASS_7_1"
    S2CLASS = "S2CLASS"


# ---------- request sub-models --------------------------------------------

class SelectedArticleSources(_Strict):
    """Catalog scoping fields ftsearch consumes for filtering / pricing.

    The customer-article-number variants from legacy (`customerArticleNumbers...`,
    `customerManagedArticleNumberListId`, `uiCustomerArticleNumberSourceId`)
    are dropped here — they served the now-removed `CUSTOMER_ARTICLE_NUMBER`
    mode (§2.1) and the ACL must reject any non-empty values rather than
    forward them.
    """
    closed_catalog_version_ids: list[str] = Field(default_factory=list, alias="closedCatalogVersionIds")
    catalog_version_ids_ordered_by_preference: list[str] = Field(default_factory=list, alias="catalogVersionIdsOrderedByPreference")
    source_price_list_ids: list[str] = Field(default_factory=list, alias="sourcePriceListIds")
    customer_uploaded_core_article_list_source_ids: list[str] = Field(default_factory=list, alias="customerUploadedCoreArticleListSourceIds")


class FeatureFilter(_Strict):
    name: str
    values: list[str]


class PriceFilter(_Strict):
    """Per spec §3: `min`/`max` are integer minor units; `currencyCode` is
    consumed only for decoding bounds via the currency's default fraction
    digits — it is *not* used to match `prices.currency`."""
    min: int | None = None
    max: int | None = None
    currency_code: str = Field(alias="currencyCode", min_length=3, max_length=3)


class BlockedEClassGroup(_Strict):
    e_class_group_code: int = Field(alias="eClassGroupCode")
    value: bool


class BlockedEClassVendorsFilter(_Strict):
    vendor_ids: list[str] = Field(default_factory=list, alias="vendorIds")
    e_class_version: EClassVersion = Field(alias="eClassVersion")
    blocked_e_class_groups: list[BlockedEClassGroup] = Field(default_factory=list, alias="blockedEClassGroups")


class EClassesAggregation(_Strict):
    id: str
    e_classes: list[int] = Field(default_factory=list, alias="eClasses")


# ---------- request --------------------------------------------------------

class SearchRequest(_Strict):
    """Body for `POST /{collection}/_search`.

    Pagination (`page`, `pageSize`) and `sort` are query-string parameters
    per §3, not body fields — they're handled at the route layer.
    """

    query: str | None = None
    search_mode: SearchMode = Field(alias="searchMode")
    selected_article_sources: SelectedArticleSources = Field(alias="selectedArticleSources")

    article_ids_filter: list[str] = Field(default_factory=list, alias="articleIdsFilter")
    vendor_ids_filter: list[str] = Field(default_factory=list, alias="vendorIdsFilter")
    manufacturers_filter: list[str] = Field(default_factory=list, alias="manufacturersFilter")
    max_delivery_time: int = Field(default=0, alias="maxDeliveryTime", ge=0)
    required_features: list[FeatureFilter] = Field(default_factory=list, alias="requiredFeatures")
    price_filter: PriceFilter | None = Field(default=None, alias="priceFilter")

    accessories_for_article_number: str | None = Field(default=None, alias="accessoriesForArticleNumber")
    spare_parts_for_article_number: str | None = Field(default=None, alias="sparePartsForArticleNumber")
    similar_to_article_number: str | None = Field(default=None, alias="similarToArticleNumber")

    current_category_path_elements: list[str] = Field(default_factory=list, alias="currentCategoryPathElements")
    current_eclass5_code: int | None = Field(default=None, alias="currentEClass5Code")
    current_eclass7_code: int | None = Field(default=None, alias="currentEClass7Code")
    current_s2class_code: int | None = Field(default=None, alias="currentS2ClassCode")

    core_sortiment_only: bool = Field(default=False, alias="coreSortimentOnly")
    closed_marketplace_only: bool = Field(default=False, alias="closedMarketplaceOnly")

    summaries: list[SummaryKind] = Field(default_factory=list)
    core_articles_vendors_filter: list[str] = Field(default_factory=list, alias="coreArticlesVendorsFilter")
    blocked_eclass_vendors_filters: list[BlockedEClassVendorsFilter] = Field(default_factory=list, alias="blockedEClassVendorsFilters")

    currency: str = Field(pattern=r"^[A-Z]{3}$")

    eclasses_filter: list[int] = Field(default_factory=list, alias="eClassesFilter")
    eclasses_aggregations: list[EClassesAggregation] = Field(default_factory=list, alias="eClassesAggregations")
    s2class_for_product_categories: bool = Field(default=False, alias="s2ClassForProductCategories")


# ---------- response sub-models -------------------------------------------

class Article(_Strict):
    """Per F2 packet: ftsearch returns `score` per hit; the ACL drops it
    and stubs `explanation = "N/A"` for the legacy envelope (§2.2)."""
    article_id: str = Field(alias="articleId")
    score: float | None = None


class VendorSummary(_Strict):
    vendor_id: str = Field(alias="vendorId")
    count: int = Field(ge=0)


class NameCount(_Strict):
    name: str
    count: int = Field(ge=0)


class FeatureValueCount(_Strict):
    value: str
    count: int = Field(ge=0)


class FeatureSummary(_Strict):
    name: str
    count: int = Field(ge=0)
    values: list[FeatureValueCount] = Field(default_factory=list)


class PricesSummary(_Strict):
    min: float = 0.0
    max: float = 0.0
    currency_code: str = Field(alias="currencyCode", min_length=3, max_length=3)


class CategoryBucket(_Strict):
    category_path_elements: list[str] = Field(default_factory=list, alias="categoryPathElements")
    count: int = Field(ge=0)


class CategoriesSummary(_Strict):
    current_category_path_elements: list[str] = Field(default_factory=list, alias="currentCategoryPathElements")
    same_level: list[CategoryBucket] = Field(default_factory=list, alias="sameLevel")
    children: list[CategoryBucket] = Field(default_factory=list)


class EClassBucket(_Strict):
    group: int
    count: int = Field(ge=0)


class EClassCategories(_Strict):
    selected_e_class_group: int | None = Field(default=None, alias="selectedEClassGroup")
    same_level: list[EClassBucket] = Field(default_factory=list, alias="sameLevel")
    children: list[EClassBucket] = Field(default_factory=list)


class EClassesAggregationCount(_Strict):
    id: str
    count: int = Field(ge=0)


class Summaries(_Strict):
    """All sub-summaries default empty/null. F5 fills them per `summaries`
    list in the request; HITS_ONLY mode emits the envelope with everything
    empty (per §3 mode rules — "summaries empty/omitted")."""
    vendor_summaries: list[VendorSummary] = Field(default_factory=list, alias="vendorSummaries")
    manufacturer_summaries: list[NameCount] = Field(default_factory=list, alias="manufacturerSummaries")
    feature_summaries: list[FeatureSummary] = Field(default_factory=list, alias="featureSummaries")
    prices_summary: list[PricesSummary] = Field(default_factory=list, alias="pricesSummary")
    categories_summary: CategoriesSummary | None = Field(default=None, alias="categoriesSummary")
    eclass5_categories: EClassCategories | None = Field(default=None, alias="eClass5Categories")
    eclass7_categories: EClassCategories | None = Field(default=None, alias="eClass7Categories")
    s2class_categories: EClassCategories | None = Field(default=None, alias="s2ClassCategories")
    eclasses_aggregations: list[EClassesAggregationCount] = Field(default_factory=list, alias="eClassesAggregations")


class Metadata(_Strict):
    page: int = Field(ge=1)
    page_size: int = Field(alias="pageSize", ge=0, le=500)
    page_count: int = Field(alias="pageCount", ge=0)
    term: str | None = None
    hit_count: int = Field(alias="hitCount", ge=0)


class SearchResponse(_Strict):
    articles: list[Article] = Field(default_factory=list)
    summaries: Summaries
    metadata: Metadata


# ---------- sort param parsing --------------------------------------------

class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


SORTABLE_FIELDS = frozenset({"name", "price", "articleId"})


class SortClause(_Strict):
    field: str
    direction: SortDirection

    @field_validator("field")
    @classmethod
    def _known_field(cls, v: str) -> str:
        if v not in SORTABLE_FIELDS:
            raise ValueError(f"unknown sort field {v!r}; expected one of {sorted(SORTABLE_FIELDS)}")
        return v


def parse_sort_params(raw: list[str]) -> list[SortClause]:
    """Each `?sort=` value is `<field>,<asc|desc>`. Repeats are ordered."""
    out: list[SortClause] = []
    for item in raw:
        head, sep, tail = item.partition(",")
        if not sep:
            raise ValueError(f"sort param {item!r} must be of the form '<field>,<asc|desc>'")
        out.append(SortClause(field=head.strip(), direction=SortDirection(tail.strip().lower())))
    return out
