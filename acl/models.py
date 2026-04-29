"""Pydantic models for the legacy `/article-features/search` request.

Mirrors the schema in `acl/openapi.yaml` (the contract source of
truth). `extra='forbid'` so unknown fields land as 400 — next-gen
callers passing a dropped field (like `searchArticlesBy: ARTICLE_NUMBER`,
which §2.1 removed) get a clear error rather than silent acceptance.

A3 owns the response models; A2 only needs the request shape so the
mapper can validate input + translate to the ftsearch DTO.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """All ACL DTOs reject unknown fields and accept either the
    legacy camelCase wire name or the snake_case Python name."""
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class SearchMode(str, Enum):
    HITS_ONLY = "HITS_ONLY"
    SUMMARIES_ONLY = "SUMMARIES_ONLY"
    BOTH = "BOTH"


class SearchArticlesBy(str, Enum):
    """§2.1 deviation — single-value enum. Any other value rejected
    by Pydantic before reaching the mapper."""
    STANDARD = "STANDARD"


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


class SelectedArticleSources(_Strict):
    closed_catalog_version_ids: list[str] = Field(alias="closedCatalogVersionIds")
    catalog_version_ids_ordered_by_preference: list[str] = Field(
        default_factory=list, alias="catalogVersionIdsOrderedByPreference",
    )
    source_price_list_ids: list[str] = Field(default_factory=list, alias="sourcePriceListIds")
    customer_article_numbers_indexing_source_ids: list[str] = Field(
        default_factory=list, alias="customerArticleNumbersIndexingSourceIds",
    )
    customer_uploaded_core_article_list_source_ids: list[str] = Field(
        default_factory=list, alias="customerUploadedCoreArticleListSourceIds",
    )
    customer_managed_article_number_list_id: str | None = Field(
        default=None, alias="customerManagedArticleNumberListId",
    )
    ui_customer_article_number_source_id: str | None = Field(
        default=None, alias="uiCustomerArticleNumberSourceId",
    )


class FeatureFilter(_Strict):
    name: str
    values: list[str] = Field(default_factory=list)


class PriceFilter(_Strict):
    """Per §3 "Currency fields — two roles": `currencyCode` here drives
    bound-decoding (decimal places per ISO 4217) only, not match. The
    top-level `currency` on the request governs the match.

    Cross-field rule (§3): `currencyCode` is required (non-null)
    whenever `min` or `max` is set — without it ftsearch can't decode
    the integer minor units into a decimal amount."""
    min: int | None = None
    max: int | None = None
    currency_code: str | None = Field(
        default=None, alias="currencyCode", pattern=r"^[A-Z]{3}$",
    )

    @model_validator(mode="after")
    def _currency_code_required_when_bound_set(self) -> "PriceFilter":
        if (self.min is not None or self.max is not None) and self.currency_code is None:
            raise ValueError(
                "priceFilter.currencyCode is required when priceFilter.min "
                "or priceFilter.max is set (per spec §3 — drives bound-decoding "
                "via that currency's ISO 4217 fraction-digit count)"
            )
        return self


class BlockedEClassGroup(_Strict):
    e_class_group_code: int = Field(alias="eClassGroupCode")
    value: bool


class BlockedEClassVendorsFilter(_Strict):
    vendor_ids: list[str] = Field(alias="vendorIds")
    e_class_version: EClassVersion = Field(alias="eClassVersion")
    blocked_e_class_groups: list[BlockedEClassGroup] = Field(
        default_factory=list, alias="blockedEClassGroups",
    )


class EClassesAggregation(_Strict):
    id: str
    e_classes: list[int] = Field(default_factory=list, alias="eClasses")


class LegacySearchRequest(_Strict):
    """Body for `POST /article-features/search` per spec §3."""

    search_mode: SearchMode = Field(alias="searchMode")
    # §2.1 — single-value enum.
    search_articles_by: SearchArticlesBy = Field(alias="searchArticlesBy")
    selected_article_sources: SelectedArticleSources = Field(alias="selectedArticleSources")

    query_string: str | None = Field(default=None, alias="queryString")

    article_ids_filter: list[str] = Field(default_factory=list, alias="articleIdsFilter")
    vendor_ids_filter: list[str] = Field(default_factory=list, alias="vendorIdsFilter")
    manufacturers_filter: list[str] = Field(default_factory=list, alias="manufacturersFilter")
    max_delivery_time: int = Field(alias="maxDeliveryTime", ge=0)
    required_features: list[FeatureFilter] = Field(default_factory=list, alias="requiredFeatures")
    price_filter: PriceFilter | None = Field(default=None, alias="priceFilter")

    accessories_for_article_number: str | None = Field(
        default=None, alias="accessoriesForArticleNumber",
    )
    spare_parts_for_article_number: str | None = Field(
        default=None, alias="sparePartsForArticleNumber",
    )
    similar_to_article_number: str | None = Field(
        default=None, alias="similarToArticleNumber",
    )

    current_category_path_elements: list[str] = Field(
        default_factory=list, alias="currentCategoryPathElements",
    )
    current_e_class5_code: int | None = Field(default=None, alias="currentEClass5Code")
    current_e_class7_code: int | None = Field(default=None, alias="currentEClass7Code")
    current_s2_class_code: int | None = Field(default=None, alias="currentS2ClassCode")

    core_sortiment_only: bool = Field(alias="coreSortimentOnly")
    closed_marketplace_only: bool = Field(alias="closedMarketplaceOnly")

    summaries: list[SummaryKind] = Field(default_factory=list)
    core_articles_vendors_filter: list[str] = Field(
        default_factory=list, alias="coreArticlesVendorsFilter",
    )
    blocked_e_class_vendors_filters: list[BlockedEClassVendorsFilter] = Field(
        default_factory=list, alias="blockedEClassVendorsFilters",
    )

    currency: str = Field(pattern=r"^[A-Z]{3}$")

    # §2.2 — accepted but ftsearch never sees it. Response mapper (A3)
    # stubs `articles[].explanation = "N/A"` when this is True.
    explain: bool

    e_classes_filter: list[int] = Field(default_factory=list, alias="eClassesFilter")
    e_classes_aggregations: list[EClassesAggregation] = Field(
        default_factory=list, alias="eClassesAggregations",
    )
    s2_class_for_product_categories: bool = Field(
        default=False, alias="s2ClassForProductCategories",
    )

    @field_validator("search_articles_by", mode="before")
    @classmethod
    def _accept_only_standard(cls, v):
        """Pydantic enum coercion would 422 anything outside the
        single-value enum; we trap it here so the error message
        explicitly references §2.1."""
        if v not in ("STANDARD", SearchArticlesBy.STANDARD):
            raise ValueError(
                "searchArticlesBy must be 'STANDARD' (per spec §2.1 — other "
                "values from the legacy enum were dropped in this contract)"
            )
        return v
