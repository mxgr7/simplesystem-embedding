use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use arrow_array::builder::{BooleanBuilder, Int32Builder, Int64Builder, LargeStringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use clap::{ArgAction, Parser, ValueEnum};
use flate2::read::GzDecoder;
use glob::glob;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rayon::prelude::*;
use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputCompression {
    Snappy,
    Uncompressed,
}

impl OutputCompression {
    fn parquet(self) -> Compression {
        match self {
            Self::Snappy => Compression::SNAPPY,
            Self::Uncompressed => Compression::UNCOMPRESSED,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SelectionMode {
    Lexical,
    Largest,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputFormat {
    Ndjson,
    Auto,
}

#[derive(Debug, Clone, Parser)]
#[command(author, version, about = "Convert Mongo-style *.json.gz offer exports into projected parquet shards")]
struct Args {
    #[arg(long, default_value = "/data/mongodb-export-2026-03-04/offers/*.json.gz")]
    input_glob: String,

    #[arg(long, default_value = "/data/mongodb-export-2026-03-04/offers_parquet_projected")]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 2048)]
    limit: usize,

    #[arg(long, value_enum, default_value_t = SelectionMode::Lexical)]
    selection_mode: SelectionMode,

    #[arg(long, value_enum, default_value_t = InputFormat::Ndjson)]
    input_format: InputFormat,

    #[arg(long, default_value_t = default_threads())]
    threads: usize,

    #[arg(long)]
    shards: Option<usize>,

    #[arg(long, default_value_t = 8192)]
    batch_rows: usize,

    #[arg(long, default_value_t = 131072)]
    row_group_rows: usize,

    #[arg(long, value_enum, default_value_t = OutputCompression::Snappy)]
    compression: OutputCompression,

    #[arg(long, action = ArgAction::SetTrue)]
    include_json_blobs: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    include_raw_json: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    overwrite: bool,
}

#[derive(Debug, Clone)]
struct FileEntry {
    file_id: i32,
    path: PathBuf,
    compressed_size: u64,
}

#[derive(Debug, Default, Clone)]
struct ShardStats {
    files: usize,
    rows: usize,
    compressed_bytes: u64,
    parquet_bytes: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct Count(i32);

impl<'de> Deserialize<'de> for Count {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CountVisitor;

        impl<'de> Visitor<'de> for CountVisitor {
            type Value = Count;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a JSON array")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Count, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut count = 0i32;
                while seq.next_element::<IgnoredAny>()?.is_some() {
                    count += 1;
                }
                Ok(Count(count))
            }

            fn visit_none<E>(self) -> Result<Count, E>
            where
                E: serde::de::Error,
            {
                Ok(Count(0))
            }

            fn visit_unit<E>(self) -> Result<Count, E>
            where
                E: serde::de::Error,
            {
                Ok(Count(0))
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Count, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Deserialize::deserialize(deserializer)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Count, A::Error>
            where
                A: MapAccess<'de>,
            {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Count(0))
            }

            fn visit_bool<E>(self, _v: bool) -> Result<Count, E>
            where
                E: serde::de::Error,
            {
                Ok(Count(0))
            }

            fn visit_str<E>(self, _v: &str) -> Result<Count, E>
            where
                E: serde::de::Error,
            {
                Ok(Count(0))
            }
        }

        deserializer.deserialize_any(CountVisitor)
    }
}

#[derive(Debug, Deserialize)]
struct MongoOid<'a> {
    #[serde(borrow, rename = "$oid")]
    value: &'a str,
}

#[derive(Debug, Deserialize)]
struct MongoBinary<'a> {
    #[serde(borrow, rename = "$binary")]
    value: MongoBinaryInner<'a>,
}

#[derive(Debug, Deserialize)]
struct MongoBinaryInner<'a> {
    #[serde(borrow)]
    base64: &'a str,
}

#[derive(Debug, Deserialize)]
struct NumberInt<'a> {
    #[serde(borrow, rename = "$numberInt")]
    value: &'a str,
}

#[derive(Debug, Deserialize)]
struct NumberLong<'a> {
    #[serde(borrow, rename = "$numberLong")]
    value: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MaybeI64<'a> {
    I64(i64),
    U64(u64),
    #[serde(borrow)]
    Str(&'a str),
    #[serde(borrow)]
    NumberInt(NumberInt<'a>),
    #[serde(borrow)]
    NumberLong(NumberLong<'a>),
}

impl MaybeI64<'_> {
    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(v) => Some(*v),
            Self::U64(v) => i64::try_from(*v).ok(),
            Self::Str(v) => v.parse().ok(),
            Self::NumberInt(v) => v.value.parse().ok(),
            Self::NumberLong(v) => v.value.parse().ok(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ExportRow<'a> {
    #[serde(borrow, rename = "_id")]
    id: Option<MongoOid<'a>>,
    #[serde(borrow, rename = "articleNumber")]
    article_number: Option<&'a str>,
    #[serde(borrow, rename = "vendorId")]
    vendor_id: Option<MongoBinary<'a>>,
    #[serde(borrow, rename = "catalogVersionId")]
    catalog_version_id: Option<MongoBinary<'a>>,
    #[serde(rename = "importEpoch")]
    import_epoch: Option<MaybeI64<'a>>,
    #[serde(borrow)]
    offer: Option<Offer<'a>>,
}

#[derive(Debug, Deserialize)]
struct Offer<'a> {
    #[serde(borrow, rename = "_id")]
    id: Option<MongoBinary<'a>>,
    #[serde(borrow, rename = "catalogVersionId")]
    catalog_version_id: Option<MongoBinary<'a>>,
    #[serde(borrow, rename = "offerParams")]
    offer_params: Option<OfferParams<'a>>,
    #[serde(borrow)]
    pricings: Option<Pricings<'a>>,
}

#[derive(Debug, Deserialize)]
struct OfferParams<'a> {
    #[serde(borrow)]
    name: Option<&'a str>,
    #[serde(borrow, rename = "orderUnit")]
    order_unit: Option<&'a str>,
    #[serde(borrow)]
    tax: Option<&'a str>,
    #[serde(borrow)]
    description: Option<&'a str>,
    #[serde(borrow)]
    ean: Option<&'a str>,
    #[serde(borrow, rename = "hsCode")]
    hs_code: Option<&'a str>,
    #[serde(borrow, rename = "manufacturerName")]
    manufacturer_name: Option<&'a str>,
    #[serde(borrow, rename = "customerArticleNumber")]
    customer_article_number: Option<&'a str>,
    #[serde(borrow, rename = "manufacturerArticleNumber")]
    manufacturer_article_number: Option<&'a str>,
    #[serde(borrow, rename = "manufacturerArticleType")]
    manufacturer_article_type: Option<&'a str>,
    #[serde(borrow, rename = "vendorRemarks")]
    vendor_remarks: Option<&'a str>,
    #[serde(rename = "minimumOrder")]
    minimum_order: Option<MaybeI64<'a>>,
    #[serde(rename = "quantityInterval")]
    quantity_interval: Option<MaybeI64<'a>>,
    #[serde(borrow, rename = "contentUnit")]
    content_unit: Option<&'a str>,
    #[serde(rename = "deliveryTime")]
    delivery_time: Option<MaybeI64<'a>>,
    #[serde(borrow, rename = "contentAmount")]
    content_amount: Option<&'a str>,
    #[serde(borrow, rename = "priceQuantity")]
    price_quantity: Option<&'a str>,
    #[serde(rename = "coreSortiment")]
    core_sortiment: Option<bool>,
    dependent: Option<bool>,
    expired: Option<bool>,
    #[serde(default)]
    keywords: Count,
    #[serde(default)]
    features: Count,
    #[serde(default)]
    images: Count,
    #[serde(default)]
    downloads: Count,
    #[serde(default, rename = "categoryPaths")]
    category_paths: Count,
}

#[derive(Debug, Deserialize)]
struct Pricings<'a> {
    #[serde(borrow)]
    open: Option<Pricing<'a>>,
    #[serde(borrow)]
    closed: Option<Pricing<'a>>,
}

#[derive(Debug, Deserialize)]
struct Pricing<'a> {
    #[serde(borrow)]
    prices: Option<Prices<'a>>,
}

#[derive(Debug, Deserialize)]
struct Prices<'a> {
    #[serde(borrow, rename = "currencyCode")]
    currency_code: Option<&'a str>,
    #[serde(default, borrow, rename = "staggeredPrices")]
    staggered_prices: Vec<StaggeredPrice<'a>>,
}

#[derive(Debug, Deserialize)]
struct StaggeredPrice<'a> {
    #[serde(borrow)]
    price: Option<&'a str>,
}

struct BatchBuilder {
    source_file_id: Int32Builder,
    offer_id: LargeStringBuilder,
    article_number: LargeStringBuilder,
    vendor_id: LargeStringBuilder,
    catalog_version_id: LargeStringBuilder,
    import_epoch: Int64Builder,
    inner_offer_id: LargeStringBuilder,
    inner_catalog_version_id: LargeStringBuilder,
    name: LargeStringBuilder,
    order_unit: LargeStringBuilder,
    tax: LargeStringBuilder,
    description: LargeStringBuilder,
    ean: LargeStringBuilder,
    hs_code: LargeStringBuilder,
    manufacturer_name: LargeStringBuilder,
    customer_article_number: LargeStringBuilder,
    manufacturer_article_number: LargeStringBuilder,
    manufacturer_article_type: LargeStringBuilder,
    vendor_remarks: LargeStringBuilder,
    minimum_order: Int64Builder,
    quantity_interval: Int64Builder,
    content_unit: LargeStringBuilder,
    delivery_time: Int64Builder,
    content_amount: LargeStringBuilder,
    price_quantity: LargeStringBuilder,
    core_sortiment: BooleanBuilder,
    dependent: BooleanBuilder,
    expired: BooleanBuilder,
    keyword_count: Int32Builder,
    feature_count: Int32Builder,
    image_count: Int32Builder,
    download_count: Int32Builder,
    category_path_count: Int32Builder,
    pricing_count: Int32Builder,
    marker_count: Int32Builder,
    customer_article_number_count: Int32Builder,
    open_price: LargeStringBuilder,
    open_currency: LargeStringBuilder,
    closed_price: LargeStringBuilder,
    closed_currency: LargeStringBuilder,
    thumbnail_json: LargeStringBuilder,
    features_json: LargeStringBuilder,
    keywords_json: LargeStringBuilder,
    images_json: LargeStringBuilder,
    downloads_json: LargeStringBuilder,
    eclass_groups_json: LargeStringBuilder,
    category_paths_json: LargeStringBuilder,
    pricings_json: LargeStringBuilder,
    related_article_numbers_json: LargeStringBuilder,
    markers_json: LargeStringBuilder,
    customer_article_numbers_json: LargeStringBuilder,
    raw_json: LargeStringBuilder,
    rows: usize,
}

impl BatchBuilder {
    fn new(capacity: usize) -> Self {
        let avg = capacity.saturating_mul(64);
        let large = capacity.saturating_mul(256);

        Self {
            source_file_id: Int32Builder::with_capacity(capacity),
            offer_id: LargeStringBuilder::with_capacity(capacity, avg),
            article_number: LargeStringBuilder::with_capacity(capacity, avg),
            vendor_id: LargeStringBuilder::with_capacity(capacity, avg),
            catalog_version_id: LargeStringBuilder::with_capacity(capacity, avg),
            import_epoch: Int64Builder::with_capacity(capacity),
            inner_offer_id: LargeStringBuilder::with_capacity(capacity, avg),
            inner_catalog_version_id: LargeStringBuilder::with_capacity(capacity, avg),
            name: LargeStringBuilder::with_capacity(capacity, large),
            order_unit: LargeStringBuilder::with_capacity(capacity, avg),
            tax: LargeStringBuilder::with_capacity(capacity, avg),
            description: LargeStringBuilder::with_capacity(capacity, large),
            ean: LargeStringBuilder::with_capacity(capacity, avg),
            hs_code: LargeStringBuilder::with_capacity(capacity, avg),
            manufacturer_name: LargeStringBuilder::with_capacity(capacity, avg),
            customer_article_number: LargeStringBuilder::with_capacity(capacity, avg),
            manufacturer_article_number: LargeStringBuilder::with_capacity(capacity, avg),
            manufacturer_article_type: LargeStringBuilder::with_capacity(capacity, avg),
            vendor_remarks: LargeStringBuilder::with_capacity(capacity, avg),
            minimum_order: Int64Builder::with_capacity(capacity),
            quantity_interval: Int64Builder::with_capacity(capacity),
            content_unit: LargeStringBuilder::with_capacity(capacity, avg),
            delivery_time: Int64Builder::with_capacity(capacity),
            content_amount: LargeStringBuilder::with_capacity(capacity, avg),
            price_quantity: LargeStringBuilder::with_capacity(capacity, avg),
            core_sortiment: BooleanBuilder::with_capacity(capacity),
            dependent: BooleanBuilder::with_capacity(capacity),
            expired: BooleanBuilder::with_capacity(capacity),
            keyword_count: Int32Builder::with_capacity(capacity),
            feature_count: Int32Builder::with_capacity(capacity),
            image_count: Int32Builder::with_capacity(capacity),
            download_count: Int32Builder::with_capacity(capacity),
            category_path_count: Int32Builder::with_capacity(capacity),
            pricing_count: Int32Builder::with_capacity(capacity),
            marker_count: Int32Builder::with_capacity(capacity),
            customer_article_number_count: Int32Builder::with_capacity(capacity),
            open_price: LargeStringBuilder::with_capacity(capacity, avg),
            open_currency: LargeStringBuilder::with_capacity(capacity, avg),
            closed_price: LargeStringBuilder::with_capacity(capacity, avg),
            closed_currency: LargeStringBuilder::with_capacity(capacity, avg),
            thumbnail_json: LargeStringBuilder::with_capacity(capacity, large),
            features_json: LargeStringBuilder::with_capacity(capacity, large),
            keywords_json: LargeStringBuilder::with_capacity(capacity, large),
            images_json: LargeStringBuilder::with_capacity(capacity, large),
            downloads_json: LargeStringBuilder::with_capacity(capacity, large),
            eclass_groups_json: LargeStringBuilder::with_capacity(capacity, large),
            category_paths_json: LargeStringBuilder::with_capacity(capacity, large),
            pricings_json: LargeStringBuilder::with_capacity(capacity, large),
            related_article_numbers_json: LargeStringBuilder::with_capacity(capacity, large),
            markers_json: LargeStringBuilder::with_capacity(capacity, large),
            customer_article_numbers_json: LargeStringBuilder::with_capacity(capacity, large),
            raw_json: LargeStringBuilder::with_capacity(capacity, large),
            rows: 0,
        }
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn append(&mut self, source_file_id: i32, row: &Value, include_json_blobs: bool, include_raw_json: bool) {
        self.source_file_id.append_value(source_file_id);

        push_opt_str(&mut self.offer_id, mongo_oid(get_path(row, &["_id"])));
        push_opt_str(&mut self.article_number, get_str(row, &["articleNumber"]));
        push_opt_str(&mut self.vendor_id, mongo_binary_base64(get_path(row, &["vendorId"])));
        push_opt_str(
            &mut self.catalog_version_id,
            mongo_binary_base64(get_path(row, &["catalogVersionId"])),
        );
        push_opt_i64(&mut self.import_epoch, get_i64(row, &["importEpoch"]));
        push_opt_str(&mut self.inner_offer_id, mongo_binary_base64(get_path(row, &["offer", "_id"])));
        push_opt_str(
            &mut self.inner_catalog_version_id,
            mongo_binary_base64(get_path(row, &["offer", "catalogVersionId"])),
        );

        push_opt_str(&mut self.name, get_str(row, &["offer", "offerParams", "name"]));
        push_opt_str(&mut self.order_unit, get_str(row, &["offer", "offerParams", "orderUnit"]));
        push_opt_str(&mut self.tax, get_str(row, &["offer", "offerParams", "tax"]));
        push_opt_str(
            &mut self.description,
            get_str(row, &["offer", "offerParams", "description"]),
        );
        push_opt_str(&mut self.ean, get_str(row, &["offer", "offerParams", "ean"]));
        push_opt_str(&mut self.hs_code, get_str(row, &["offer", "offerParams", "hsCode"]));
        push_opt_str(
            &mut self.manufacturer_name,
            get_str(row, &["offer", "offerParams", "manufacturerName"]),
        );
        push_opt_str(
            &mut self.customer_article_number,
            get_str(row, &["offer", "offerParams", "customerArticleNumber"]),
        );
        push_opt_str(
            &mut self.manufacturer_article_number,
            get_str(row, &["offer", "offerParams", "manufacturerArticleNumber"]),
        );
        push_opt_str(
            &mut self.manufacturer_article_type,
            get_str(row, &["offer", "offerParams", "manufacturerArticleType"]),
        );
        push_opt_str(
            &mut self.vendor_remarks,
            get_str(row, &["offer", "offerParams", "vendorRemarks"]),
        );
        push_opt_i64(&mut self.minimum_order, get_i64(row, &["offer", "offerParams", "minimumOrder"]));
        push_opt_i64(
            &mut self.quantity_interval,
            get_i64(row, &["offer", "offerParams", "quantityInterval"]),
        );
        push_opt_str(&mut self.content_unit, get_str(row, &["offer", "offerParams", "contentUnit"]));
        push_opt_i64(&mut self.delivery_time, get_i64(row, &["offer", "offerParams", "deliveryTime"]));
        push_opt_str(
            &mut self.content_amount,
            get_str(row, &["offer", "offerParams", "contentAmount"]),
        );
        push_opt_str(
            &mut self.price_quantity,
            get_str(row, &["offer", "offerParams", "priceQuantity"]),
        );
        push_opt_bool(
            &mut self.core_sortiment,
            get_bool(row, &["offer", "offerParams", "coreSortiment"]),
        );
        push_opt_bool(&mut self.dependent, get_bool(row, &["offer", "offerParams", "dependent"]));
        push_opt_bool(&mut self.expired, get_bool(row, &["offer", "offerParams", "expired"]));

        self.keyword_count.append_value(array_len(row, &["offer", "offerParams", "keywords"]));
        self.feature_count.append_value(array_len(row, &["offer", "offerParams", "features"]));
        self.image_count.append_value(array_len(row, &["offer", "offerParams", "images"]));
        self.download_count.append_value(array_len(row, &["offer", "offerParams", "downloads"]));
        self.category_path_count.append_value(array_len(row, &["offer", "offerParams", "categoryPaths"]));
        self.pricing_count.append_value(array_len(row, &["pricings"]));
        self.marker_count.append_value(array_len(row, &["markers"]));
        self.customer_article_number_count
            .append_value(array_len(row, &["customerArticleNumbers"]));

        push_opt_str(
            &mut self.open_price,
            first_staggered_price(row, &["offer", "pricings", "open", "prices", "staggeredPrices"]),
        );
        push_opt_str(
            &mut self.open_currency,
            get_str(row, &["offer", "pricings", "open", "prices", "currencyCode"]),
        );
        push_opt_str(
            &mut self.closed_price,
            first_staggered_price(row, &["offer", "pricings", "closed", "prices", "staggeredPrices"]),
        );
        push_opt_str(
            &mut self.closed_currency,
            get_str(row, &["offer", "pricings", "closed", "prices", "currencyCode"]),
        );

        if include_json_blobs {
            push_opt_json(&mut self.thumbnail_json, get_path(row, &["offer", "offerParams", "thumbnail"]));
            push_opt_json(&mut self.features_json, get_path(row, &["offer", "offerParams", "features"]));
            push_opt_json(&mut self.keywords_json, get_path(row, &["offer", "offerParams", "keywords"]));
            push_opt_json(&mut self.images_json, get_path(row, &["offer", "offerParams", "images"]));
            push_opt_json(&mut self.downloads_json, get_path(row, &["offer", "offerParams", "downloads"]));
            push_opt_json(
                &mut self.eclass_groups_json,
                get_path(row, &["offer", "offerParams", "eclassGroups"]),
            );
            push_opt_json(
                &mut self.category_paths_json,
                get_path(row, &["offer", "offerParams", "categoryPaths"]),
            );
            push_opt_json(&mut self.pricings_json, get_path(row, &["pricings"]));
            push_opt_json(
                &mut self.related_article_numbers_json,
                get_path(row, &["offer", "relatedArticleNumbers"]),
            );
            push_opt_json(&mut self.markers_json, get_path(row, &["markers"]));
            push_opt_json(
                &mut self.customer_article_numbers_json,
                get_path(row, &["customerArticleNumbers"]),
            );
        } else {
            self.thumbnail_json.append_null();
            self.features_json.append_null();
            self.keywords_json.append_null();
            self.images_json.append_null();
            self.downloads_json.append_null();
            self.eclass_groups_json.append_null();
            self.category_paths_json.append_null();
            self.pricings_json.append_null();
            self.related_article_numbers_json.append_null();
            self.markers_json.append_null();
            self.customer_article_numbers_json.append_null();
        }

        if include_raw_json {
            match serde_json::to_string(row) {
                Ok(json) => self.raw_json.append_value(&json),
                Err(_) => self.raw_json.append_null(),
            }
        } else {
            self.raw_json.append_null();
        }

        self.rows += 1;
    }

    fn append_fast(&mut self, source_file_id: i32, row: &ExportRow) {
        self.source_file_id.append_value(source_file_id);

        let offer = row.offer.as_ref();
        let params = offer.and_then(|offer| offer.offer_params.as_ref());
        let pricings = offer.and_then(|offer| offer.pricings.as_ref());
        let open_prices = pricings.and_then(|pricings| pricings.open.as_ref()).and_then(|pricing| pricing.prices.as_ref());
        let closed_prices = pricings.and_then(|pricings| pricings.closed.as_ref()).and_then(|pricing| pricing.prices.as_ref());

        push_opt_str(&mut self.offer_id, row.id.as_ref().map(|v| v.value));
        push_opt_str(&mut self.article_number, row.article_number);
        push_opt_str(&mut self.vendor_id, row.vendor_id.as_ref().map(|v| v.value.base64));
        push_opt_str(
            &mut self.catalog_version_id,
            row.catalog_version_id.as_ref().map(|v| v.value.base64),
        );
        push_opt_i64(&mut self.import_epoch, row.import_epoch.as_ref().and_then(MaybeI64::as_i64));
        push_opt_str(&mut self.inner_offer_id, offer.and_then(|v| v.id.as_ref().map(|v| v.value.base64)));
        push_opt_str(
            &mut self.inner_catalog_version_id,
            offer.and_then(|v| v.catalog_version_id.as_ref().map(|v| v.value.base64)),
        );

        push_opt_str(&mut self.name, params.and_then(|v| v.name));
        push_opt_str(&mut self.order_unit, params.and_then(|v| v.order_unit));
        push_opt_str(&mut self.tax, params.and_then(|v| v.tax));
        push_opt_str(&mut self.description, params.and_then(|v| v.description));
        push_opt_str(&mut self.ean, params.and_then(|v| v.ean));
        push_opt_str(&mut self.hs_code, params.and_then(|v| v.hs_code));
        push_opt_str(&mut self.manufacturer_name, params.and_then(|v| v.manufacturer_name));
        push_opt_str(
            &mut self.customer_article_number,
            params.and_then(|v| v.customer_article_number),
        );
        push_opt_str(
            &mut self.manufacturer_article_number,
            params.and_then(|v| v.manufacturer_article_number),
        );
        push_opt_str(
            &mut self.manufacturer_article_type,
            params.and_then(|v| v.manufacturer_article_type),
        );
        push_opt_str(&mut self.vendor_remarks, params.and_then(|v| v.vendor_remarks));
        push_opt_i64(
            &mut self.minimum_order,
            params.and_then(|v| v.minimum_order.as_ref()).and_then(MaybeI64::as_i64),
        );
        push_opt_i64(
            &mut self.quantity_interval,
            params.and_then(|v| v.quantity_interval.as_ref()).and_then(MaybeI64::as_i64),
        );
        push_opt_str(&mut self.content_unit, params.and_then(|v| v.content_unit));
        push_opt_i64(
            &mut self.delivery_time,
            params.and_then(|v| v.delivery_time.as_ref()).and_then(MaybeI64::as_i64),
        );
        push_opt_str(&mut self.content_amount, params.and_then(|v| v.content_amount));
        push_opt_str(&mut self.price_quantity, params.and_then(|v| v.price_quantity));
        push_opt_bool(&mut self.core_sortiment, params.and_then(|v| v.core_sortiment));
        push_opt_bool(&mut self.dependent, params.and_then(|v| v.dependent));
        push_opt_bool(&mut self.expired, params.and_then(|v| v.expired));

        self.keyword_count.append_value(params.map(|v| v.keywords.0).unwrap_or(0));
        self.feature_count.append_value(params.map(|v| v.features.0).unwrap_or(0));
        self.image_count.append_value(params.map(|v| v.images.0).unwrap_or(0));
        self.download_count.append_value(params.map(|v| v.downloads.0).unwrap_or(0));
        self.category_path_count.append_value(params.map(|v| v.category_paths.0).unwrap_or(0));
        self.pricing_count.append_value(0);
        self.marker_count.append_value(0);
        self.customer_article_number_count.append_value(0);

        push_opt_str(
            &mut self.open_price,
            open_prices.and_then(|prices| prices.staggered_prices.first()).and_then(|price| price.price),
        );
        push_opt_str(&mut self.open_currency, open_prices.and_then(|prices| prices.currency_code));
        push_opt_str(
            &mut self.closed_price,
            closed_prices.and_then(|prices| prices.staggered_prices.first()).and_then(|price| price.price),
        );
        push_opt_str(&mut self.closed_currency, closed_prices.and_then(|prices| prices.currency_code));

        self.thumbnail_json.append_null();
        self.features_json.append_null();
        self.keywords_json.append_null();
        self.images_json.append_null();
        self.downloads_json.append_null();
        self.eclass_groups_json.append_null();
        self.category_paths_json.append_null();
        self.pricings_json.append_null();
        self.related_article_numbers_json.append_null();
        self.markers_json.append_null();
        self.customer_article_numbers_json.append_null();
        self.raw_json.append_null();

        self.rows += 1;
    }

    fn into_record_batch(mut self, schema: SchemaRef) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.source_file_id.finish()),
            Arc::new(self.offer_id.finish()),
            Arc::new(self.article_number.finish()),
            Arc::new(self.vendor_id.finish()),
            Arc::new(self.catalog_version_id.finish()),
            Arc::new(self.import_epoch.finish()),
            Arc::new(self.inner_offer_id.finish()),
            Arc::new(self.inner_catalog_version_id.finish()),
            Arc::new(self.name.finish()),
            Arc::new(self.order_unit.finish()),
            Arc::new(self.tax.finish()),
            Arc::new(self.description.finish()),
            Arc::new(self.ean.finish()),
            Arc::new(self.hs_code.finish()),
            Arc::new(self.manufacturer_name.finish()),
            Arc::new(self.customer_article_number.finish()),
            Arc::new(self.manufacturer_article_number.finish()),
            Arc::new(self.manufacturer_article_type.finish()),
            Arc::new(self.vendor_remarks.finish()),
            Arc::new(self.minimum_order.finish()),
            Arc::new(self.quantity_interval.finish()),
            Arc::new(self.content_unit.finish()),
            Arc::new(self.delivery_time.finish()),
            Arc::new(self.content_amount.finish()),
            Arc::new(self.price_quantity.finish()),
            Arc::new(self.core_sortiment.finish()),
            Arc::new(self.dependent.finish()),
            Arc::new(self.expired.finish()),
            Arc::new(self.keyword_count.finish()),
            Arc::new(self.feature_count.finish()),
            Arc::new(self.image_count.finish()),
            Arc::new(self.download_count.finish()),
            Arc::new(self.category_path_count.finish()),
            Arc::new(self.pricing_count.finish()),
            Arc::new(self.marker_count.finish()),
            Arc::new(self.customer_article_number_count.finish()),
            Arc::new(self.open_price.finish()),
            Arc::new(self.open_currency.finish()),
            Arc::new(self.closed_price.finish()),
            Arc::new(self.closed_currency.finish()),
            Arc::new(self.thumbnail_json.finish()),
            Arc::new(self.features_json.finish()),
            Arc::new(self.keywords_json.finish()),
            Arc::new(self.images_json.finish()),
            Arc::new(self.downloads_json.finish()),
            Arc::new(self.eclass_groups_json.finish()),
            Arc::new(self.category_paths_json.finish()),
            Arc::new(self.pricings_json.finish()),
            Arc::new(self.related_article_numbers_json.finish()),
            Arc::new(self.markers_json.finish()),
            Arc::new(self.customer_article_numbers_json.finish()),
            Arc::new(self.raw_json.finish()),
        ];

        Ok(RecordBatch::try_new(schema, columns)?)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.threads == 0 {
        bail!("--threads must be >= 1");
    }
    if args.batch_rows == 0 {
        bail!("--batch-rows must be >= 1");
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .context("failed to build rayon thread pool")?;

    prepare_output_dir(&args.output_dir, args.overwrite)?;

    let start = Instant::now();
    let files = collect_files(&args.input_glob, args.limit, args.selection_mode)?;
    if files.is_empty() {
        bail!("no files matched {}", args.input_glob);
    }

    write_manifest(&args.output_dir, &files)?;

    let shard_count = args
        .shards
        .unwrap_or_else(|| files.len().min(args.threads.saturating_mul(4).max(1)));
    let buckets = bucketize(files, shard_count.max(1));
    let schema = schema();

    eprintln!(
        "converting {} files into {} parquet shards using {} threads",
        buckets.iter().map(|bucket| bucket.len()).sum::<usize>(),
        buckets.len(),
        args.threads
    );

    let stats: Vec<ShardStats> = buckets
        .into_par_iter()
        .enumerate()
        .map(|(idx, bucket)| convert_shard(idx, bucket, &args, schema.clone()))
        .collect::<Result<Vec<_>>>()?;

    let elapsed = start.elapsed().as_secs_f64();
    let total = stats.into_iter().fold(ShardStats::default(), |mut acc, shard| {
        acc.files += shard.files;
        acc.rows += shard.rows;
        acc.compressed_bytes += shard.compressed_bytes;
        acc.parquet_bytes += shard.parquet_bytes;
        acc
    });

    eprintln!(
        "done files={} rows={} input={} output={} elapsed={:.2}s input_throughput={}/s",
        total.files,
        total.rows,
        human_bytes(total.compressed_bytes),
        human_bytes(total.parquet_bytes),
        elapsed,
        human_bytes((total.compressed_bytes as f64 / elapsed.max(0.001)) as u64),
    );

    Ok(())
}

fn default_threads() -> usize {
    num_cpus::get().max(1)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source_file_id", DataType::Int32, false),
        Field::new("offer_id", DataType::LargeUtf8, true),
        Field::new("article_number", DataType::LargeUtf8, true),
        Field::new("vendor_id", DataType::LargeUtf8, true),
        Field::new("catalog_version_id", DataType::LargeUtf8, true),
        Field::new("import_epoch", DataType::Int64, true),
        Field::new("inner_offer_id", DataType::LargeUtf8, true),
        Field::new("inner_catalog_version_id", DataType::LargeUtf8, true),
        Field::new("name", DataType::LargeUtf8, true),
        Field::new("order_unit", DataType::LargeUtf8, true),
        Field::new("tax", DataType::LargeUtf8, true),
        Field::new("description", DataType::LargeUtf8, true),
        Field::new("ean", DataType::LargeUtf8, true),
        Field::new("hs_code", DataType::LargeUtf8, true),
        Field::new("manufacturer_name", DataType::LargeUtf8, true),
        Field::new("customer_article_number", DataType::LargeUtf8, true),
        Field::new("manufacturer_article_number", DataType::LargeUtf8, true),
        Field::new("manufacturer_article_type", DataType::LargeUtf8, true),
        Field::new("vendor_remarks", DataType::LargeUtf8, true),
        Field::new("minimum_order", DataType::Int64, true),
        Field::new("quantity_interval", DataType::Int64, true),
        Field::new("content_unit", DataType::LargeUtf8, true),
        Field::new("delivery_time", DataType::Int64, true),
        Field::new("content_amount", DataType::LargeUtf8, true),
        Field::new("price_quantity", DataType::LargeUtf8, true),
        Field::new("core_sortiment", DataType::Boolean, true),
        Field::new("dependent", DataType::Boolean, true),
        Field::new("expired", DataType::Boolean, true),
        Field::new("keyword_count", DataType::Int32, false),
        Field::new("feature_count", DataType::Int32, false),
        Field::new("image_count", DataType::Int32, false),
        Field::new("download_count", DataType::Int32, false),
        Field::new("category_path_count", DataType::Int32, false),
        Field::new("pricing_count", DataType::Int32, false),
        Field::new("marker_count", DataType::Int32, false),
        Field::new("customer_article_number_count", DataType::Int32, false),
        Field::new("open_price", DataType::LargeUtf8, true),
        Field::new("open_currency", DataType::LargeUtf8, true),
        Field::new("closed_price", DataType::LargeUtf8, true),
        Field::new("closed_currency", DataType::LargeUtf8, true),
        Field::new("thumbnail_json", DataType::LargeUtf8, true),
        Field::new("features_json", DataType::LargeUtf8, true),
        Field::new("keywords_json", DataType::LargeUtf8, true),
        Field::new("images_json", DataType::LargeUtf8, true),
        Field::new("downloads_json", DataType::LargeUtf8, true),
        Field::new("eclass_groups_json", DataType::LargeUtf8, true),
        Field::new("category_paths_json", DataType::LargeUtf8, true),
        Field::new("pricings_json", DataType::LargeUtf8, true),
        Field::new("related_article_numbers_json", DataType::LargeUtf8, true),
        Field::new("markers_json", DataType::LargeUtf8, true),
        Field::new("customer_article_numbers_json", DataType::LargeUtf8, true),
        Field::new("raw_json", DataType::LargeUtf8, true),
    ]))
}

fn prepare_output_dir(output_dir: &Path, overwrite: bool) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    if overwrite {
        for entry in fs::read_dir(output_dir)
            .with_context(|| format!("failed to read {}", output_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) == Some("parquet") || path.file_name().and_then(|x| x.to_str()) == Some("manifest.tsv") {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
        }
    }

    Ok(())
}

fn collect_files(pattern: &str, limit: usize, selection_mode: SelectionMode) -> Result<Vec<FileEntry>> {
    let mut files = Vec::new();

    for entry in glob(pattern).with_context(|| format!("invalid glob {pattern}"))? {
        let path = entry?;
        let meta = fs::metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?;
        if meta.is_file() {
            files.push(FileEntry {
                file_id: 0,
                path,
                compressed_size: meta.len(),
            });
        }
    }

    match selection_mode {
        SelectionMode::Lexical => files.sort_by(|a, b| a.path.cmp(&b.path)),
        SelectionMode::Largest => files.sort_by(|a, b| b.compressed_size.cmp(&a.compressed_size).then_with(|| a.path.cmp(&b.path))),
    }

    if limit > 0 && files.len() > limit {
        files.truncate(limit);
    }

    for (idx, file) in files.iter_mut().enumerate() {
        file.file_id = idx as i32;
    }

    Ok(files)
}

fn write_manifest(output_dir: &Path, files: &[FileEntry]) -> Result<()> {
    let manifest_path = output_dir.join("manifest.tsv");
    let mut writer = BufWriter::with_capacity(1024 * 1024, File::create(&manifest_path)?);
    use std::io::Write;
    writeln!(writer, "source_file_id\tpath\tcompressed_bytes")?;
    for file in files {
        writeln!(writer, "{}\t{}\t{}", file.file_id, file.path.display(), file.compressed_size)?;
    }
    writer.flush()?;
    Ok(())
}

fn bucketize(files: Vec<FileEntry>, shards: usize) -> Vec<Vec<FileEntry>> {
    let mut buckets: Vec<Vec<FileEntry>> = (0..shards).map(|_| Vec::new()).collect();
    let mut sizes = vec![0u64; shards];

    for file in files {
        let (idx, _) = sizes
            .iter()
            .enumerate()
            .min_by_key(|(_, size)| **size)
            .unwrap();
        sizes[idx] = sizes[idx].saturating_add(file.compressed_size);
        buckets[idx].push(file);
    }

    buckets.into_iter().filter(|bucket| !bucket.is_empty()).collect()
}

fn convert_shard(shard_idx: usize, files: Vec<FileEntry>, args: &Args, schema: SchemaRef) -> Result<ShardStats> {
    let out_path = args.output_dir.join(format!("part-{shard_idx:05}.parquet"));
    let file = File::create(&out_path)
        .with_context(|| format!("failed to create {}", out_path.display()))?;
    let sink = BufWriter::with_capacity(16 * 1024 * 1024, file);

    let props = WriterProperties::builder()
        .set_compression(args.compression.parquet())
        .set_statistics_enabled(EnabledStatistics::None)
        .set_max_row_group_size(args.row_group_rows)
        .build();

    let mut writer = ArrowWriter::try_new(sink, schema.clone(), Some(props))
        .with_context(|| format!("failed to open parquet writer for {}", out_path.display()))?;
    let mut batch = BatchBuilder::new(args.batch_rows);
    let mut rows = 0usize;
    let compressed_bytes = files.iter().map(|file| file.compressed_size).sum::<u64>();

    for file in &files {
        match args.input_format {
            InputFormat::Ndjson if !args.include_json_blobs && !args.include_raw_json => {
                rows += stream_ndjson_file(file, args, schema.clone(), &mut writer, &mut batch)?;
            }
            _ => {
                let parsed_rows = read_rows(&file.path, file.compressed_size)
                    .with_context(|| format!("failed to parse {}", file.path.display()))?;

                for row in &parsed_rows {
                    batch.append(
                        file.file_id,
                        row,
                        args.include_json_blobs,
                        args.include_raw_json,
                    );
                    rows += 1;

                    if batch.len() >= args.batch_rows {
                        let flushed = std::mem::replace(&mut batch, BatchBuilder::new(args.batch_rows));
                        writer.write(&flushed.into_record_batch(schema.clone())?)?;
                    }
                }
            }
        }
    }

    if batch.len() > 0 {
        writer.write(&batch.into_record_batch(schema)?)?;
    }

    writer.close()?;
    let parquet_bytes = fs::metadata(&out_path)
        .with_context(|| format!("failed to stat {}", out_path.display()))?
        .len();

    eprintln!(
        "finished {} files -> {} rows -> {} ({})",
        files.len(),
        rows,
        out_path.display(),
        human_bytes(parquet_bytes)
    );

    Ok(ShardStats {
        files: files.len(),
        rows,
        compressed_bytes,
        parquet_bytes,
    })
}

fn stream_ndjson_file(
    file: &FileEntry,
    args: &Args,
    schema: SchemaRef,
    writer: &mut ArrowWriter<BufWriter<File>>,
    batch: &mut BatchBuilder,
) -> Result<usize> {
    let file_handle = File::open(&file.path)
        .with_context(|| format!("failed to open {}", file.path.display()))?;
    let input = BufReader::with_capacity(4 * 1024 * 1024, file_handle);
    let decoder = GzDecoder::new(input);
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, decoder);
    let mut line = Vec::with_capacity(256 * 1024);
    let mut rows = 0usize;

    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        let start = line.iter().position(|b| !b.is_ascii_whitespace());
        let end = line.iter().rposition(|b| !b.is_ascii_whitespace());
        let (start, end) = match (start, end) {
            (Some(start), Some(end)) if start <= end => (start, end + 1),
            _ => continue,
        };

        let row = simd_json::serde::from_slice::<ExportRow>(&mut line[start..end])?;
        batch.append_fast(file.file_id, &row);
        rows += 1;

        if batch.len() >= args.batch_rows {
            let flushed = std::mem::replace(batch, BatchBuilder::new(args.batch_rows));
            writer.write(&flushed.into_record_batch(schema.clone())?)?;
        }
    }

    Ok(rows)
}

fn read_rows(path: &Path, compressed_size: u64) -> Result<Vec<Value>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::with_capacity(4 * 1024 * 1024, file);
    let mut decoder = GzDecoder::new(reader);

    let capacity = (compressed_size.saturating_mul(6)).min(512 * 1024 * 1024) as usize;
    let mut bytes = Vec::with_capacity(capacity.max(4096));
    decoder.read_to_end(&mut bytes)?;

    let trimmed = trim_ascii(&bytes);
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    match trimmed[0] {
        b'[' => parse_single_json(trimmed.to_vec()),
        b'{' => {
            let mut probe = trimmed.to_vec();
            match simd_json::serde::from_slice::<Value>(&mut probe) {
                Ok(Value::Array(rows)) => Ok(rows),
                Ok(Value::Object(mut obj)) => match obj.remove("records") {
                    Some(Value::Array(rows)) => Ok(rows),
                    _ => Ok(vec![Value::Object(obj)]),
                },
                Ok(other) => Ok(vec![other]),
                Err(_) => parse_ndjson(trimmed),
            }
        }
        _ => parse_ndjson(trimmed),
    }
}

fn parse_single_json(mut bytes: Vec<u8>) -> Result<Vec<Value>> {
    match simd_json::serde::from_slice::<Value>(&mut bytes)? {
        Value::Array(rows) => Ok(rows),
        Value::Object(mut obj) => match obj.remove("records") {
            Some(Value::Array(rows)) => Ok(rows),
            _ => Ok(vec![Value::Object(obj)]),
        },
        other => Ok(vec![other]),
    }
}

fn parse_ndjson(bytes: &[u8]) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        let trimmed = trim_ascii(line);
        if trimmed.is_empty() {
            continue;
        }
        let mut owned = trimmed.to_vec();
        let row = simd_json::serde::from_slice::<Value>(&mut owned)?;
        rows.push(row);
    }
    Ok(rows)
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0usize;
    let mut end = bytes.len();

    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    &bytes[start..end]
}

fn get_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cur = value;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

fn get_str<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    get_path(value, path)?.as_str()
}

fn get_i64(value: &Value, path: &[&str]) -> Option<i64> {
    let v = get_path(value, path)?;
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_u64().and_then(|x| i64::try_from(x).ok())),
        Value::String(s) => s.parse().ok(),
        Value::Object(obj) => obj
            .get("$numberInt")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .or_else(|| obj.get("$numberLong").and_then(|v| v.as_str()).and_then(|s| s.parse().ok())),
        _ => None,
    }
}

fn get_bool(value: &Value, path: &[&str]) -> Option<bool> {
    get_path(value, path)?.as_bool()
}

fn array_len(value: &Value, path: &[&str]) -> i32 {
    get_path(value, path)
        .and_then(|v| v.as_array())
        .map(|v| v.len() as i32)
        .unwrap_or(0)
}

fn first_staggered_price<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let prices = get_path(value, path)?.as_array()?;
    prices.first()?.get("price")?.as_str()
}

fn mongo_oid(value: Option<&Value>) -> Option<&str> {
    value?.get("$oid")?.as_str()
}

fn mongo_binary_base64(value: Option<&Value>) -> Option<&str> {
    value?.get("$binary")?.get("base64")?.as_str()
}

fn push_opt_str(builder: &mut LargeStringBuilder, value: Option<&str>) {
    match value {
        Some(value) => builder.append_value(value),
        None => builder.append_null(),
    }
}

fn push_opt_i64(builder: &mut Int64Builder, value: Option<i64>) {
    match value {
        Some(value) => builder.append_value(value),
        None => builder.append_null(),
    }
}

fn push_opt_bool(builder: &mut BooleanBuilder, value: Option<bool>) {
    match value {
        Some(value) => builder.append_value(value),
        None => builder.append_null(),
    }
}

fn push_opt_json(builder: &mut LargeStringBuilder, value: Option<&Value>) {
    match value {
        Some(Value::Null) | None => builder.append_null(),
        Some(other) => match serde_json::to_string(other) {
            Ok(json) => builder.append_value(&json),
            Err(_) => builder.append_null(),
        },
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2}{}", UNITS[unit])
}
