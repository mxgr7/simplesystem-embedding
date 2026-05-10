use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use arrow_array::builder::{
    Float32Builder, Float64Builder, Int32Builder, ListBuilder, StringBuilder, StructBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use base64::Engine;
use base64::engine::general_purpose::{STANDARD as BASE64_STANDARD, URL_SAFE_NO_PAD};
use bincode::{deserialize_from, serialize_into};
use clap::{ArgAction, Args as ClapArgs, Parser, Subcommand, ValueEnum};
use flate2::read::GzDecoder;
use glob::glob;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rayon::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use serde::de::{IgnoredAny, MapAccess, Visitor};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const CATALOG_CURRENCIES: [&str; 7] = ["eur", "chf", "huf", "pln", "gbp", "czk", "cny"];
const MAX_PRICE_SENTINEL: f64 = 3.4028234e38;
const DEFAULT_S2CLASS_CODE: i32 = 90909090;
const PATH_SEPARATOR: char = '¦';
const PATH_ESCAPE: char = '|';
const HASH_FIELD_SEP: u8 = 0;
const RELATIONSHIP_LIMIT: usize = 4096;
const MARKER_LIMIT: usize = 64;
const TEXT_CODES_LIMIT: usize = 8192;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputCompression {
    Snappy,
    Zstd,
    Uncompressed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
enum TempCompression {
    Uncompressed,
    Zstd,
}

impl OutputCompression {
    fn parquet(self) -> Compression {
        match self {
            Self::Snappy => Compression::SNAPPY,
            Self::Zstd => Compression::ZSTD(Default::default()),
            Self::Uncompressed => Compression::UNCOMPRESSED,
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about = "Materialize final articles/offer_rows parquet from local MongoDB gzipped NDJSON exports")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
    Run(Args),
    PartitionOffers(Args),
    PartitionPricings(Args),
    PartitionMarkers(Args),
    PartitionCans(Args),
    ProcessJoinBuckets(Args),
    RepartitionArticlePartials(Args),
    MaterializeArticles(Args),
}

#[derive(Debug, Clone, ClapArgs)]
struct Args {
    #[arg(long, default_value = "/data/mongodb-export-2026-03-04")]
    source_root: PathBuf,

    #[arg(long, default_value = "atlas-*.json.gz")]
    offers_glob: String,

    #[arg(long, default_value = "atlas-*.json.gz")]
    pricings_glob: String,

    #[arg(long, default_value = "atlas-*.json.gz")]
    markers_glob: String,

    #[arg(long, default_value = "atlas-*.json.gz")]
    cans_glob: String,

    #[arg(long, default_value = "/data/mongodb-export-2026-03-04/indexer_parquet_rust")]
    output_root: PathBuf,

    #[arg(long = "artifact-root", visible_alias = "temp-dir", default_value = "/data/mongodb-export-2026-03-04/indexer_parquet_rust_tmp")]
    temp_dir: PathBuf,

    #[arg(long, default_value_t = 80)]
    memory_limit_gb: usize,

    #[arg(long, default_value_t = default_threads())]
    threads: usize,

    #[arg(long, default_value_t = 0)]
    parser_workers: usize,

    #[arg(long, default_value_t = 0)]
    join_bucket_workers: usize,

    #[arg(long, default_value_t = 0)]
    article_bucket_workers: usize,

    #[arg(long)]
    join_buckets: Option<usize>,

    #[arg(long)]
    article_buckets: Option<usize>,

    #[arg(long, default_value_t = 65536)]
    batch_rows: usize,

    #[arg(long, default_value_t = 262144)]
    row_group_rows: usize,

    #[arg(long, value_enum, default_value_t = OutputCompression::Snappy)]
    compression: OutputCompression,

    #[arg(long, value_enum, default_value_t = TempCompression::Zstd)]
    temp_compression: TempCompression,

    #[arg(long, default_value_t = 1)]
    temp_zstd_level: i32,

    #[arg(long, default_value_t = 64)]
    max_open_temp_files: usize,

    #[arg(long)]
    offer_file_limit: Option<usize>,

    #[arg(long)]
    pricing_file_limit: Option<usize>,

    #[arg(long)]
    marker_file_limit: Option<usize>,

    #[arg(long)]
    can_file_limit: Option<usize>,

    #[arg(long, action = ArgAction::SetTrue)]
    overwrite: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    offers_only: bool,
}

#[derive(Debug, Clone)]
struct Settings {
    parser_workers: usize,
    join_bucket_workers: usize,
    article_bucket_workers: usize,
    join_buckets: usize,
    article_buckets: usize,
    offer_partition_buffer_bytes: usize,
    temp_compression: TempCompression,
    temp_zstd_level: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionArtifactMeta {
    join_buckets: usize,
    temp_compression: TempCompression,
    temp_zstd_level: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArticleBucketsArtifactMeta {
    article_buckets: usize,
    temp_compression: TempCompression,
    temp_zstd_level: i32,
}

#[derive(Debug, Clone)]
struct CollectionFiles {
    offers: Vec<PathBuf>,
    pricings: Vec<PathBuf>,
    markers: Vec<PathBuf>,
    cans: Vec<PathBuf>,
}

#[derive(Debug, Default, Serialize)]
struct RunStats {
    offers_files: usize,
    pricings_files: usize,
    markers_files: usize,
    cans_files: usize,
    offers_rows: usize,
    pricing_rows: usize,
    marker_rows: usize,
    can_rows: usize,
    offer_rows_rows: usize,
    article_partial_rows: usize,
    article_rows_rows: usize,
}

#[derive(Debug, Clone, Copy, Eq)]
struct JoinKey {
    vendor: [u8; 16],
    article_hash: u64,
}

impl PartialEq for JoinKey {
    fn eq(&self, other: &Self) -> bool {
        self.vendor == other.vendor && self.article_hash == other.article_hash
    }
}

impl Hash for JoinKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vendor.hash(state);
        self.article_hash.hash(state);
    }
}

impl JoinKey {
    fn from_parts(vendor: [u8; 16], article_number: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        article_number.hash(&mut hasher);
        Self {
            vendor,
            article_hash: hasher.finish(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TempPricingDetails {
    source_price_list_id: Option<[u8; 16]>,
    type_name: Option<String>,
    currency_code: Option<String>,
    staggered_prices: Vec<TempStaggeredPrice>,
    price_quantity: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TempStaggeredPrice {
    min_quantity: Option<String>,
    price: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TempCanPair {
    value: String,
    version_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OfferProjectedTemp {
    vendor: [u8; 16],
    article_number: String,
    catalog_version: [u8; 16],
    article_hash: String,
    id: String,
    vendor_id: String,
    catalog_version_id: String,
    name: String,
    manufacturer_name: String,
    ean: String,
    delivery_time_days_max: i32,
    eclass5_code: Vec<i32>,
    eclass7_code: Vec<i32>,
    s2class_code: Vec<i32>,
    relationship_accessory_for: Vec<String>,
    relationship_spare_part_for: Vec<String>,
    relationship_similar_to: Vec<String>,
    category_l1: Vec<String>,
    category_l2: Vec<String>,
    category_l3: Vec<String>,
    category_l4: Vec<String>,
    category_l5: Vec<String>,
    features: Vec<String>,
    inline_pricing_open: Option<TempPricingDetails>,
    inline_pricing_closed: Option<TempPricingDetails>,
    inline_can_pair: Option<TempCanPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PricingTemp {
    vendor: [u8; 16],
    article_number: String,
    pricing: TempPricingDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarkerTemp {
    vendor: [u8; 16],
    article_number: String,
    source: [u8; 16],
    enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CanTemp {
    vendor: [u8; 16],
    article_number: String,
    version: [u8; 16],
    value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectedPrice {
    price: f64,
    currency: String,
    priority: i32,
    source_price_list_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomerArticleNumberEntry {
    value: String,
    version_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArticlePartialRow {
    article_hash: String,
    name: String,
    manufacturer_name: String,
    category_l1: Vec<String>,
    category_l2: Vec<String>,
    category_l3: Vec<String>,
    category_l4: Vec<String>,
    category_l5: Vec<String>,
    eclass5_code: Vec<i32>,
    eclass7_code: Vec<i32>,
    s2class_code: Vec<i32>,
    eans: Vec<String>,
    article_numbers: Vec<String>,
    customer_article_numbers: Vec<CustomerArticleNumberEntry>,
    eur_price_min: f64,
    eur_price_max: f64,
    chf_price_min: f64,
    chf_price_max: f64,
    huf_price_min: f64,
    huf_price_max: f64,
    pln_price_min: f64,
    pln_price_max: f64,
    gbp_price_min: f64,
    gbp_price_max: f64,
    czk_price_min: f64,
    czk_price_max: f64,
    cny_price_min: f64,
    cny_price_max: f64,
}

#[derive(Debug, Clone)]
struct PriceEnvelope {
    mins: [f64; 7],
    maxs: [f64; 7],
    price_list_ids: Vec<String>,
    currencies: Vec<String>,
}

#[derive(Debug, Clone)]
struct ArticlePartialAcc {
    article_hash: String,
    name: String,
    manufacturer_name: String,
    category_l1: Vec<String>,
    category_l2: Vec<String>,
    category_l3: Vec<String>,
    category_l4: Vec<String>,
    category_l5: Vec<String>,
    eclass5_code: Vec<i32>,
    eclass7_code: Vec<i32>,
    s2class_code: Vec<i32>,
    eans: BTreeSet<String>,
    article_numbers: BTreeSet<String>,
    customer_article_numbers: BTreeMap<String, BTreeSet<String>>,
    mins: [f64; 7],
    maxs: [f64; 7],
}

#[derive(Debug, Clone, Deserialize)]
struct MongoBinary {
    #[serde(rename = "$binary")]
    value: MongoBinaryInner,
}

#[derive(Debug, Clone, Deserialize)]
struct MongoBinaryInner {
    base64: String,
}

#[derive(Debug)]
struct MaybeI64(i64);

impl<'de> Deserialize<'de> for MaybeI64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MaybeI64Visitor;

        impl<'de> Visitor<'de> for MaybeI64Visitor {
            type Value = MaybeI64;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an integer-like value")
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(MaybeI64(v))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i64::try_from(v)
                    .map(MaybeI64)
                    .map_err(|_| E::custom("u64 out of range"))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                v.parse::<i64>()
                    .map(MaybeI64)
                    .map_err(|_| E::custom("invalid integer string"))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "$numberInt" | "$numberLong" => {
                            let value = map.next_value::<String>()?;
                            let parsed = value.parse::<i64>().map_err(|_| serde::de::Error::custom("invalid wrapped integer"))?;
                            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                            return Ok(MaybeI64(parsed));
                        }
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Err(serde::de::Error::custom("missing wrapped integer"))
            }
        }

        deserializer.deserialize_any(MaybeI64Visitor)
    }
}

fn deserialize_stringish_opt<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StringishVisitor;

    impl<'de> Visitor<'de> for StringishVisitor {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("string-like optional value")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserialize_stringish_opt(deserializer)
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v))
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            while let Some(key) = map.next_key::<&str>()? {
                match key {
                    "$numberInt" | "$numberLong" => {
                        let value = map.next_value::<String>()?;
                        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                        return Ok(Some(value));
                    }
                    _ => {
                        let _ = map.next_value::<IgnoredAny>()?;
                    }
                }
            }
            Ok(None)
        }
    }

    deserializer.deserialize_any(StringishVisitor)
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct EclassGroups {
    #[serde(default, rename = "ECLASS_5_1")]
    eclass_5_1: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_6")]
    eclass_6: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_7_1")]
    eclass_7_1: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_8")]
    eclass_8: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_9")]
    eclass_9: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_10")]
    eclass_10: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_11")]
    eclass_11: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_12")]
    eclass_12: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_13")]
    eclass_13: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_14")]
    eclass_14: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_15")]
    eclass_15: Vec<CodeI32>,
    #[serde(default, rename = "ECLASS_16")]
    eclass_16: Vec<CodeI32>,
    #[serde(default, rename = "S2CLASS")]
    s2class: Vec<CodeI32>,
}

impl EclassGroups {
    fn highest_non_s2(&self) -> Option<(usize, &[CodeI32])> {
        if !self.eclass_16.is_empty() { return Some((16, &self.eclass_16)); }
        if !self.eclass_15.is_empty() { return Some((15, &self.eclass_15)); }
        if !self.eclass_14.is_empty() { return Some((14, &self.eclass_14)); }
        if !self.eclass_13.is_empty() { return Some((13, &self.eclass_13)); }
        if !self.eclass_12.is_empty() { return Some((12, &self.eclass_12)); }
        if !self.eclass_11.is_empty() { return Some((11, &self.eclass_11)); }
        if !self.eclass_10.is_empty() { return Some((10, &self.eclass_10)); }
        if !self.eclass_9.is_empty() { return Some((9, &self.eclass_9)); }
        if !self.eclass_8.is_empty() { return Some((8, &self.eclass_8)); }
        if !self.eclass_7_1.is_empty() { return Some((7, &self.eclass_7_1)); }
        if !self.eclass_6.is_empty() { return Some((6, &self.eclass_6)); }
        if !self.eclass_5_1.is_empty() { return Some((5, &self.eclass_5_1)); }
        None
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
struct CodeI32(i32);

impl<'de> Deserialize<'de> for CodeI32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CodeI32Visitor;

        impl<'de> Visitor<'de> for CodeI32Visitor {
            type Value = CodeI32;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an int-like eclass code")
            }

            fn visit_i64<E>(self, value: i64) -> Result<CodeI32, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(value)
                    .map(CodeI32)
                    .map_err(|_| E::custom("code out of i32 range"))
            }

            fn visit_u64<E>(self, value: u64) -> Result<CodeI32, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(value)
                    .map(CodeI32)
                    .map_err(|_| E::custom("code out of i32 range"))
            }

            fn visit_str<E>(self, value: &str) -> Result<CodeI32, E>
            where
                E: serde::de::Error,
            {
                value.parse::<i32>().map(CodeI32).map_err(|_| E::custom("invalid code string"))
            }

            fn visit_map<A>(self, mut map: A) -> Result<CodeI32, A::Error>
            where
                A: MapAccess<'de>,
            {
                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "$numberInt" | "$numberLong" => {
                            let value = map.next_value::<String>()?;
                            let parsed = value.parse::<i32>().map_err(|_| serde::de::Error::custom("invalid wrapped code"))?;
                            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                            return Ok(CodeI32(parsed));
                        }
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Err(serde::de::Error::custom("missing wrapped code"))
            }
        }

        deserializer.deserialize_any(CodeI32Visitor)
    }
}

#[derive(Debug, Deserialize)]
struct RawOfferExport {
    #[serde(rename = "articleNumber")]
    article_number: String,
    #[serde(rename = "vendorId")]
    vendor_id: MongoBinary,
    #[serde(rename = "catalogVersionId")]
    catalog_version_id: MongoBinary,
    offer: RawInnerOffer,
}

#[derive(Debug, Deserialize)]
struct RawInnerOffer {
    #[serde(rename = "offerParams")]
    offer_params: RawOfferParams,
    #[serde(default)]
    pricings: RawInlinePricings,
    #[serde(default, rename = "relatedArticleNumbers")]
    related_article_numbers: RawRelatedArticleNumbers,
}

#[derive(Debug, Default, Deserialize)]
struct RawRelatedArticleNumbers {
    #[serde(default, rename = "accessoryFor")]
    accessory_for: Vec<String>,
    #[serde(default, rename = "sparePartFor")]
    spare_part_for: Vec<String>,
    #[serde(default, rename = "similarTo")]
    similar_to: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawOfferParams {
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "manufacturerName")]
    manufacturer_name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    ean: Option<String>,
    #[serde(default, rename = "customerArticleNumber")]
    customer_article_number: Option<String>,
    #[serde(default, rename = "deliveryTime")]
    delivery_time: Option<MaybeI64>,
    #[serde(default, rename = "manufacturerArticleNumber")]
    manufacturer_article_number: Option<String>,
    #[serde(default, rename = "manufacturerArticleType")]
    manufacturer_article_type: Option<String>,
    #[serde(default, rename = "categoryPaths")]
    category_paths: Vec<RawCategoryPath>,
    #[serde(default)]
    features: Vec<RawFeature>,
    #[serde(default, rename = "eclassGroups")]
    eclass_groups: EclassGroups,
}

#[derive(Debug, Default, Deserialize)]
struct RawInlinePricings {
    #[serde(default)]
    open: Option<RawPricingDetails>,
    #[serde(default)]
    closed: Option<RawPricingDetails>,
}

#[derive(Debug, Deserialize)]
struct RawFeature {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    values: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawCategoryPath {
    #[serde(default)]
    elements: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawPricingExport {
    #[serde(rename = "articleNumber")]
    article_number: String,
    #[serde(rename = "vendorId")]
    vendor_id: MongoBinary,
    #[serde(rename = "pricingDetails")]
    pricing_details: RawPricingDetails,
}

#[derive(Debug, Deserialize)]
struct RawMarkerExport {
    #[serde(rename = "articleNumber")]
    article_number: String,
    #[serde(rename = "vendorId")]
    vendor_id: MongoBinary,
    #[serde(rename = "coreArticleListSourceId")]
    source_id: MongoBinary,
    #[serde(rename = "coreArticleMarker")]
    marker: bool,
}

#[derive(Debug, Deserialize)]
struct RawCanExport {
    #[serde(rename = "articleNumber")]
    article_number: String,
    #[serde(rename = "vendorId")]
    vendor_id: MongoBinary,
    #[serde(rename = "customerArticleNumbersListVersionId")]
    version_id: MongoBinary,
    #[serde(rename = "customerArticleNumber")]
    customer_article_number: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawPricingDetails {
    #[serde(default, rename = "sourcePriceListId")]
    source_price_list_id: Option<MongoBinary>,
    #[serde(default, rename = "type")]
    type_name: Option<String>,
    #[serde(default)]
    prices: Option<RawPrices>,
    #[serde(default, deserialize_with = "deserialize_stringish_opt", rename = "priceQuantity")]
    price_quantity: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawPrices {
    #[serde(default, rename = "currencyCode")]
    currency_code: Option<String>,
    #[serde(default, rename = "staggeredPrices")]
    staggered_prices: Vec<RawStaggeredPrice>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawStaggeredPrice {
    #[serde(default, deserialize_with = "deserialize_stringish_opt", rename = "minQuantity")]
    min_quantity: Option<String>,
    #[serde(default, deserialize_with = "deserialize_stringish_opt")]
    price: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run_all(&args),
        Command::PartitionOffers(args) => run_partition_offers_phase(&args),
        Command::PartitionPricings(args) => run_partition_pricings_phase(&args),
        Command::PartitionMarkers(args) => run_partition_markers_phase(&args),
        Command::PartitionCans(args) => run_partition_cans_phase(&args),
        Command::ProcessJoinBuckets(args) => run_process_join_buckets_phase(&args),
        Command::RepartitionArticlePartials(args) => run_repartition_article_partials_phase(&args),
        Command::MaterializeArticles(args) => run_materialize_articles_phase(&args),
    }
}

fn run_all(args: &Args) -> Result<()> {
    let started = Instant::now();
    prepare_run_roots(args)?;

    let settings = build_settings(args);
    announce_settings(args, &settings);

    let files = discover_files(args)?;
    write_input_manifests(&args.output_root, &files)?;
    write_input_manifests(&args.temp_dir, &files)?;
    println!(
        "files: offers={} pricings={} markers={} cans={}",
        files.offers.len(), files.pricings.len(), files.markers.len(), files.cans.len()
    );

    ensure_partition_artifact_meta(&args.temp_dir, &settings)?;

    let s2_mappings = Arc::new(load_s2_mappings()?);
    let offer_schema = offer_rows_schema();
    let article_schema = articles_schema();

    let mut stats = RunStats {
        offers_files: files.offers.len(),
        pricings_files: files.pricings.len(),
        markers_files: files.markers.len(),
        cans_files: files.cans.len(),
        ..Default::default()
    };

    let t0 = Instant::now();
    stats.offers_rows = partition_offers(args, &settings, &files.offers, s2_mappings.clone())?;
    println!("partitioned offers in {:.2}s", t0.elapsed().as_secs_f64());

    if !args.offers_only {
        let t0 = Instant::now();
        stats.pricing_rows = partition_pricings(args, &settings, &files.pricings)?;
        println!("partitioned pricings in {:.2}s", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        stats.marker_rows = partition_markers(args, &settings, &files.markers)?;
        println!("partitioned markers in {:.2}s", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        stats.can_rows = partition_cans(args, &settings, &files.cans)?;
        println!("partitioned cans in {:.2}s", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let (offer_rows_rows, article_partial_rows) = process_join_buckets(
            args,
            &settings,
            offer_schema,
            article_schema.clone(),
        )?;
        stats.offer_rows_rows = offer_rows_rows;
        stats.article_partial_rows = article_partial_rows;
        println!("processed join buckets in {:.2}s", t0.elapsed().as_secs_f64());

        ensure_article_buckets_artifact_meta(&args.temp_dir, &settings)?;

        let t0 = Instant::now();
        repartition_article_partials(args, &settings)?;
        println!("repartitioned article partials in {:.2}s", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        stats.article_rows_rows = materialize_articles(args, &settings, article_schema)?;
        println!("materialized articles in {:.2}s", t0.elapsed().as_secs_f64());
    }

    let report_path = args.output_root.join("run_stats.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&stats)?)?;
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    println!("report: {}", report_path.display());
    println!("artifact_root: {}", args.temp_dir.display());
    println!("stats: {}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}

fn run_partition_offers_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let settings = build_settings(args);
    announce_settings(args, &settings);
    ensure_partition_artifact_meta(&args.temp_dir, &settings)?;
    prepare_phase_dir(&args.temp_dir.join("partition/offers"), args.overwrite, "offers partition artifact")?;
    let files = discover_offer_files(args)?;
    write_manifest(args.temp_dir.join("inputs/offers.txt"), &files)?;
    println!("files: offers={}", files.len());
    let s2_mappings = Arc::new(load_s2_mappings()?);
    let t0 = Instant::now();
    let rows = partition_offers(args, &settings, &files, s2_mappings)?;
    println!("partitioned offers in {:.2}s", t0.elapsed().as_secs_f64());
    println!("rows: {}", rows);
    println!("artifact_root: {}", args.temp_dir.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_partition_pricings_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let settings = build_settings(args);
    announce_settings(args, &settings);
    ensure_partition_artifact_meta(&args.temp_dir, &settings)?;
    prepare_phase_dir(&args.temp_dir.join("partition/pricings"), args.overwrite, "pricings partition artifact")?;
    let files = discover_pricing_files(args)?;
    write_manifest(args.temp_dir.join("inputs/pricings.txt"), &files)?;
    println!("files: pricings={}", files.len());
    let t0 = Instant::now();
    let rows = partition_pricings(args, &settings, &files)?;
    println!("partitioned pricings in {:.2}s", t0.elapsed().as_secs_f64());
    println!("rows: {}", rows);
    println!("artifact_root: {}", args.temp_dir.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_partition_markers_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let settings = build_settings(args);
    announce_settings(args, &settings);
    ensure_partition_artifact_meta(&args.temp_dir, &settings)?;
    prepare_phase_dir(&args.temp_dir.join("partition/markers"), args.overwrite, "markers partition artifact")?;
    let files = discover_marker_files(args)?;
    write_manifest(args.temp_dir.join("inputs/markers.txt"), &files)?;
    println!("files: markers={}", files.len());
    let t0 = Instant::now();
    let rows = partition_markers(args, &settings, &files)?;
    println!("partitioned markers in {:.2}s", t0.elapsed().as_secs_f64());
    println!("rows: {}", rows);
    println!("artifact_root: {}", args.temp_dir.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_partition_cans_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let settings = build_settings(args);
    announce_settings(args, &settings);
    ensure_partition_artifact_meta(&args.temp_dir, &settings)?;
    prepare_phase_dir(&args.temp_dir.join("partition/cans"), args.overwrite, "customer article number partition artifact")?;
    let files = discover_can_files(args)?;
    write_manifest(args.temp_dir.join("inputs/cans.txt"), &files)?;
    println!("files: cans={}", files.len());
    let t0 = Instant::now();
    let rows = partition_cans(args, &settings, &files)?;
    println!("partitioned cans in {:.2}s", t0.elapsed().as_secs_f64());
    println!("rows: {}", rows);
    println!("artifact_root: {}", args.temp_dir.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_process_join_buckets_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let mut settings = build_settings(args);
    let partition_meta = load_partition_artifact_meta(&args.temp_dir)?;
    apply_partition_artifact_meta(&mut settings, &partition_meta);
    announce_settings(args, &settings);
    prepare_phase_dir(&args.output_root.join("offer_rows"), args.overwrite, "offer_rows output")?;
    prepare_phase_dir(&args.temp_dir.join("article_partials"), args.overwrite, "article partial artifact")?;
    let offer_schema = offer_rows_schema();
    let article_schema = articles_schema();
    let t0 = Instant::now();
    let (offer_rows_rows, article_partial_rows) = process_join_buckets(args, &settings, offer_schema, article_schema)?;
    println!("processed join buckets in {:.2}s", t0.elapsed().as_secs_f64());
    println!("offer_rows_rows: {}", offer_rows_rows);
    println!("article_partial_rows: {}", article_partial_rows);
    println!("artifact_root: {}", args.temp_dir.display());
    println!("output_root: {}", args.output_root.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_repartition_article_partials_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let settings = build_settings(args);
    announce_settings(args, &settings);
    ensure_article_buckets_artifact_meta(&args.temp_dir, &settings)?;
    prepare_phase_dir(&args.temp_dir.join("article_buckets"), args.overwrite, "article bucket artifact")?;
    let t0 = Instant::now();
    repartition_article_partials(args, &settings)?;
    println!("repartitioned article partials in {:.2}s", t0.elapsed().as_secs_f64());
    println!("artifact_root: {}", args.temp_dir.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_materialize_articles_phase(args: &Args) -> Result<()> {
    let started = Instant::now();
    let mut settings = build_settings(args);
    let article_bucket_meta = load_article_buckets_artifact_meta(&args.temp_dir)?;
    apply_article_buckets_artifact_meta(&mut settings, &article_bucket_meta);
    announce_settings(args, &settings);
    prepare_phase_dir(&args.output_root.join("articles"), args.overwrite, "articles output")?;
    let article_schema = articles_schema();
    let t0 = Instant::now();
    let rows = materialize_articles(args, &settings, article_schema)?;
    println!("materialized articles in {:.2}s", t0.elapsed().as_secs_f64());
    println!("article_rows_rows: {}", rows);
    println!("artifact_root: {}", args.temp_dir.display());
    println!("output_root: {}", args.output_root.display());
    println!("done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn announce_settings(args: &Args, settings: &Settings) {
    println!(
        "settings: parser_workers={} join_bucket_workers={} article_bucket_workers={} join_buckets={} article_buckets={} offer_partition_buffer_mb_per_worker={} temp_compression={:?} temp_zstd_level={} memory_limit_gb={}",
        settings.parser_workers,
        settings.join_bucket_workers,
        settings.article_bucket_workers,
        settings.join_buckets,
        settings.article_buckets,
        settings.offer_partition_buffer_bytes / (1024 * 1024),
        settings.temp_compression,
        settings.temp_zstd_level,
        args.memory_limit_gb,
    );
}

fn prepare_run_roots(args: &Args) -> Result<()> {
    if args.output_root.exists() {
        if !args.overwrite {
            bail!("output root already exists: {} (pass --overwrite)", args.output_root.display());
        }
        fs::remove_dir_all(&args.output_root)?;
    }
    if args.temp_dir.exists() {
        if !args.overwrite {
            bail!("artifact root already exists: {} (pass --overwrite)", args.temp_dir.display());
        }
        fs::remove_dir_all(&args.temp_dir)?;
    }
    fs::create_dir_all(args.output_root.join("offer_rows"))?;
    fs::create_dir_all(args.output_root.join("articles"))?;
    fs::create_dir_all(&args.temp_dir)?;
    Ok(())
}

fn prepare_phase_dir(path: &Path, overwrite: bool, label: &str) -> Result<()> {
    if path.exists() {
        if !overwrite {
            bail!("{} already exists: {} (pass --overwrite)", label, path.display());
        }
        fs::remove_dir_all(path)?;
    }
    fs::create_dir_all(path)?;
    Ok(())
}

fn partition_artifact_meta_path(root: &Path) -> PathBuf {
    root.join("partition_artifact.json")
}

fn article_buckets_artifact_meta_path(root: &Path) -> PathBuf {
    root.join("article_buckets_artifact.json")
}

fn ensure_partition_artifact_meta(root: &Path, settings: &Settings) -> Result<()> {
    fs::create_dir_all(root)?;
    let path = partition_artifact_meta_path(root);
    let expected = PartitionArtifactMeta {
        join_buckets: settings.join_buckets,
        temp_compression: settings.temp_compression,
        temp_zstd_level: settings.temp_zstd_level,
    };
    if path.exists() {
        let existing: PartitionArtifactMeta = serde_json::from_slice(&fs::read(&path)?)?;
        if existing.join_buckets != expected.join_buckets
            || existing.temp_compression != expected.temp_compression
            || existing.temp_zstd_level != expected.temp_zstd_level
        {
            bail!(
                "partition artifact metadata mismatch at {}: existing join_buckets={} temp_compression={:?} temp_zstd_level={}, requested join_buckets={} temp_compression={:?} temp_zstd_level={}",
                path.display(),
                existing.join_buckets,
                existing.temp_compression,
                existing.temp_zstd_level,
                expected.join_buckets,
                expected.temp_compression,
                expected.temp_zstd_level,
            );
        }
        return Ok(());
    }
    fs::write(path, serde_json::to_vec_pretty(&expected)?)?;
    Ok(())
}

fn load_partition_artifact_meta(root: &Path) -> Result<PartitionArtifactMeta> {
    let path = partition_artifact_meta_path(root);
    let meta = serde_json::from_slice(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)?;
    Ok(meta)
}

fn apply_partition_artifact_meta(settings: &mut Settings, meta: &PartitionArtifactMeta) {
    settings.join_buckets = meta.join_buckets;
    settings.temp_compression = meta.temp_compression;
    settings.temp_zstd_level = meta.temp_zstd_level;
}

fn ensure_article_buckets_artifact_meta(root: &Path, settings: &Settings) -> Result<()> {
    fs::create_dir_all(root)?;
    let path = article_buckets_artifact_meta_path(root);
    let expected = ArticleBucketsArtifactMeta {
        article_buckets: settings.article_buckets,
        temp_compression: settings.temp_compression,
        temp_zstd_level: settings.temp_zstd_level,
    };
    if path.exists() {
        let existing: ArticleBucketsArtifactMeta = serde_json::from_slice(&fs::read(&path)?)?;
        if existing.article_buckets != expected.article_buckets
            || existing.temp_compression != expected.temp_compression
            || existing.temp_zstd_level != expected.temp_zstd_level
        {
            bail!(
                "article bucket artifact metadata mismatch at {}: existing article_buckets={} temp_compression={:?} temp_zstd_level={}, requested article_buckets={} temp_compression={:?} temp_zstd_level={}",
                path.display(),
                existing.article_buckets,
                existing.temp_compression,
                existing.temp_zstd_level,
                expected.article_buckets,
                expected.temp_compression,
                expected.temp_zstd_level,
            );
        }
        return Ok(());
    }
    fs::write(path, serde_json::to_vec_pretty(&expected)?)?;
    Ok(())
}

fn load_article_buckets_artifact_meta(root: &Path) -> Result<ArticleBucketsArtifactMeta> {
    let path = article_buckets_artifact_meta_path(root);
    let meta = serde_json::from_slice(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)?;
    Ok(meta)
}

fn apply_article_buckets_artifact_meta(settings: &mut Settings, meta: &ArticleBucketsArtifactMeta) {
    settings.article_buckets = meta.article_buckets;
    settings.temp_compression = meta.temp_compression;
    settings.temp_zstd_level = meta.temp_zstd_level;
}

fn build_settings(args: &Args) -> Settings {
    let parser_workers = if args.parser_workers > 0 { args.parser_workers } else { args.threads.max(1) };
    let join_bucket_workers = if args.join_bucket_workers > 0 {
        args.join_bucket_workers
    } else {
        args.threads.min((args.memory_limit_gb / 8).max(1)).max(1)
    };
    let article_bucket_workers = if args.article_bucket_workers > 0 {
        args.article_bucket_workers
    } else {
        args.threads.min((args.memory_limit_gb / 6).max(1)).max(1)
    };
    let join_buckets = args.join_buckets.unwrap_or_else(|| {
        if args.offers_only {
            256
        } else {
            derived_join_buckets(args.memory_limit_gb)
        }
    });
    let article_buckets = args.article_buckets.unwrap_or_else(|| derived_article_buckets(args.memory_limit_gb));
    let offer_partition_buffer_bytes = derived_offer_partition_buffer_bytes(args.memory_limit_gb, parser_workers);
    Settings {
        parser_workers,
        join_bucket_workers,
        article_bucket_workers,
        join_buckets,
        article_buckets,
        offer_partition_buffer_bytes,
        temp_compression: args.temp_compression,
        temp_zstd_level: args.temp_zstd_level,
    }
}

fn derived_join_buckets(memory_limit_gb: usize) -> usize {
    let mut buckets = 1024usize;
    if memory_limit_gb < 80 {
        buckets = buckets.saturating_mul((80 + memory_limit_gb - 1) / memory_limit_gb.max(1));
    } else if memory_limit_gb >= 160 {
        buckets /= 2;
    }
    buckets.next_power_of_two().max(256)
}

fn derived_article_buckets(memory_limit_gb: usize) -> usize {
    let mut buckets = 1024usize;
    if memory_limit_gb < 80 {
        buckets = buckets.saturating_mul((80 + memory_limit_gb - 1) / memory_limit_gb.max(1));
    }
    buckets.next_power_of_two().max(256)
}

fn derived_offer_partition_buffer_bytes(memory_limit_gb: usize, parser_workers: usize) -> usize {
    let total_gb = if memory_limit_gb >= 32 {
        (memory_limit_gb / 4).min(16)
    } else {
        (memory_limit_gb / 2).max(1)
    };
    let bytes = total_gb
        .saturating_mul(1024)
        .saturating_mul(1024)
        .saturating_mul(1024)
        / parser_workers.max(1);
    bytes.max(64 * 1024 * 1024)
}

fn write_input_manifests(output_root: &Path, files: &CollectionFiles) -> Result<()> {
    let dir = output_root.join("inputs");
    fs::create_dir_all(&dir)?;
    write_manifest(dir.join("offers.txt"), &files.offers)?;
    write_manifest(dir.join("pricings.txt"), &files.pricings)?;
    write_manifest(dir.join("markers.txt"), &files.markers)?;
    write_manifest(dir.join("cans.txt"), &files.cans)?;
    Ok(())
}

fn write_manifest(path: PathBuf, files: &[PathBuf]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    for file in files {
        writeln!(writer, "{}", file.display())?;
    }
    writer.flush()?;
    Ok(())
}

fn discover_offer_files(args: &Args) -> Result<Vec<PathBuf>> {
    let offers = expand_glob(args.source_root.join("offers").join(&args.offers_glob).to_string_lossy().as_ref(), args.offer_file_limit)?;
    if offers.is_empty() {
        bail!("offers collection resolved to zero files");
    }
    Ok(offers)
}

fn discover_pricing_files(args: &Args) -> Result<Vec<PathBuf>> {
    let pricings = expand_glob(args.source_root.join("pricings").join(&args.pricings_glob).to_string_lossy().as_ref(), args.pricing_file_limit)?;
    if pricings.is_empty() {
        bail!("pricings collection resolved to zero files");
    }
    Ok(pricings)
}

fn discover_marker_files(args: &Args) -> Result<Vec<PathBuf>> {
    let markers = expand_glob(args.source_root.join("coreArticleMarkers").join(&args.markers_glob).to_string_lossy().as_ref(), args.marker_file_limit)?;
    if markers.is_empty() {
        bail!("coreArticleMarkers collection resolved to zero files");
    }
    Ok(markers)
}

fn discover_can_files(args: &Args) -> Result<Vec<PathBuf>> {
    let cans = expand_glob(args.source_root.join("customerArticleNumbers").join(&args.cans_glob).to_string_lossy().as_ref(), args.can_file_limit)?;
    if cans.is_empty() {
        bail!("customerArticleNumbers collection resolved to zero files");
    }
    Ok(cans)
}

fn discover_files(args: &Args) -> Result<CollectionFiles> {
    let offers = discover_offer_files(args)?;
    if args.offers_only {
        return Ok(CollectionFiles {
            offers,
            pricings: Vec::new(),
            markers: Vec::new(),
            cans: Vec::new(),
        });
    }

    let pricings = discover_pricing_files(args)?;
    let markers = discover_marker_files(args)?;
    let cans = discover_can_files(args)?;
    Ok(CollectionFiles { offers, pricings, markers, cans })
}

fn expand_glob(pattern: &str, limit: Option<usize>) -> Result<Vec<PathBuf>> {
    let mut paths = glob(pattern)?.collect::<std::result::Result<Vec<_>, _>>()?;
    paths.sort();
    if let Some(limit) = limit {
        paths.truncate(limit);
    }
    Ok(paths)
}

fn partition_offers(args: &Args, settings: &Settings, files: &[PathBuf], s2_mappings: Arc<Vec<Option<HashMap<i32, i32>>>>) -> Result<usize> {
    let worker_dirs = prepare_worker_dirs(args.temp_dir.join("partition/offers"), settings.parser_workers)?;
    let total = files
        .par_chunks(chunk_size(files.len(), settings.parser_workers))
        .enumerate()
        .map(|(worker_idx, chunk)| -> Result<usize> {
            let mut writer = BucketWriterSet::with_buffer_limit(
                worker_dirs[worker_idx].clone(),
                args.max_open_temp_files,
                settings.offer_partition_buffer_bytes,
                settings.temp_compression,
                settings.temp_zstd_level,
            )?;
            let mut rows = 0usize;
            for path in chunk {
                rows += stream_gzip_lines(path, |line| {
                    let raw: RawOfferExport = serde_json::from_slice(line)?;
                    let projected = preproject_offer(raw, &s2_mappings)?;
                    let bucket = join_bucket(projected.vendor, &projected.article_number, settings.join_buckets);
                    writer.write(bucket, &projected)
                })?;
            }
            writer.finish()?;
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(total.into_iter().sum())
}

fn partition_pricings(args: &Args, settings: &Settings, files: &[PathBuf]) -> Result<usize> {
    let worker_dirs = prepare_worker_dirs(args.temp_dir.join("partition/pricings"), settings.parser_workers)?;
    let total = files
        .par_chunks(chunk_size(files.len(), settings.parser_workers))
        .enumerate()
        .map(|(worker_idx, chunk)| -> Result<usize> {
            let mut writer = BucketWriterSet::new(
                worker_dirs[worker_idx].clone(),
                args.max_open_temp_files,
                settings.temp_compression,
                settings.temp_zstd_level,
            )?;
            let mut rows = 0usize;
            for path in chunk {
                rows += stream_gzip_lines(path, |line| {
                    let raw: RawPricingExport = serde_json::from_slice(line)?;
                    let row = PricingTemp {
                        vendor: decode_binary_uuid(&raw.vendor_id)?,
                        article_number: raw.article_number,
                        pricing: convert_pricing_details(raw.pricing_details)?,
                    };
                    let bucket = join_bucket(row.vendor, &row.article_number, settings.join_buckets);
                    writer.write(bucket, &row)
                })?;
            }
            writer.finish()?;
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(total.into_iter().sum())
}

fn partition_markers(args: &Args, settings: &Settings, files: &[PathBuf]) -> Result<usize> {
    let worker_dirs = prepare_worker_dirs(args.temp_dir.join("partition/markers"), settings.parser_workers)?;
    let total = files
        .par_chunks(chunk_size(files.len(), settings.parser_workers))
        .enumerate()
        .map(|(worker_idx, chunk)| -> Result<usize> {
            let mut writer = BucketWriterSet::new(
                worker_dirs[worker_idx].clone(),
                args.max_open_temp_files,
                settings.temp_compression,
                settings.temp_zstd_level,
            )?;
            let mut rows = 0usize;
            for path in chunk {
                rows += stream_gzip_lines(path, |line| {
                    let raw: RawMarkerExport = serde_json::from_slice(line)?;
                    let row = MarkerTemp {
                        vendor: decode_binary_uuid(&raw.vendor_id)?,
                        article_number: raw.article_number,
                        source: decode_binary_uuid(&raw.source_id)?,
                        enabled: raw.marker,
                    };
                    let bucket = join_bucket(row.vendor, &row.article_number, settings.join_buckets);
                    writer.write(bucket, &row)
                })?;
            }
            writer.finish()?;
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(total.into_iter().sum())
}

fn partition_cans(args: &Args, settings: &Settings, files: &[PathBuf]) -> Result<usize> {
    let worker_dirs = prepare_worker_dirs(args.temp_dir.join("partition/cans"), settings.parser_workers)?;
    let total = files
        .par_chunks(chunk_size(files.len(), settings.parser_workers))
        .enumerate()
        .map(|(worker_idx, chunk)| -> Result<usize> {
            let mut writer = BucketWriterSet::new(
                worker_dirs[worker_idx].clone(),
                args.max_open_temp_files,
                settings.temp_compression,
                settings.temp_zstd_level,
            )?;
            let mut rows = 0usize;
            for path in chunk {
                rows += stream_gzip_lines(path, |line| {
                    let raw: RawCanExport = serde_json::from_slice(line)?;
                    let row = CanTemp {
                        vendor: decode_binary_uuid(&raw.vendor_id)?,
                        article_number: raw.article_number,
                        version: decode_binary_uuid(&raw.version_id)?,
                        value: raw.customer_article_number,
                    };
                    let bucket = join_bucket(row.vendor, &row.article_number, settings.join_buckets);
                    writer.write(bucket, &row)
                })?;
            }
            writer.finish()?;
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(total.into_iter().sum())
}

fn process_join_buckets(args: &Args, settings: &Settings, offer_schema: SchemaRef, _article_schema: SchemaRef) -> Result<(usize, usize)> {
    let join_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.join_bucket_workers)
        .build()?;
    let progress = Arc::new(JoinBucketProgress {
        started: Instant::now(),
        total_buckets: settings.join_buckets,
        completed_buckets: AtomicUsize::new(0),
        offer_rows: AtomicUsize::new(0),
        article_partial_rows: AtomicUsize::new(0),
    });
    let (progress_stop_tx, progress_stop_rx) = mpsc::channel();
    let progress_handle = spawn_join_bucket_progress(progress.clone(), progress_stop_rx);
    let totals = join_pool.install(|| {
        (0..settings.join_buckets)
            .into_par_iter()
            .map(|bucket| {
                let result = process_one_join_bucket(args, settings, bucket, offer_schema.clone());
                if let Ok((offer_rows, article_partial_rows)) = result.as_ref() {
                    progress.completed_buckets.fetch_add(1, Ordering::Relaxed);
                    progress.offer_rows.fetch_add(*offer_rows, Ordering::Relaxed);
                    progress.article_partial_rows.fetch_add(*article_partial_rows, Ordering::Relaxed);
                }
                result
            })
            .collect::<Result<Vec<_>>>()
    });
    let _ = progress_stop_tx.send(());
    let _ = progress_handle.join();
    let totals = totals?;
    Ok(totals.into_iter().fold((0usize, 0usize), |acc, x| (acc.0 + x.0, acc.1 + x.1)))
}

struct JoinBucketProgress {
    started: Instant,
    total_buckets: usize,
    completed_buckets: AtomicUsize,
    offer_rows: AtomicUsize,
    article_partial_rows: AtomicUsize,
}

fn spawn_join_bucket_progress(progress: Arc<JoinBucketProgress>, stop_rx: mpsc::Receiver<()>) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        match stop_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(_) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let completed = progress.completed_buckets.load(Ordering::Relaxed);
                let offer_rows = progress.offer_rows.load(Ordering::Relaxed);
                let article_partial_rows = progress.article_partial_rows.load(Ordering::Relaxed);
                let elapsed = progress.started.elapsed().as_secs_f64();
                let pct = if progress.total_buckets == 0 {
                    100.0
                } else {
                    (completed as f64 * 100.0) / progress.total_buckets as f64
                };
                let eta_seconds = if completed == 0 {
                    None
                } else {
                    let per_bucket = elapsed / completed as f64;
                    Some(per_bucket * progress.total_buckets.saturating_sub(completed) as f64)
                };
                println!(
                    "phase2 progress: buckets={}/{} ({:.1}%) offer_rows={} article_partial_rows={} elapsed={} eta={}",
                    completed,
                    progress.total_buckets,
                    pct,
                    offer_rows,
                    article_partial_rows,
                    format_progress_duration(elapsed),
                    eta_seconds.map(format_progress_duration).unwrap_or_else(|| "n/a".to_string()),
                );
            }
        }
    })
}

fn format_progress_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{}h{:02}m{:02}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m{:02}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

fn progress_pct_and_eta(total_units: usize, completed_units: usize, elapsed_seconds: f64) -> (f64, Option<f64>) {
    let pct = if total_units == 0 {
        100.0
    } else {
        (completed_units as f64 * 100.0) / total_units as f64
    };
    let eta_seconds = if completed_units == 0 {
        None
    } else {
        let per_unit = elapsed_seconds / completed_units as f64;
        Some(per_unit * total_units.saturating_sub(completed_units) as f64)
    };
    (pct, eta_seconds)
}

struct ArticlePartialRepartitionProgress {
    started: Instant,
    total_files: usize,
    completed_files: AtomicUsize,
    article_partial_rows: AtomicUsize,
}

fn spawn_article_partial_repartition_progress(
    progress: Arc<ArticlePartialRepartitionProgress>,
    stop_rx: mpsc::Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        match stop_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(_) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let completed = progress.completed_files.load(Ordering::Relaxed);
                let article_partial_rows = progress.article_partial_rows.load(Ordering::Relaxed);
                let elapsed = progress.started.elapsed().as_secs_f64();
                let (pct, eta_seconds) = progress_pct_and_eta(progress.total_files, completed, elapsed);
                println!(
                    "phase3 progress: files={}/{} ({:.1}%) article_partial_rows={} elapsed={} eta={}",
                    completed,
                    progress.total_files,
                    pct,
                    article_partial_rows,
                    format_progress_duration(elapsed),
                    eta_seconds.map(format_progress_duration).unwrap_or_else(|| "n/a".to_string()),
                );
            }
        }
    })
}

struct ArticleMaterializationProgress {
    started: Instant,
    total_buckets: usize,
    completed_buckets: AtomicUsize,
    article_rows: AtomicUsize,
}

fn spawn_article_materialization_progress(
    progress: Arc<ArticleMaterializationProgress>,
    stop_rx: mpsc::Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        match stop_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(_) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let completed = progress.completed_buckets.load(Ordering::Relaxed);
                let article_rows = progress.article_rows.load(Ordering::Relaxed);
                let elapsed = progress.started.elapsed().as_secs_f64();
                let (pct, eta_seconds) = progress_pct_and_eta(progress.total_buckets, completed, elapsed);
                println!(
                    "phase4 progress: buckets={}/{} ({:.1}%) article_rows={} elapsed={} eta={}",
                    completed,
                    progress.total_buckets,
                    pct,
                    article_rows,
                    format_progress_duration(elapsed),
                    eta_seconds.map(format_progress_duration).unwrap_or_else(|| "n/a".to_string()),
                );
            }
        }
    })
}

fn process_one_join_bucket(args: &Args, settings: &Settings, bucket: usize, offer_schema: SchemaRef) -> Result<(usize, usize)> {
    let pricings = load_join_bucket::<PricingTemp>(
        &args.temp_dir.join("partition/pricings"),
        settings.parser_workers,
        bucket,
        settings.temp_compression,
    )?;
    let markers = load_join_bucket::<MarkerTemp>(
        &args.temp_dir.join("partition/markers"),
        settings.parser_workers,
        bucket,
        settings.temp_compression,
    )?;
    let cans = load_join_bucket::<CanTemp>(
        &args.temp_dir.join("partition/cans"),
        settings.parser_workers,
        bucket,
        settings.temp_compression,
    )?;

    let mut pricing_map: HashMap<JoinKey, Vec<PricingTemp>> = HashMap::new();
    for row in pricings {
        pricing_map.entry(JoinKey::from_parts(row.vendor, &row.article_number)).or_default().push(row);
    }
    let mut marker_map: HashMap<JoinKey, Vec<MarkerTemp>> = HashMap::new();
    for row in markers {
        marker_map.entry(JoinKey::from_parts(row.vendor, &row.article_number)).or_default().push(row);
    }
    let mut can_map: HashMap<JoinKey, Vec<CanTemp>> = HashMap::new();
    for row in cans {
        can_map.entry(JoinKey::from_parts(row.vendor, &row.article_number)).or_default().push(row);
    }

    let offer_output = args.output_root.join("offer_rows").join(format!("part-{bucket:05}.parquet"));
    let mut offer_sink = OfferRowsSink::new(offer_output, offer_schema, args.batch_rows, args.row_group_rows, args.compression)?;
    let article_partial_path = args.temp_dir.join("article_partials");
    fs::create_dir_all(&article_partial_path)?;
    let mut partial_writer = BufWriter::new(File::create(article_partial_path.join(format!("part-{bucket:05}.bin")))?);
    let offers: Vec<OfferProjectedTemp> = load_join_bucket(
        &args.temp_dir.join("partition/offers"),
        settings.parser_workers,
        bucket,
        settings.temp_compression,
    )?;

    let mut offer_rows = 0usize;
    let mut article_partial_rows = 0usize;
    let mut partials: HashMap<String, ArticlePartialAcc> = HashMap::new();

    for offer in offers {
        let key = JoinKey::from_parts(offer.vendor, &offer.article_number);
        let joined_pricings = pricing_map.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
        let joined_markers = marker_map.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
        let joined_cans = can_map.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);

        let mut prices = Vec::with_capacity(joined_pricings.len() + 2);
        if let Some(p) = offer.inline_pricing_open.as_ref().and_then(project_one_pricing) {
            prices.push(p);
        }
        if let Some(p) = offer.inline_pricing_closed.as_ref().and_then(project_one_pricing) {
            prices.push(p);
        }
        for p in joined_pricings {
            if let Some(pp) = project_one_pricing(&p.pricing) {
                prices.push(pp);
            }
        }

        let (enabled_sources, disabled_sources) = project_markers(joined_markers);
        let customer_article_numbers = project_customer_numbers(joined_cans, offer.inline_can_pair.as_ref());
        let envelope = offer_envelope(&prices);

        offer_sink.append(OfferRowRef {
            id: &offer.id,
            article_hash: &offer.article_hash,
            ean: &offer.ean,
            article_number: &offer.article_number,
            vendor_id: &offer.vendor_id,
            catalog_version_id: &offer.catalog_version_id,
            prices: &prices,
            delivery_time_days_max: offer.delivery_time_days_max,
            core_marker_enabled_sources: &enabled_sources,
            core_marker_disabled_sources: &disabled_sources,
            features: &offer.features,
            relationship_accessory_for: &offer.relationship_accessory_for,
            relationship_spare_part_for: &offer.relationship_spare_part_for,
            relationship_similar_to: &offer.relationship_similar_to,
            price_list_ids: &envelope.price_list_ids,
            currencies: &envelope.currencies,
            mins: &envelope.mins,
            maxs: &envelope.maxs,
        })?;
        offer_rows += 1;

        let entry = partials.entry(offer.article_hash.clone()).or_insert_with(|| ArticlePartialAcc::new(&offer));
        entry.merge_offer(
            &offer.ean,
            &offer.article_number,
            &customer_article_numbers,
            &envelope.mins,
            &envelope.maxs,
        );
    }

    offer_sink.finish()?;

    for partial in partials.into_values() {
        serialize_into(&mut partial_writer, &partial.to_row())?;
        article_partial_rows += 1;
    }
    partial_writer.flush()?;

    Ok((offer_rows, article_partial_rows))
}

fn repartition_article_partials(args: &Args, settings: &Settings) -> Result<()> {
    let partial_root = args.temp_dir.join("article_partials");
    let mut inputs = glob(partial_root.join("part-*.bin").to_string_lossy().as_ref())?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    inputs.sort();
    let worker_dirs = prepare_worker_dirs(args.temp_dir.join("article_buckets"), settings.article_bucket_workers)?;
    let progress = Arc::new(ArticlePartialRepartitionProgress {
        started: Instant::now(),
        total_files: inputs.len(),
        completed_files: AtomicUsize::new(0),
        article_partial_rows: AtomicUsize::new(0),
    });
    let (progress_stop_tx, progress_stop_rx) = mpsc::channel();
    let progress_handle = spawn_article_partial_repartition_progress(progress.clone(), progress_stop_rx);
    let result = inputs
        .par_chunks(chunk_size(inputs.len(), settings.article_bucket_workers))
        .enumerate()
        .map(|(worker_idx, chunk)| -> Result<()> {
            let progress = progress.clone();
            let mut writer = BucketWriterSet::new(
                worker_dirs[worker_idx].clone(),
                args.max_open_temp_files,
                settings.temp_compression,
                settings.temp_zstd_level,
            )?;
            for path in chunk {
                let reader = BufReader::new(File::open(path)?);
                let mut reader = reader;
                let mut file_rows = 0usize;
                loop {
                    match deserialize_from::<_, ArticlePartialRow>(&mut reader) {
                        Ok(row) => {
                            let bucket = article_bucket(&row.article_hash, settings.article_buckets);
                            writer.write(bucket, &row)?;
                            file_rows += 1;
                        }
                        Err(err) => {
                            if matches!(*err, bincode::ErrorKind::Io(ref io) if io.kind() == std::io::ErrorKind::UnexpectedEof) {
                                break;
                            }
                            return Err(err.into());
                        }
                    }
                }
                progress.completed_files.fetch_add(1, Ordering::Relaxed);
                progress.article_partial_rows.fetch_add(file_rows, Ordering::Relaxed);
            }
            writer.finish()?;
            Ok(())
        })
        .collect::<Result<Vec<_>>>();
    let _ = progress_stop_tx.send(());
    let _ = progress_handle.join();
    result?;
    Ok(())
}

fn materialize_articles(args: &Args, settings: &Settings, article_schema: SchemaRef) -> Result<usize> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.article_bucket_workers)
        .build()?;
    let progress = Arc::new(ArticleMaterializationProgress {
        started: Instant::now(),
        total_buckets: settings.article_buckets,
        completed_buckets: AtomicUsize::new(0),
        article_rows: AtomicUsize::new(0),
    });
    let (progress_stop_tx, progress_stop_rx) = mpsc::channel();
    let progress_handle = spawn_article_materialization_progress(progress.clone(), progress_stop_rx);
    let rows = pool.install(|| {
        (0..settings.article_buckets)
            .into_par_iter()
            .map(|bucket| {
                let result = materialize_one_article_bucket(args, settings, bucket, article_schema.clone());
                if let Ok(rows) = result.as_ref() {
                    progress.completed_buckets.fetch_add(1, Ordering::Relaxed);
                    progress.article_rows.fetch_add(*rows, Ordering::Relaxed);
                }
                result
            })
            .collect::<Result<Vec<_>>>()
    });
    let _ = progress_stop_tx.send(());
    let _ = progress_handle.join();
    let rows = rows?;
    Ok(rows.into_iter().sum())
}

fn materialize_one_article_bucket(args: &Args, settings: &Settings, bucket: usize, article_schema: SchemaRef) -> Result<usize> {
    let rows: Vec<ArticlePartialRow> = load_join_bucket(
        &args.temp_dir.join("article_buckets"),
        settings.article_bucket_workers,
        bucket,
        settings.temp_compression,
    )?;
    if rows.is_empty() {
        return Ok(0);
    }

    let mut accs: HashMap<String, ArticlePartialAcc> = HashMap::new();
    for row in rows {
        let entry = accs.entry(row.article_hash.clone()).or_insert_with(|| ArticlePartialAcc::from_row(&row));
        if entry.article_hash != row.article_hash {
            bail!("article hash collision state mismatch");
        }
        entry.merge_partial(&row);
    }

    let out = args.output_root.join("articles").join(format!("part-{bucket:05}.parquet"));
    let mut sink = ArticlesSink::new(out, article_schema, args.batch_rows, args.row_group_rows, args.compression)?;
    let mut count = 0usize;
    for acc in accs.into_values() {
        let row = acc.to_final_article();
        sink.append(&row)?;
        count += 1;
    }
    sink.finish()?;
    Ok(count)
}

fn prepare_worker_dirs(root: PathBuf, workers: usize) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::with_capacity(workers);
    for idx in 0..workers {
        let dir = root.join(format!("w{idx:02}"));
        fs::create_dir_all(&dir)?;
        dirs.push(dir);
    }
    Ok(dirs)
}

fn chunk_size(total: usize, workers: usize) -> usize {
    ((total + workers.saturating_sub(1)) / workers.max(1)).max(1)
}

fn stream_gzip_lines<F>(path: &Path, mut f: F) -> Result<usize>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut reader = BufReader::with_capacity(1024 * 1024, decoder);
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut rows = 0usize;
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        trim_ascii_in_place(&mut buf);
        if buf.is_empty() {
            continue;
        }
        f(&buf)?;
        rows += 1;
    }
    Ok(rows)
}

fn trim_ascii_in_place(buf: &mut Vec<u8>) {
    let mut start = 0usize;
    let mut end = buf.len();
    while start < end && buf[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && buf[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    if start > 0 || end < buf.len() {
        let slice = buf[start..end].to_vec();
        *buf = slice;
    }
}

fn decode_binary_uuid(binary: &MongoBinary) -> Result<[u8; 16]> {
    let bytes = BASE64_STANDARD.decode(binary.value.base64.as_bytes())?;
    if bytes.len() != 16 {
        bail!("expected 16 UUID bytes, got {}", bytes.len());
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn uuid_str(bytes: [u8; 16]) -> String {
    Uuid::from_bytes(bytes).to_string()
}

fn join_bucket(vendor: [u8; 16], article_number: &str, buckets: usize) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    vendor.hash(&mut hasher);
    article_number.hash(&mut hasher);
    (hasher.finish() as usize) & (buckets - 1)
}

fn article_bucket(article_hash: &str, buckets: usize) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    article_hash.hash(&mut hasher);
    (hasher.finish() as usize) & (buckets - 1)
}

fn preproject_offer(raw: RawOfferExport, s2_mappings: &[Option<HashMap<i32, i32>>]) -> Result<OfferProjectedTemp> {
    let vendor = decode_binary_uuid(&raw.vendor_id)?;
    let catalog_version = decode_binary_uuid(&raw.catalog_version_id)?;
    let vendor_id = uuid_str(vendor);
    let catalog_version_id = uuid_str(catalog_version);
    let params = raw.offer.offer_params;
    let article_number = raw.article_number;
    let article_number_b64 = URL_SAFE_NO_PAD.encode(article_number.as_bytes());
    let id = format!("{}:{}:{}", vendor_id, article_number_b64, catalog_version_id);

    let name = params.name.unwrap_or_default();
    let manufacturer_name = params.manufacturer_name.unwrap_or_default();
    let description = params.description.unwrap_or_default();
    let ean = params.ean.unwrap_or_default();
    let manufacturer_article_number = params.manufacturer_article_number.unwrap_or_default();
    let manufacturer_article_type = params.manufacturer_article_type.unwrap_or_default();
    let category_l = project_categories(&params.category_paths);
    let features = project_features(&params.features);
    let article_hash = compute_article_hash(
        &name,
        &manufacturer_name,
        &description,
        &params.category_paths,
        &ean,
        &article_number,
        &manufacturer_article_number,
        &manufacturer_article_type,
    );
    let inline_can_pair = if !params.customer_article_number.clone().unwrap_or_default().is_empty() {
        Some(TempCanPair {
            value: params.customer_article_number.unwrap_or_default(),
            version_id: catalog_version_id.clone(),
        })
    } else {
        None
    };

    Ok(OfferProjectedTemp {
        vendor,
        article_number,
        catalog_version,
        article_hash,
        id,
        vendor_id,
        catalog_version_id,
        name,
        manufacturer_name,
        ean,
        delivery_time_days_max: params.delivery_time.map(|v| i32::try_from(v.0).unwrap_or(0)).unwrap_or(0),
        eclass5_code: project_eclass(&params.eclass_groups, 5),
        eclass7_code: project_eclass(&params.eclass_groups, 7),
        s2class_code: derive_s2class_hierarchy(Some(&params.eclass_groups), s2_mappings),
        relationship_accessory_for: take_vec(raw.offer.related_article_numbers.accessory_for, RELATIONSHIP_LIMIT),
        relationship_spare_part_for: take_vec(raw.offer.related_article_numbers.spare_part_for, RELATIONSHIP_LIMIT),
        relationship_similar_to: take_vec(raw.offer.related_article_numbers.similar_to, RELATIONSHIP_LIMIT),
        category_l1: category_l[0].clone(),
        category_l2: category_l[1].clone(),
        category_l3: category_l[2].clone(),
        category_l4: category_l[3].clone(),
        category_l5: category_l[4].clone(),
        features,
        inline_pricing_open: raw.offer.pricings.open.map(convert_pricing_details).transpose()?,
        inline_pricing_closed: raw.offer.pricings.closed.map(convert_pricing_details).transpose()?,
        inline_can_pair,
    })
}

fn take_vec(mut values: Vec<String>, limit: usize) -> Vec<String> {
    if values.len() > limit {
        values.truncate(limit);
    }
    values
}

fn convert_pricing_details(raw: RawPricingDetails) -> Result<TempPricingDetails> {
    Ok(TempPricingDetails {
        source_price_list_id: raw.source_price_list_id.as_ref().map(decode_binary_uuid).transpose()?,
        type_name: raw.type_name,
        currency_code: raw.prices.as_ref().and_then(|p| p.currency_code.clone()),
        staggered_prices: raw
            .prices
            .map(|p| {
                p.staggered_prices
                    .into_iter()
                    .map(|s| TempStaggeredPrice {
                        min_quantity: s.min_quantity,
                        price: s.price,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        price_quantity: raw.price_quantity,
    })
}

fn project_one_pricing(pd: &TempPricingDetails) -> Option<ProjectedPrice> {
    let price = single_unit_price(pd)?;
    let priority = match pd.type_name.as_deref().unwrap_or("OPEN") {
        "OPEN" => 1,
        "CLOSED" => 2,
        "GROUP" => 3,
        "DEDICATED" => 4,
        _ => 1,
    };
    Some(ProjectedPrice {
        price,
        currency: pd.currency_code.clone().unwrap_or_default(),
        priority,
        source_price_list_id: pd.source_price_list_id.map(uuid_str).unwrap_or_default(),
    })
}

fn single_unit_price(pd: &TempPricingDetails) -> Option<f64> {
    let mut best: Option<(&TempStaggeredPrice, Decimal)> = None;
    for sp in &pd.staggered_prices {
        let price = parse_decimal_opt(sp.price.as_deref())?;
        let min_q = parse_decimal_default(sp.min_quantity.as_deref(), Decimal::ZERO);
        match &best {
            Some((_, cur)) if min_q >= *cur => {}
            _ => best = Some((sp, min_q)),
        }
        let _ = price;
    }
    let sp = best?.0;
    let base_price = parse_decimal_opt(sp.price.as_deref())?;
    let qty = parse_decimal_default(pd.price_quantity.as_deref(), Decimal::ONE);
    let qty = if qty.is_zero() { Decimal::ONE } else { qty };
    (base_price / qty).to_f64()
}

fn parse_decimal_opt(value: Option<&str>) -> Option<Decimal> {
    let value = value?;
    if value.is_empty() {
        return None;
    }
    Decimal::from_str_exact(value).ok()
}

fn parse_decimal_default(value: Option<&str>, default: Decimal) -> Decimal {
    parse_decimal_opt(value).unwrap_or(default)
}

fn project_features(features: &[RawFeature]) -> Vec<String> {
    let mut out = Vec::new();
    for feature in features {
        let name = feature.name.clone().unwrap_or_default();
        for value in &feature.values {
            if value.contains('=') {
                continue;
            }
            out.push(format!("{}={}", name, value));
        }
    }
    out
}

fn project_categories(paths: &[RawCategoryPath]) -> [Vec<String>; 5] {
    let mut bins: [Vec<String>; 5] = Default::default();
    for path in paths {
        for depth in 1..=path.elements.len().min(5) {
            let encoded = encode_category_path(&path.elements[..depth]);
            if !bins[depth - 1].contains(&encoded) {
                bins[depth - 1].push(encoded);
            }
        }
    }
    bins
}

fn encode_category_path(elements: &[String]) -> String {
    let mut out = String::new();
    for (idx, elem) in elements.iter().enumerate() {
        if idx > 0 {
            out.push(PATH_SEPARATOR);
        }
        for ch in elem.chars() {
            if ch == PATH_SEPARATOR {
                out.push(PATH_ESCAPE);
            } else {
                out.push(ch);
            }
        }
    }
    out
}

fn compute_article_hash(
    name: &str,
    manufacturer_name: &str,
    description: &str,
    category_paths: &[RawCategoryPath],
    ean: &str,
    article_number: &str,
    manufacturer_article_number: &str,
    manufacturer_article_type: &str,
) -> String {
    let mut paths = category_paths
        .iter()
        .filter(|p| !p.elements.is_empty())
        .map(|p| p.elements.join("¦"))
        .collect::<Vec<_>>();
    paths.sort();
    let cats = paths.join("\x1e");

    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(manufacturer_name.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(description.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(cats.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(ean.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(article_number.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(manufacturer_article_number.as_bytes());
    hasher.update([HASH_FIELD_SEP]);
    hasher.update(manufacturer_article_type.as_bytes());
    let digest = hasher.finalize();
    hex16(&digest[..16])
}

fn hex16(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn project_eclass(groups: &EclassGroups, version: usize) -> Vec<i32> {
    let values = match version {
        5 => &groups.eclass_5_1,
        7 => &groups.eclass_7_1,
        _ => return Vec::new(),
    };
    let mut out = Vec::with_capacity(values.len().saturating_mul(4));
    for code in values {
        expand_eclass_hierarchy(code.0, &mut out);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn expand_eclass_hierarchy(code: i32, out: &mut Vec<i32>) {
    out.push((code / 1_000_000) * 1_000_000);
    out.push((code / 10_000) * 10_000);
    out.push((code / 100) * 100);
    out.push(code);
}

fn derive_s2class_hierarchy(groups: Option<&EclassGroups>, mappings: &[Option<HashMap<i32, i32>>]) -> Vec<i32> {
    let Some(groups) = groups else {
        return default_s2class_hierarchy();
    };
    let Some((version, codes)) = groups.highest_non_s2() else {
        return default_s2class_hierarchy();
    };
    let Some(map) = mappings[version].as_ref() else {
        return default_s2class_hierarchy();
    };
    let mut leaves = Vec::with_capacity(codes.len());
    for code in codes {
        if let Some(mapped) = map.get(&code.0) {
            leaves.push(*mapped);
        }
    }
    if leaves.is_empty() {
        return default_s2class_hierarchy();
    }
    leaves.sort_unstable();
    leaves.dedup();
    let mut hierarchy = Vec::with_capacity(leaves.len() * 4);
    for leaf in leaves {
        expand_eclass_hierarchy(leaf, &mut hierarchy);
    }
    hierarchy.sort_unstable();
    hierarchy.dedup();
    hierarchy
}

fn default_s2class_hierarchy() -> Vec<i32> {
    let mut out = Vec::with_capacity(4);
    expand_eclass_hierarchy(DEFAULT_S2CLASS_CODE, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn load_s2_mappings() -> Result<Vec<Option<HashMap<i32, i32>>>> {
    let mut mappings = (0..17).map(|_| None).collect::<Vec<_>>();
    let mapping_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../indexer/classification_mapping");
    for version in 5..=16 {
        let path = mapping_dir.join(format!("{version}-s2.bin.gz"));
        if !path.exists() {
            continue;
        }
        let file = File::open(&path)?;
        let mut decoder = GzDecoder::new(BufReader::new(file));
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        if bytes.len() < 16 {
            bail!("mapping file too small: {}", path.display());
        }
        let magic = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
        let format_version = i32::from_be_bytes(bytes[4..8].try_into().unwrap());
        let count = i32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if magic != 0x4D415050 {
            bail!("bad magic in {}", path.display());
        }
        if format_version != 1 {
            bail!("unsupported mapping version {} in {}", format_version, path.display());
        }
        let mut map = HashMap::with_capacity(count);
        for i in 0..count {
            let off = 16 + i * 8;
            let from_code = i32::from_be_bytes(bytes[off..off + 4].try_into().unwrap());
            let to_code = i32::from_be_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            map.insert(from_code, to_code);
        }
        mappings[version] = Some(map);
    }
    Ok(mappings)
}

fn project_markers(markers: &[MarkerTemp]) -> (Vec<String>, Vec<String>) {
    let mut enabled = Vec::new();
    let mut disabled = Vec::new();
    let mut seen_enabled = BTreeSet::new();
    let mut seen_disabled = BTreeSet::new();
    for marker in markers {
        let src = uuid_str(marker.source);
        if marker.enabled {
            if seen_enabled.insert(src.clone()) {
                enabled.push(src);
            }
        } else if seen_disabled.insert(src.clone()) {
            disabled.push(src);
        }
    }
    if enabled.len() > MARKER_LIMIT {
        enabled.truncate(MARKER_LIMIT);
    }
    if disabled.len() > MARKER_LIMIT {
        disabled.truncate(MARKER_LIMIT);
    }
    (enabled, disabled)
}

fn project_customer_numbers(joined: &[CanTemp], inline: Option<&TempCanPair>) -> Vec<CustomerArticleNumberEntry> {
    let mut by_value: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for can in joined {
        if can.value.is_empty() {
            continue;
        }
        by_value.entry(can.value.clone()).or_default().insert(uuid_str(can.version));
    }
    if let Some(inline) = inline {
        if !inline.value.is_empty() {
            by_value.entry(inline.value.clone()).or_default().insert(inline.version_id.clone());
        }
    }
    by_value
        .into_iter()
        .map(|(value, version_ids)| CustomerArticleNumberEntry {
            value,
            version_ids: version_ids.into_iter().collect(),
        })
        .collect()
}

fn offer_envelope(prices: &[ProjectedPrice]) -> PriceEnvelope {
    let mut mins = [MAX_PRICE_SENTINEL; 7];
    let mut maxs = [-MAX_PRICE_SENTINEL; 7];
    let mut price_list_ids = BTreeSet::new();
    let mut currencies = BTreeSet::new();
    for price in prices {
        if !price.source_price_list_id.is_empty() {
            price_list_ids.insert(price.source_price_list_id.clone());
        }
        if !price.currency.is_empty() {
            currencies.insert(price.currency.to_lowercase());
        }
        let ccy = price.currency.to_lowercase();
        if let Some(idx) = CATALOG_CURRENCIES.iter().position(|&c| c == ccy) {
            mins[idx] = mins[idx].min(price.price);
            maxs[idx] = maxs[idx].max(price.price);
        }
    }
    PriceEnvelope {
        mins,
        maxs,
        price_list_ids: price_list_ids.into_iter().collect(),
        currencies: currencies.into_iter().collect(),
    }
}

impl ArticlePartialAcc {
    fn new(offer: &OfferProjectedTemp) -> Self {
        Self {
            article_hash: offer.article_hash.clone(),
            name: offer.name.clone(),
            manufacturer_name: offer.manufacturer_name.clone(),
            category_l1: offer.category_l1.clone(),
            category_l2: offer.category_l2.clone(),
            category_l3: offer.category_l3.clone(),
            category_l4: offer.category_l4.clone(),
            category_l5: offer.category_l5.clone(),
            eclass5_code: offer.eclass5_code.clone(),
            eclass7_code: offer.eclass7_code.clone(),
            s2class_code: offer.s2class_code.clone(),
            eans: BTreeSet::new(),
            article_numbers: BTreeSet::new(),
            customer_article_numbers: BTreeMap::new(),
            mins: [MAX_PRICE_SENTINEL; 7],
            maxs: [-MAX_PRICE_SENTINEL; 7],
        }
    }

    fn from_row(row: &ArticlePartialRow) -> Self {
        let mut acc = Self {
            article_hash: row.article_hash.clone(),
            name: row.name.clone(),
            manufacturer_name: row.manufacturer_name.clone(),
            category_l1: row.category_l1.clone(),
            category_l2: row.category_l2.clone(),
            category_l3: row.category_l3.clone(),
            category_l4: row.category_l4.clone(),
            category_l5: row.category_l5.clone(),
            eclass5_code: row.eclass5_code.clone(),
            eclass7_code: row.eclass7_code.clone(),
            s2class_code: row.s2class_code.clone(),
            eans: row.eans.iter().cloned().collect(),
            article_numbers: row.article_numbers.iter().cloned().collect(),
            customer_article_numbers: BTreeMap::new(),
            mins: [
                row.eur_price_min,
                row.chf_price_min,
                row.huf_price_min,
                row.pln_price_min,
                row.gbp_price_min,
                row.czk_price_min,
                row.cny_price_min,
            ],
            maxs: [
                row.eur_price_max,
                row.chf_price_max,
                row.huf_price_max,
                row.pln_price_max,
                row.gbp_price_max,
                row.czk_price_max,
                row.cny_price_max,
            ],
        };
        for entry in &row.customer_article_numbers {
            acc.customer_article_numbers
                .entry(entry.value.clone())
                .or_default()
                .extend(entry.version_ids.iter().cloned());
        }
        acc
    }

    fn merge_offer(
        &mut self,
        ean: &str,
        article_number: &str,
        customer_article_numbers: &[CustomerArticleNumberEntry],
        mins: &[f64; 7],
        maxs: &[f64; 7],
    ) {
        if !ean.is_empty() {
            self.eans.insert(ean.to_string());
        }
        if !article_number.is_empty() {
            self.article_numbers.insert(article_number.to_string());
        }
        for entry in customer_article_numbers {
            self.customer_article_numbers
                .entry(entry.value.clone())
                .or_default()
                .extend(entry.version_ids.iter().cloned());
        }
        for idx in 0..7 {
            self.mins[idx] = self.mins[idx].min(mins[idx]);
            self.maxs[idx] = self.maxs[idx].max(maxs[idx]);
        }
    }

    fn merge_partial(&mut self, row: &ArticlePartialRow) {
        self.eans.extend(row.eans.iter().cloned());
        self.article_numbers.extend(row.article_numbers.iter().cloned());
        for entry in &row.customer_article_numbers {
            self.customer_article_numbers
                .entry(entry.value.clone())
                .or_default()
                .extend(entry.version_ids.iter().cloned());
        }
        let mins = [
            row.eur_price_min,
            row.chf_price_min,
            row.huf_price_min,
            row.pln_price_min,
            row.gbp_price_min,
            row.czk_price_min,
            row.cny_price_min,
        ];
        let maxs = [
            row.eur_price_max,
            row.chf_price_max,
            row.huf_price_max,
            row.pln_price_max,
            row.gbp_price_max,
            row.czk_price_max,
            row.cny_price_max,
        ];
        for idx in 0..7 {
            self.mins[idx] = self.mins[idx].min(mins[idx]);
            self.maxs[idx] = self.maxs[idx].max(maxs[idx]);
        }
    }

    fn to_row(&self) -> ArticlePartialRow {
        ArticlePartialRow {
            article_hash: self.article_hash.clone(),
            name: self.name.clone(),
            manufacturer_name: self.manufacturer_name.clone(),
            category_l1: self.category_l1.clone(),
            category_l2: self.category_l2.clone(),
            category_l3: self.category_l3.clone(),
            category_l4: self.category_l4.clone(),
            category_l5: self.category_l5.clone(),
            eclass5_code: self.eclass5_code.clone(),
            eclass7_code: self.eclass7_code.clone(),
            s2class_code: self.s2class_code.clone(),
            eans: self.eans.iter().cloned().collect(),
            article_numbers: self.article_numbers.iter().cloned().collect(),
            customer_article_numbers: self
                .customer_article_numbers
                .iter()
                .map(|(value, version_ids)| CustomerArticleNumberEntry {
                    value: value.clone(),
                    version_ids: version_ids.iter().cloned().collect(),
                })
                .collect(),
            eur_price_min: self.mins[0],
            eur_price_max: self.maxs[0],
            chf_price_min: self.mins[1],
            chf_price_max: self.maxs[1],
            huf_price_min: self.mins[2],
            huf_price_max: self.maxs[2],
            pln_price_min: self.mins[3],
            pln_price_max: self.maxs[3],
            gbp_price_min: self.mins[4],
            gbp_price_max: self.maxs[4],
            czk_price_min: self.mins[5],
            czk_price_max: self.maxs[5],
            cny_price_min: self.mins[6],
            cny_price_max: self.maxs[6],
        }
    }

    fn to_final_article(&self) -> FinalArticleRow {
        let mut pieces = Vec::new();
        if !self.name.is_empty() {
            pieces.push(self.name.clone());
        }
        if !self.manufacturer_name.is_empty() {
            pieces.push(self.manufacturer_name.clone());
        }
        pieces.extend(self.eans.iter().cloned());
        pieces.extend(self.article_numbers.iter().cloned());
        let mut text_codes = pieces.join(" ");
        if text_codes.chars().count() > TEXT_CODES_LIMIT {
            text_codes = text_codes.chars().take(TEXT_CODES_LIMIT).collect();
        }
        FinalArticleRow {
            article_hash: self.article_hash.clone(),
            name: self.name.clone(),
            manufacturer_name: self.manufacturer_name.clone(),
            category_l1: self.category_l1.clone(),
            category_l2: self.category_l2.clone(),
            category_l3: self.category_l3.clone(),
            category_l4: self.category_l4.clone(),
            category_l5: self.category_l5.clone(),
            eclass5_code: self.eclass5_code.clone(),
            eclass7_code: self.eclass7_code.clone(),
            s2class_code: self.s2class_code.clone(),
            text_codes,
            customer_article_numbers: self
                .customer_article_numbers
                .iter()
                .map(|(value, version_ids)| CustomerArticleNumberEntry {
                    value: value.clone(),
                    version_ids: version_ids.iter().cloned().collect(),
                })
                .collect(),
            mins: self.mins,
            maxs: self.maxs,
        }
    }
}

#[derive(Debug, Clone)]
struct FinalArticleRow {
    article_hash: String,
    name: String,
    manufacturer_name: String,
    category_l1: Vec<String>,
    category_l2: Vec<String>,
    category_l3: Vec<String>,
    category_l4: Vec<String>,
    category_l5: Vec<String>,
    eclass5_code: Vec<i32>,
    eclass7_code: Vec<i32>,
    s2class_code: Vec<i32>,
    text_codes: String,
    customer_article_numbers: Vec<CustomerArticleNumberEntry>,
    mins: [f64; 7],
    maxs: [f64; 7],
}

struct BucketWriterSet {
    root: PathBuf,
    max_open: usize,
    open: HashMap<usize, BufWriter<File>>,
    order: VecDeque<usize>,
    buffers: HashMap<usize, Vec<u8>>,
    buffered_bytes: usize,
    max_buffered_bytes: usize,
    temp_compression: TempCompression,
    temp_zstd_level: i32,
}

impl BucketWriterSet {
    fn new(root: PathBuf, max_open: usize, temp_compression: TempCompression, temp_zstd_level: i32) -> Result<Self> {
        Self::with_buffer_limit(root, max_open, 8 * 1024 * 1024, temp_compression, temp_zstd_level)
    }

    fn with_buffer_limit(
        root: PathBuf,
        max_open: usize,
        max_buffered_bytes: usize,
        temp_compression: TempCompression,
        temp_zstd_level: i32,
    ) -> Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            max_open: max_open.max(1),
            open: HashMap::new(),
            order: VecDeque::new(),
            buffers: HashMap::new(),
            buffered_bytes: 0,
            max_buffered_bytes: max_buffered_bytes.max(1024 * 1024),
            temp_compression,
            temp_zstd_level,
        })
    }

    fn write<T: Serialize>(&mut self, bucket: usize, value: &T) -> Result<()> {
        let buffer = self.buffers.entry(bucket).or_default();
        let before = buffer.len();
        serialize_into(&mut *buffer, value)?;
        self.buffered_bytes += buffer.len().saturating_sub(before);
        if self.buffered_bytes >= self.max_buffered_bytes {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<()> {
        let buckets = self.buffers.keys().copied().collect::<Vec<_>>();
        for bucket in buckets {
            let data = match self.buffers.remove(&bucket) {
                Some(data) if !data.is_empty() => data,
                _ => continue,
            };
            self.buffered_bytes = self.buffered_bytes.saturating_sub(data.len());
            let temp_compression = self.temp_compression;
            let temp_zstd_level = self.temp_zstd_level;
            let writer = self.writer_for(bucket)?;
            match temp_compression {
                TempCompression::Uncompressed => writer.write_all(&data)?,
                TempCompression::Zstd => {
                    let compressed = zstd::stream::encode_all(std::io::Cursor::new(&data), temp_zstd_level)?;
                    writer.write_all(&(compressed.len() as u64).to_le_bytes())?;
                    writer.write_all(&compressed)?;
                }
            }
        }
        Ok(())
    }

    fn writer_for(&mut self, bucket: usize) -> Result<&mut BufWriter<File>> {
        if !self.open.contains_key(&bucket) {
            self.ensure_capacity()?;
            let path = self.root.join(format!("bucket_{bucket:05}.bin"));
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            self.open.insert(bucket, BufWriter::with_capacity(1024 * 1024, file));
        }
        self.touch(bucket);
        Ok(self.open.get_mut(&bucket).unwrap())
    }

    fn ensure_capacity(&mut self) -> Result<()> {
        while self.open.len() >= self.max_open {
            if let Some(bucket) = self.order.pop_front() {
                if let Some(mut writer) = self.open.remove(&bucket) {
                    writer.flush()?;
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    fn touch(&mut self, bucket: usize) {
        if let Some(pos) = self.order.iter().position(|&b| b == bucket) {
            self.order.remove(pos);
        }
        self.order.push_back(bucket);
    }

    fn finish(mut self) -> Result<()> {
        self.flush_pending()?;
        for (_, mut writer) in self.open.drain() {
            writer.flush()?;
        }
        Ok(())
    }
}

fn load_join_bucket<T>(root: &Path, worker_count: usize, bucket: usize, temp_compression: TempCompression) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let mut out = Vec::new();
    for worker in 0..worker_count {
        let path = root.join(format!("w{worker:02}/bucket_{bucket:05}.bin"));
        if !path.exists() {
            continue;
        }
        match temp_compression {
            TempCompression::Uncompressed => {
                let mut reader = BufReader::new(File::open(path)?);
                load_bincode_records(&mut reader, &mut out)?;
            }
            TempCompression::Zstd => {
                let mut reader = BufReader::new(File::open(path)?);
                loop {
                    if reader.fill_buf()?.is_empty() {
                        break;
                    }
                    let mut len_bytes = [0u8; 8];
                    reader.read_exact(&mut len_bytes)?;
                    let len = u64::from_le_bytes(len_bytes) as usize;
                    let mut compressed = vec![0u8; len];
                    reader.read_exact(&mut compressed)?;
                    let decompressed = zstd::stream::decode_all(std::io::Cursor::new(compressed))?;
                    let mut cursor = std::io::Cursor::new(decompressed);
                    load_bincode_records(&mut cursor, &mut out)?;
                }
            }
        }
    }
    Ok(out)
}

fn load_bincode_records<T, R>(reader: &mut R, out: &mut Vec<T>) -> Result<()>
where
    T: for<'de> Deserialize<'de>,
    R: Read,
{
    loop {
        match deserialize_from::<_, T>(&mut *reader) {
            Ok(value) => out.push(value),
            Err(err) => {
                if matches!(*err, bincode::ErrorKind::Io(ref io) if io.kind() == std::io::ErrorKind::UnexpectedEof) {
                    break;
                }
                return Err(err.into());
            }
        }
    }
    Ok(())
}

fn parquet_props(compression: OutputCompression, row_group_rows: usize) -> WriterProperties {
    WriterProperties::builder()
        .set_compression(compression.parquet())
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_max_row_group_size(row_group_rows)
        .build()
}

fn offer_rows_schema() -> SchemaRef {
    let price_struct = DataType::Struct(Fields::from(vec![
        Field::new("price", DataType::Float64, true),
        Field::new("currency", DataType::Utf8, true),
        Field::new("priority", DataType::Int32, true),
        Field::new("sourcePriceListId", DataType::Utf8, true),
    ]));
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("article_hash", DataType::Utf8, true),
        Field::new("_placeholder_vector", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), true),
        Field::new("ean", DataType::Utf8, true),
        Field::new("article_number", DataType::Utf8, true),
        Field::new("vendor_id", DataType::Utf8, true),
        Field::new("catalog_version_id", DataType::Utf8, true),
        Field::new("prices", DataType::List(Arc::new(Field::new("item", price_struct, true))), true),
        Field::new("delivery_time_days_max", DataType::Int32, true),
        Field::new("core_marker_enabled_sources", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("core_marker_disabled_sources", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("features", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("relationship_accessory_for", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("relationship_spare_part_for", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("relationship_similar_to", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("price_list_ids", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("currencies", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("eur_price_min", DataType::Float64, true),
        Field::new("eur_price_max", DataType::Float64, true),
        Field::new("chf_price_min", DataType::Float64, true),
        Field::new("chf_price_max", DataType::Float64, true),
        Field::new("huf_price_min", DataType::Float64, true),
        Field::new("huf_price_max", DataType::Float64, true),
        Field::new("pln_price_min", DataType::Float64, true),
        Field::new("pln_price_max", DataType::Float64, true),
        Field::new("gbp_price_min", DataType::Float64, true),
        Field::new("gbp_price_max", DataType::Float64, true),
        Field::new("czk_price_min", DataType::Float64, true),
        Field::new("czk_price_max", DataType::Float64, true),
        Field::new("cny_price_min", DataType::Float64, true),
        Field::new("cny_price_max", DataType::Float64, true),
    ]))
}

fn articles_schema() -> SchemaRef {
    let can_struct = DataType::Struct(Fields::from(vec![
        Field::new("value", DataType::Utf8, true),
        Field::new(
            "version_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));
    Arc::new(Schema::new(vec![
        Field::new("article_hash", DataType::Utf8, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("manufacturerName", DataType::Utf8, true),
        Field::new("category_l1", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("category_l2", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("category_l3", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("category_l4", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("category_l5", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        Field::new("eclass5_code", DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), true),
        Field::new("eclass7_code", DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), true),
        Field::new("s2class_code", DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), true),
        Field::new("text_codes", DataType::Utf8, true),
        Field::new("customer_article_numbers", DataType::List(Arc::new(Field::new("item", can_struct, true))), true),
        Field::new("eur_price_min", DataType::Float64, true),
        Field::new("eur_price_max", DataType::Float64, true),
        Field::new("chf_price_min", DataType::Float64, true),
        Field::new("chf_price_max", DataType::Float64, true),
        Field::new("huf_price_min", DataType::Float64, true),
        Field::new("huf_price_max", DataType::Float64, true),
        Field::new("pln_price_min", DataType::Float64, true),
        Field::new("pln_price_max", DataType::Float64, true),
        Field::new("gbp_price_min", DataType::Float64, true),
        Field::new("gbp_price_max", DataType::Float64, true),
        Field::new("czk_price_min", DataType::Float64, true),
        Field::new("czk_price_max", DataType::Float64, true),
        Field::new("cny_price_min", DataType::Float64, true),
        Field::new("cny_price_max", DataType::Float64, true),
    ]))
}

struct OfferRowRef<'a> {
    id: &'a str,
    article_hash: &'a str,
    ean: &'a str,
    article_number: &'a str,
    vendor_id: &'a str,
    catalog_version_id: &'a str,
    prices: &'a [ProjectedPrice],
    delivery_time_days_max: i32,
    core_marker_enabled_sources: &'a [String],
    core_marker_disabled_sources: &'a [String],
    features: &'a [String],
    relationship_accessory_for: &'a [String],
    relationship_spare_part_for: &'a [String],
    relationship_similar_to: &'a [String],
    price_list_ids: &'a [String],
    currencies: &'a [String],
    mins: &'a [f64; 7],
    maxs: &'a [f64; 7],
}

struct OfferRowsSink {
    writer: ArrowWriter<BufWriter<File>>,
    schema: SchemaRef,
    batch_rows: usize,
    rows: usize,
    b: OfferBuilders,
}

impl OfferRowsSink {
    fn new(path: PathBuf, schema: SchemaRef, batch_rows: usize, row_group_rows: usize, compression: OutputCompression) -> Result<Self> {
        let file = BufWriter::new(File::create(path)?);
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(parquet_props(compression, row_group_rows)))?;
        Ok(Self {
            writer,
            schema,
            batch_rows,
            rows: 0,
            b: OfferBuilders::new(batch_rows.max(1024)),
        })
    }

    fn append(&mut self, row: OfferRowRef<'_>) -> Result<()> {
        self.b.append(row)?;
        self.rows += 1;
        if self.rows >= self.batch_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let batch = self.b.finish(self.schema.clone())?;
        self.writer.write(&batch)?;
        self.b = OfferBuilders::new(self.batch_rows.max(1024));
        self.rows = 0;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.flush()?;
        self.writer.close()?;
        Ok(())
    }
}

struct OfferBuilders {
    id: StringBuilder,
    article_hash: StringBuilder,
    placeholder_vector: ListBuilder<Float32Builder>,
    ean: StringBuilder,
    article_number: StringBuilder,
    vendor_id: StringBuilder,
    catalog_version_id: StringBuilder,
    prices: ListBuilder<StructBuilder>,
    delivery_time_days_max: Int32Builder,
    core_marker_enabled_sources: ListBuilder<StringBuilder>,
    core_marker_disabled_sources: ListBuilder<StringBuilder>,
    features: ListBuilder<StringBuilder>,
    relationship_accessory_for: ListBuilder<StringBuilder>,
    relationship_spare_part_for: ListBuilder<StringBuilder>,
    relationship_similar_to: ListBuilder<StringBuilder>,
    price_list_ids: ListBuilder<StringBuilder>,
    currencies: ListBuilder<StringBuilder>,
    price_mins: [Float64Builder; 7],
    price_maxs: [Float64Builder; 7],
}

impl OfferBuilders {
    fn new(capacity: usize) -> Self {
        let price_struct = StructBuilder::new(
            Fields::from(vec![
                Field::new("price", DataType::Float64, true),
                Field::new("currency", DataType::Utf8, true),
                Field::new("priority", DataType::Int32, true),
                Field::new("sourcePriceListId", DataType::Utf8, true),
            ]),
            vec![
                Box::new(Float64Builder::with_capacity(capacity)),
                Box::new(StringBuilder::with_capacity(capacity, capacity * 16)),
                Box::new(Int32Builder::with_capacity(capacity)),
                Box::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            ],
        );
        Self {
            id: StringBuilder::with_capacity(capacity, capacity * 48),
            article_hash: StringBuilder::with_capacity(capacity, capacity * 32),
            placeholder_vector: ListBuilder::new(Float32Builder::with_capacity(capacity * 2)),
            ean: StringBuilder::with_capacity(capacity, capacity * 16),
            article_number: StringBuilder::with_capacity(capacity, capacity * 24),
            vendor_id: StringBuilder::with_capacity(capacity, capacity * 36),
            catalog_version_id: StringBuilder::with_capacity(capacity, capacity * 36),
            prices: ListBuilder::new(price_struct),
            delivery_time_days_max: Int32Builder::with_capacity(capacity),
            core_marker_enabled_sources: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            core_marker_disabled_sources: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            features: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            relationship_accessory_for: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            relationship_spare_part_for: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            relationship_similar_to: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            price_list_ids: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            currencies: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 8)),
            price_mins: std::array::from_fn(|_| Float64Builder::with_capacity(capacity)),
            price_maxs: std::array::from_fn(|_| Float64Builder::with_capacity(capacity)),
        }
    }

    fn append(&mut self, row: OfferRowRef<'_>) -> Result<()> {
        self.id.append_value(row.id);
        self.article_hash.append_value(row.article_hash);
        self.append_f32_list(&[0.0, 0.0]);
        self.ean.append_value(row.ean);
        self.article_number.append_value(row.article_number);
        self.vendor_id.append_value(row.vendor_id);
        self.catalog_version_id.append_value(row.catalog_version_id);
        self.append_prices(row.prices)?;
        self.delivery_time_days_max.append_value(row.delivery_time_days_max);
        Self::append_string_list(&mut self.core_marker_enabled_sources, row.core_marker_enabled_sources);
        Self::append_string_list(&mut self.core_marker_disabled_sources, row.core_marker_disabled_sources);
        Self::append_string_list(&mut self.features, row.features);
        Self::append_string_list(&mut self.relationship_accessory_for, row.relationship_accessory_for);
        Self::append_string_list(&mut self.relationship_spare_part_for, row.relationship_spare_part_for);
        Self::append_string_list(&mut self.relationship_similar_to, row.relationship_similar_to);
        Self::append_string_list(&mut self.price_list_ids, row.price_list_ids);
        Self::append_string_list(&mut self.currencies, row.currencies);
        for idx in 0..7 {
            self.price_mins[idx].append_value(row.mins[idx]);
            self.price_maxs[idx].append_value(row.maxs[idx]);
        }
        Ok(())
    }

    fn append_f32_list(&mut self, values: &[f32]) {
        let builder = self.placeholder_vector.values();
        for value in values {
            builder.append_value(*value);
        }
        self.placeholder_vector.append(true);
    }

    fn append_string_list(builder: &mut ListBuilder<StringBuilder>, values: &[String]) {
        let values_builder = builder.values();
        for value in values {
            values_builder.append_value(value);
        }
        builder.append(true);
    }

    fn append_prices(&mut self, prices: &[ProjectedPrice]) -> Result<()> {
        let builder = self.prices.values();
        for price in prices {
            builder.field_builder::<Float64Builder>(0).unwrap().append_value(price.price);
            builder.field_builder::<StringBuilder>(1).unwrap().append_value(&price.currency);
            builder.field_builder::<Int32Builder>(2).unwrap().append_value(price.priority);
            builder.field_builder::<StringBuilder>(3).unwrap().append_value(&price.source_price_list_id);
            builder.append(true);
        }
        self.prices.append(true);
        Ok(())
    }

    fn finish(&mut self, schema: SchemaRef) -> Result<RecordBatch> {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.id.finish()),
            Arc::new(self.article_hash.finish()),
            Arc::new(self.placeholder_vector.finish()),
            Arc::new(self.ean.finish()),
            Arc::new(self.article_number.finish()),
            Arc::new(self.vendor_id.finish()),
            Arc::new(self.catalog_version_id.finish()),
            Arc::new(self.prices.finish()),
            Arc::new(self.delivery_time_days_max.finish()),
            Arc::new(self.core_marker_enabled_sources.finish()),
            Arc::new(self.core_marker_disabled_sources.finish()),
            Arc::new(self.features.finish()),
            Arc::new(self.relationship_accessory_for.finish()),
            Arc::new(self.relationship_spare_part_for.finish()),
            Arc::new(self.relationship_similar_to.finish()),
            Arc::new(self.price_list_ids.finish()),
            Arc::new(self.currencies.finish()),
            Arc::new(self.price_mins[0].finish()),
            Arc::new(self.price_maxs[0].finish()),
            Arc::new(self.price_mins[1].finish()),
            Arc::new(self.price_maxs[1].finish()),
            Arc::new(self.price_mins[2].finish()),
            Arc::new(self.price_maxs[2].finish()),
            Arc::new(self.price_mins[3].finish()),
            Arc::new(self.price_maxs[3].finish()),
            Arc::new(self.price_mins[4].finish()),
            Arc::new(self.price_maxs[4].finish()),
            Arc::new(self.price_mins[5].finish()),
            Arc::new(self.price_maxs[5].finish()),
            Arc::new(self.price_mins[6].finish()),
            Arc::new(self.price_maxs[6].finish()),
        ];
        Ok(RecordBatch::try_new(schema, arrays)?)
    }
}

struct ArticlesSink {
    writer: ArrowWriter<BufWriter<File>>,
    schema: SchemaRef,
    batch_rows: usize,
    rows: usize,
    b: ArticleBuilders,
}

impl ArticlesSink {
    fn new(path: PathBuf, schema: SchemaRef, batch_rows: usize, row_group_rows: usize, compression: OutputCompression) -> Result<Self> {
        let file = BufWriter::new(File::create(path)?);
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(parquet_props(compression, row_group_rows)))?;
        Ok(Self {
            writer,
            schema,
            batch_rows,
            rows: 0,
            b: ArticleBuilders::new(batch_rows.max(1024)),
        })
    }

    fn append(&mut self, row: &FinalArticleRow) -> Result<()> {
        self.b.append(row)?;
        self.rows += 1;
        if self.rows >= self.batch_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let batch = self.b.finish(self.schema.clone())?;
        self.writer.write(&batch)?;
        self.b = ArticleBuilders::new(self.batch_rows.max(1024));
        self.rows = 0;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.flush()?;
        self.writer.close()?;
        Ok(())
    }
}

struct ArticleBuilders {
    article_hash: StringBuilder,
    name: StringBuilder,
    manufacturer_name: StringBuilder,
    category_l1: ListBuilder<StringBuilder>,
    category_l2: ListBuilder<StringBuilder>,
    category_l3: ListBuilder<StringBuilder>,
    category_l4: ListBuilder<StringBuilder>,
    category_l5: ListBuilder<StringBuilder>,
    eclass5_code: ListBuilder<Int32Builder>,
    eclass7_code: ListBuilder<Int32Builder>,
    s2class_code: ListBuilder<Int32Builder>,
    text_codes: StringBuilder,
    customer_article_numbers: ListBuilder<StructBuilder>,
    price_mins: [Float64Builder; 7],
    price_maxs: [Float64Builder; 7],
}

impl ArticleBuilders {
    fn new(capacity: usize) -> Self {
        let can_struct = StructBuilder::new(
            Fields::from(vec![
                Field::new("value", DataType::Utf8, true),
                Field::new(
                    "version_ids",
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    true,
                ),
            ]),
            vec![
                Box::new(StringBuilder::with_capacity(capacity, capacity * 16)),
                Box::new(ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24))),
            ],
        );
        Self {
            article_hash: StringBuilder::with_capacity(capacity, capacity * 32),
            name: StringBuilder::with_capacity(capacity, capacity * 32),
            manufacturer_name: StringBuilder::with_capacity(capacity, capacity * 32),
            category_l1: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            category_l2: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            category_l3: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            category_l4: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            category_l5: ListBuilder::new(StringBuilder::with_capacity(capacity, capacity * 24)),
            eclass5_code: ListBuilder::new(Int32Builder::with_capacity(capacity * 4)),
            eclass7_code: ListBuilder::new(Int32Builder::with_capacity(capacity * 4)),
            s2class_code: ListBuilder::new(Int32Builder::with_capacity(capacity * 4)),
            text_codes: StringBuilder::with_capacity(capacity, capacity * 64),
            customer_article_numbers: ListBuilder::new(can_struct),
            price_mins: std::array::from_fn(|_| Float64Builder::with_capacity(capacity)),
            price_maxs: std::array::from_fn(|_| Float64Builder::with_capacity(capacity)),
        }
    }

    fn append(&mut self, row: &FinalArticleRow) -> Result<()> {
        self.article_hash.append_value(&row.article_hash);
        self.name.append_value(&row.name);
        self.manufacturer_name.append_value(&row.manufacturer_name);
        Self::append_string_list(&mut self.category_l1, &row.category_l1);
        Self::append_string_list(&mut self.category_l2, &row.category_l2);
        Self::append_string_list(&mut self.category_l3, &row.category_l3);
        Self::append_string_list(&mut self.category_l4, &row.category_l4);
        Self::append_string_list(&mut self.category_l5, &row.category_l5);
        Self::append_i32_list(&mut self.eclass5_code, &row.eclass5_code);
        Self::append_i32_list(&mut self.eclass7_code, &row.eclass7_code);
        Self::append_i32_list(&mut self.s2class_code, &row.s2class_code);
        self.text_codes.append_value(&row.text_codes);
        self.append_customer_numbers(&row.customer_article_numbers)?;
        for idx in 0..7 {
            self.price_mins[idx].append_value(row.mins[idx]);
            self.price_maxs[idx].append_value(row.maxs[idx]);
        }
        Ok(())
    }

    fn append_string_list(builder: &mut ListBuilder<StringBuilder>, values: &[String]) {
        let values_builder = builder.values();
        for value in values {
            values_builder.append_value(value);
        }
        builder.append(true);
    }

    fn append_i32_list(builder: &mut ListBuilder<Int32Builder>, values: &[i32]) {
        let values_builder = builder.values();
        for value in values {
            values_builder.append_value(*value);
        }
        builder.append(true);
    }

    fn append_customer_numbers(&mut self, values: &[CustomerArticleNumberEntry]) -> Result<()> {
        let builder = self.customer_article_numbers.values();
        for entry in values {
            builder.field_builder::<StringBuilder>(0).unwrap().append_value(&entry.value);
            let version_ids = builder.field_builder::<ListBuilder<StringBuilder>>(1).unwrap();
            let inner = version_ids.values();
            for version_id in &entry.version_ids {
                inner.append_value(version_id);
            }
            version_ids.append(true);
            builder.append(true);
        }
        self.customer_article_numbers.append(true);
        Ok(())
    }

    fn finish(&mut self, schema: SchemaRef) -> Result<RecordBatch> {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.article_hash.finish()),
            Arc::new(self.name.finish()),
            Arc::new(self.manufacturer_name.finish()),
            Arc::new(self.category_l1.finish()),
            Arc::new(self.category_l2.finish()),
            Arc::new(self.category_l3.finish()),
            Arc::new(self.category_l4.finish()),
            Arc::new(self.category_l5.finish()),
            Arc::new(self.eclass5_code.finish()),
            Arc::new(self.eclass7_code.finish()),
            Arc::new(self.s2class_code.finish()),
            Arc::new(self.text_codes.finish()),
            Arc::new(self.customer_article_numbers.finish()),
            Arc::new(self.price_mins[0].finish()),
            Arc::new(self.price_maxs[0].finish()),
            Arc::new(self.price_mins[1].finish()),
            Arc::new(self.price_maxs[1].finish()),
            Arc::new(self.price_mins[2].finish()),
            Arc::new(self.price_maxs[2].finish()),
            Arc::new(self.price_mins[3].finish()),
            Arc::new(self.price_maxs[3].finish()),
            Arc::new(self.price_mins[4].finish()),
            Arc::new(self.price_maxs[4].finish()),
            Arc::new(self.price_mins[5].finish()),
            Arc::new(self.price_maxs[5].finish()),
            Arc::new(self.price_mins[6].finish()),
            Arc::new(self.price_maxs[6].finish()),
        ];
        Ok(RecordBatch::try_new(schema, arrays)?)
    }
}

fn default_threads() -> usize {
    num_cpus::get().max(1)
}
