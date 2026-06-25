use anyhow::{Context, Result, anyhow};
use clap::Parser;
use flate2::Compression;
use flate2::write::GzEncoder;
use futures_util::TryStreamExt;
use mongodb::Client;
use mongodb::bson::{Bson, Document, doc};
use mongodb::options::Hint;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const COMPOUND_INDEX: &str = "vendorId_1_articleNumber_1";

#[derive(Parser, Debug)]
#[command(
    about = "Parallel, projected MongoDB offer exporter (vendorId-filtered, NDJSON.gz output)"
)]
struct Args {
    /// MongoDB URI. Append e.g. `?readPreference=secondaryPreferred` to offload primary.
    #[arg(long)]
    uri: String,
    /// Database name.
    #[arg(long)]
    db: String,
    /// Collection name.
    #[arg(long, default_value = "offers")]
    collection: String,
    /// Vendor UUID to filter on (hyphenated form).
    #[arg(long)]
    vendor_id: String,
    /// Output file path (e.g. vendor_<UUID>.json.gz). Workers write parts to <out>.parts/
    /// then the file is produced by concatenating them.
    #[arg(long)]
    out: PathBuf,
    /// Parallel workers. Each scans a disjoint articleNumber sub-range.
    #[arg(long, default_value_t = num_cpus::get())]
    workers: usize,
    /// Cursor batch size.
    #[arg(long, default_value_t = 10_000)]
    batch_size: u32,
    /// Gzip level (1=fast, 9=best). Use 1 for max throughput.
    #[arg(long, default_value_t = 1)]
    gzip_level: u32,
    /// Keep per-worker parts after concatenation.
    #[arg(long)]
    keep_parts: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct OfferDoc {
    #[serde(rename = "articleNumber")]
    article_number: String,
    #[serde(rename = "vendorId", serialize_with = "ser_uuid_str")]
    vendor_id: mongodb::bson::Uuid,
    offer: OfferWrap,
}

#[derive(Debug, Deserialize, Serialize)]
struct OfferWrap {
    #[serde(rename = "offerParams")]
    offer_params: OfferParams,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct OfferParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(
        rename = "categoryPaths",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    category_paths: Vec<CategoryPath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ean: Option<String>,
    #[serde(
        rename = "manufacturerName",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    manufacturer_name: Option<String>,
    #[serde(
        rename = "manufacturerArticleNumber",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    manufacturer_article_number: Option<String>,
    #[serde(
        rename = "manufacturerArticleType",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    manufacturer_article_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CategoryPath {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    elements: Vec<String>,
}

fn ser_uuid_str<S: serde::Serializer>(
    u: &mongodb::bson::Uuid,
    s: S,
) -> std::result::Result<S::Ok, S::Error> {
    s.collect_str(u)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    let vendor_uuid = uuid::Uuid::parse_str(&args.vendor_id).context("invalid --vendor-id UUID")?;
    let vendor_bson = mongodb::bson::Uuid::from_bytes(vendor_uuid.into_bytes());

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let parts_dir = {
        let mut p = args.out.clone();
        let name = p
            .file_name()
            .ok_or_else(|| anyhow!("--out has no file name"))?
            .to_owned();
        p.set_file_name(format!("{}.parts", name.to_string_lossy()));
        p
    };
    std::fs::create_dir_all(&parts_dir)?;

    let client = Client::with_uri_str(&args.uri)
        .await
        .context("connecting to MongoDB")?;
    let db = client.database(&args.db);
    let raw_coll = db.collection::<Document>(&args.collection);

    eprintln!(
        "Computing {} buckets via $bucketAuto on articleNumber (vendor {}) ...",
        args.workers, vendor_uuid
    );
    let t_buckets = Instant::now();
    let pipeline = vec![
        doc! { "$match": { "vendorId": vendor_bson } },
        doc! { "$bucketAuto": {
            "groupBy": "$articleNumber",
            "buckets": args.workers as i32,
        }},
    ];
    let mut agg_cursor = raw_coll
        .aggregate(pipeline)
        .hint(Hint::Name(COMPOUND_INDEX.to_string()))
        .await?;
    let mut buckets: Vec<(Bson, Bson)> = Vec::new();
    while let Some(d) = agg_cursor.try_next().await? {
        let id = d.get_document("_id")?;
        let min = id.get("min").cloned().unwrap_or(Bson::Null);
        let max = id.get("max").cloned().unwrap_or(Bson::Null);
        buckets.push((min, max));
    }
    if buckets.is_empty() {
        return Err(anyhow!("no documents for vendor"));
    }
    eprintln!(
        "got {} buckets in {:.1}s",
        buckets.len(),
        t_buckets.elapsed().as_secs_f64()
    );

    let typed_coll = db.collection::<OfferDoc>(&args.collection);
    let projection = doc! {
        "_id": 0,
        "articleNumber": 1,
        "vendorId": 1,
        "offer.offerParams.name": 1,
        "offer.offerParams.categoryPaths": 1,
        "offer.offerParams.description": 1,
        "offer.offerParams.ean": 1,
        "offer.offerParams.manufacturerName": 1,
        "offer.offerParams.manufacturerArticleNumber": 1,
        "offer.offerParams.manufacturerArticleType": 1,
    };

    let total = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let progress = {
        let total = total.clone();
        let stop = stop.clone();
        tokio::spawn(async move {
            let mut last = 0u64;
            let start = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_secs(5)).await;
                let cur = total.load(Ordering::Relaxed);
                let dt = start.elapsed().as_secs_f64();
                let avg = if dt > 0.0 { cur as f64 / dt } else { 0.0 };
                let inst = (cur.saturating_sub(last)) as f64 / 5.0;
                eprintln!("progress: {cur} docs ({avg:.0}/s avg, {inst:.0}/s last-5s)");
                last = cur;
            }
        })
    };

    let total_workers = buckets.len();
    let mut set = tokio::task::JoinSet::new();
    for (i, (min_b, max_b)) in buckets.into_iter().enumerate() {
        let is_last = i == total_workers - 1;
        let coll = typed_coll.clone();
        let projection = projection.clone();
        let part_path = parts_dir.join(format!("part-{:04}.ndjson.gz", i));
        let batch_size = args.batch_size;
        let level = args.gzip_level;
        let total = total.clone();
        let vendor = vendor_bson;
        set.spawn(async move {
            let an_filter = if is_last {
                doc! { "$gte": min_b, "$lte": max_b }
            } else {
                doc! { "$gte": min_b, "$lt": max_b }
            };
            let filter = doc! { "vendorId": vendor, "articleNumber": an_filter };
            let file = File::create(&part_path)?;
            let buf = BufWriter::with_capacity(4 << 20, file);
            let mut gz = GzEncoder::new(buf, Compression::new(level));
            let mut cursor = coll
                .find(filter)
                .projection(projection)
                .hint(Hint::Name(COMPOUND_INDEX.to_string()))
                .batch_size(batch_size)
                .no_cursor_timeout(true)
                .await?;
            let mut local: u64 = 0;
            let mut line_buf: Vec<u8> = Vec::with_capacity(4096);
            while let Some(d) = cursor.try_next().await? {
                line_buf.clear();
                serde_json::to_writer(&mut line_buf, &d)?;
                line_buf.push(b'\n');
                gz.write_all(&line_buf)?;
                local += 1;
                if local & 1023 == 0 {
                    total.fetch_add(1024, Ordering::Relaxed);
                }
            }
            total.fetch_add(local & 1023, Ordering::Relaxed);
            gz.finish()?.flush()?;
            Ok::<u64, anyhow::Error>(local)
        });
    }

    let mut totals: Vec<u64> = Vec::new();
    while let Some(res) = set.join_next().await {
        totals.push(res??);
    }
    stop.store(true, Ordering::Relaxed);
    let _ = progress.await;

    let grand: u64 = totals.iter().sum();
    eprintln!(
        "workers complete: {grand} docs across {} parts. concatenating...",
        total_workers
    );

    let out_file = File::create(&args.out)?;
    let mut out = BufWriter::with_capacity(8 << 20, out_file);
    for i in 0..total_workers {
        let p = parts_dir.join(format!("part-{:04}.ndjson.gz", i));
        let f = File::open(&p)?;
        let mut r = BufReader::with_capacity(8 << 20, f);
        std::io::copy(&mut r, &mut out)?;
    }
    out.flush()?;
    drop(out);

    if !args.keep_parts {
        for i in 0..total_workers {
            let _ = std::fs::remove_file(parts_dir.join(format!("part-{:04}.ndjson.gz", i)));
        }
        let _ = std::fs::remove_dir(&parts_dir);
    }

    eprintln!("DONE: {} docs -> {}", grand, args.out.display());
    Ok(())
}
