#!/usr/bin/env python3
"""Generate the "Article Indexer — Bottleneck Analysis" Grafana dashboard JSON.

Single source of truth for the provisioned dashboard at
embedding-service/observability/grafana/dashboards/indexer-bottleneck.json.
Edit here and re-run; Grafana's file provisioner reloads the JSON.

  python scripts/gen_indexer_dashboard.py
"""
import json

DS = {"type": "prometheus", "uid": "prometheus"}
RI = "$__rate_interval"
TOPIC = "prod.portal.marketplace.facts.articles.changed"
# Backfill consumer (this project) + the live prod indexer's group — both read TOPIC.
GROUP = "article-indexer-v1-semantic"
LIVE_GROUP = "article-indexer-v1"
panels = []; _id = [1]; cur = {"x": 0, "y": 0}


def nid():
    v = _id[0]; _id[0] += 1; return v


def row(title):
    cur["x"] = 0; cur["y"] += cur.get("rh", 0); cur["rh"] = 0
    panels.append({"type": "row", "id": nid(), "title": title, "collapsed": False,
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": cur["y"]}})
    cur["y"] += 1


def place(w, h):
    if cur["x"] + w > 24: cur["x"] = 0; cur["y"] += cur["rh"]; cur["rh"] = 0
    g = {"h": h, "w": w, "x": cur["x"], "y": cur["y"]}; cur["x"] += w; cur["rh"] = max(cur["rh"], h); return g


def ts(title, targets, unit="short", w=12, h=8, stack=None, fill=8, desc=""):
    p = {"type": "timeseries", "id": nid(), "title": title, "datasource": DS, "description": desc,
         "gridPos": place(w, h),
         "fieldConfig": {"defaults": {"unit": unit, "custom": {"drawStyle": "line", "lineWidth": 2,
           "fillOpacity": fill, "showPoints": "never", "stacking": {"mode": stack or "none"}}}, "overrides": []},
         "options": {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "max"]},
                     "tooltip": {"mode": "multi", "sort": "desc"}},
         "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": l} for i, (e, l) in enumerate(targets)]}
    panels.append(p); return p


def stat(title, expr, unit="short", w=6, h=4, dec=0, color="thresholds", desc=""):
    panels.append({"type": "stat", "id": nid(), "title": title, "datasource": DS, "description": desc,
       "gridPos": place(w, h),
       "fieldConfig": {"defaults": {"unit": unit, "decimals": dec, "color": {"mode": color},
         "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []},
       "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"},
       "targets": [{"refId": "A", "datasource": DS, "expr": expr}]})


def avg(stage):  # avg ms for a phase timer
    return (f'1000*rate(article_search_indexer_phase_{stage}_seconds_sum[{RI}])'
            f'/clamp_min(rate(article_search_indexer_phase_{stage}_seconds_count[{RI}]),1)')


STAGES = [("mongo_fetch", "mongo_fetch (incl backpressure)"), ("enrich", "enrich (postgres)"),
          ("mapping", "mapping"), ("embedding_mget", "embedding_mget (ES reuse)"),
          ("embedding_tei", "embedding_tei"), ("es_write", "es_write")]

# ---- Kafka ----
# Offsets are huge (~1e9/partition). rate() turns any step discontinuity
# (exporter cold-start 0->offset, or an operator re-seed that rewinds the
# committed offset = a counter "reset") into a giant false spike (~1e8/s).
# Real rates are <~2k/s, so clamp_max at 100k cleanly drops the artifacts
# without touching real data.
CAP = 100000
P_RATE = f'clamp_max(sum(rate(kafka_topic_partition_end_offset{{topic="{TOPIC}"}}[{RI}])),{CAP})'
ETADESC = ("lag / net-drain, net-drain = -d(lag)/dt over 10m (smoothed). "
           "Unreliable for ~10m after a re-seed (the offset rewind distorts the slope); "
           "a huge value means net drain <= 0 (not catching up).")


def kafka_section(title, group):
    """One Kafka producer/consumer/lag/ETA section for a single consumer group.
    Producer panels are identical across groups (same topic) but repeated so each
    section is self-contained."""
    c_rate = f'clamp_max(sum(rate(kafka_group_partition_committed_offset{{group="{group}"}}[{RI}])),{CAP})'
    net_rate = f'{c_rate} - {P_RATE}'
    lag = f'kafka_group_lag_total{{group="{group}"}}'
    row(title)
    ts("Producer vs Consumer rate",
       [(P_RATE, "producer (topic append/s)"), (c_rate, "consumer (committed/s)")],
       unit="cps", w=12, desc="Net = consumer - producer. Rates clamped at 100k/s to drop re-seed/restart rate() artifacts.")
    ts("Net drain rate (consumer - producer)", [(net_rate, "net msg/s")], unit="cps", w=12)
    ts("Consumer lag (backlog)", [(lag, "lag")], unit="short", w=12)
    ts("Topic queue size (retained)",
       [(f'sum(kafka_topic_partition_end_offset{{topic="{TOPIC}"}} - kafka_topic_partition_start_offset{{topic="{TOPIC}"}})', "retained msgs")], unit="short", w=12)
    stat("Lag now", lag, w=6)
    stat("Consumer rate", c_rate, unit="cps", w=6)
    stat("Producer rate", P_RATE, unit="cps", w=6)
    stat("Net drain", net_rate, unit="cps", w=6)
    # ETA = lag / net-drain. net-drain = -d(lag)/dt over 10m, taken from the single
    # clean `lag` gauge so it sidesteps rate()-on-offset reset spikes.
    drain = f'clamp_min(-sum(deriv({lag}[10m])), 1)'
    eta = f'sum({lag}) / {drain}'
    stat("ETA until caught up", eta, unit="dtdurations", w=6, desc=ETADESC)
    ts("ETA until caught up (trend)", [(eta, "eta")], unit="dtdurations", w=18, fill=0,
       desc=ETADESC + " Trending down = converging; flat/rising = falling behind.")


kafka_section("Kafka — article-indexer-v1-semantic (backfill consumer)", GROUP)
kafka_section("Kafka — article-indexer-v1 (live prod consumer)", LIVE_GROUP)

# ---- Change composition & buffer fan-out ----
row("Change composition & buffer fan-out")
ts("Change type by embedding impact",
   [(f'sum by (kind)(rate(article_search_indexer_change_embedding_total[{RI}]))', "{{kind}}")],
   unit="cps", w=12, stack="normal", fill=60,
   desc="Per-article: changed=an embedding-input field (name/desc/manufacturer/category/ean/mfr-no/type) actually "
        "changed → re-embed (GPU cost); unchanged=vectors reused; new=first index; model-bump=embedding model changed.")
ts("Embedding-input change ratio",
   [(f'sum(rate(article_search_indexer_change_embedding_total{{kind="changed"}}[{RI}]))'
     f'/clamp_min(sum(rate(article_search_indexer_change_embedding_total[{RI}])),1)', "changed fraction"),
    (f'sum(rate(article_search_indexer_change_embedding_total{{kind="new"}}[{RI}]))'
     f'/clamp_min(sum(rate(article_search_indexer_change_embedding_total[{RI}])),1)', "new fraction")],
   unit="percentunit", w=12, fill=0,
   desc="Fraction of processed articles touching embedding inputs — the share of the change stream that hits TEI/GPU.")
ts("Buffer vendor fan-out (vendors / buffer)",
   [(f'histogram_quantile(0.5, sum by (le)(rate(article_search_indexer_buffer_vendor_count_bucket[{RI}])))', "p50"),
    (f'histogram_quantile(0.95, sum by (le)(rate(article_search_indexer_buffer_vendor_count_bucket[{RI}])))', "p95"),
    (f'histogram_quantile(0.99, sum by (le)(rate(article_search_indexer_buffer_vendor_count_bucket[{RI}])))', "p99"),
    (f'rate(article_search_indexer_buffer_vendor_count_sum[{RI}])'
     f'/clamp_min(rate(article_search_indexer_buffer_vendor_count_count[{RI}]),1)', "avg")],
   unit="short", w=12, fill=0,
   desc="Distinct vendors per fetched buffer = number of per-vendor $in query waves (each wave capped at "
        "MAX_CONCURRENT_QUERIES=8). High p95/p99 marks the slow Mongo-fetch stretches.")
ts("Articles per buffer",
   [(f'histogram_quantile(0.5, sum by (le)(rate(article_search_indexer_buffer_article_count_bucket[{RI}])))', "p50"),
    (f'rate(article_search_indexer_buffer_article_count_sum[{RI}])'
     f'/clamp_min(rate(article_search_indexer_buffer_article_count_count[{RI}]),1)', "avg")],
   unit="short", w=12, fill=0)

# ---- Throughput / outcomes ----
row("Indexer — throughput & outcomes")
ts("Articles indexed/s", [(f'rate(article_search_indexer_processed_count_total[{RI}])', "processed/s")], unit="cps", w=8)
ts("Dead-letter/s", [(f'rate(article_search_indexer_failed_count_total[{RI}])', "dead-letter/s")], unit="cps", w=8)
ts("Batch wall time (avg)", [(f'1000*rate(article_search_indexer_batches_seconds_sum[{RI}])/clamp_min(rate(article_search_indexer_batches_seconds_count[{RI}]),1)', "ms/batch")], unit="ms", w=8)

# ---- Stage latency (THE bottleneck finder) ----
row("Pipeline stage latency — bottleneck finder")
ts("Avg latency per stage", [(avg(s), l) for s, l in STAGES], unit="ms", w=12, fill=0,
   desc="Highest line = current bottleneck. mongo_fetch is backpressure-inflated (wraps downstream).")
ts("Stage time share (stacked)", [(avg(s), l) for s, l in STAGES], unit="ms", w=12, stack="normal", fill=60)
ts("Stage completions/s", [(f'rate(article_search_indexer_phase_{s}_seconds_count[{RI}])', l) for s, l in STAGES], unit="cps", w=12, fill=0)
ts("Stage max latency", [(f'1000*article_search_indexer_phase_{s}_seconds_max', l) for s, l in STAGES], unit="ms", w=12, fill=0)

# ---- Backpressure / contention ----
row("Backpressure, queueing & contention")
ts("Reactor scheduler: submitted vs completed",
   [(f'rate(article_search_indexer_scheduler_scheduler_tasks_submitted_total[{RI}])', "submitted/s"),
    (f'rate(article_search_indexer_scheduler_scheduler_tasks_completed_seconds_count[{RI}])', "completed/s")], unit="cps", w=12)
ts("Scheduler active / pending tasks",
   [('article_search_indexer_scheduler_scheduler_tasks_active_seconds_gcount', "active"),
    ('article_search_indexer_scheduler_scheduler_tasks_pending_seconds_gcount', "pending")], unit="short", w=12)
ts("Mongo pool checkout wait (avg)",
   [(f'1000*rate(article_search_indexer_mongo_pool_checkout_seconds_sum{{result="success"}}[{RI}])/clamp_min(rate(article_search_indexer_mongo_pool_checkout_seconds_count{{result="success"}}[{RI}]),1)', "ms wait"),
    (f'rate(article_search_indexer_mongo_pool_checkout_seconds_count{{result="failed"}}[{RI}])', "checkout failures/s")], unit="ms", w=12)
ts("In-flight batches / articles",
   [('article_search_indexer_batches_active_seconds_gcount', "active batches"),
    ('article_search_indexer_articles_active_seconds_gcount', "active articles")], unit="short", w=12)

# ---- Embedding subsystem ----
row("Embedding subsystem (embedding-service + TEI)")
ts("embedding-service requests/s by status",
   [(f'sum by (status)(rate(embedding_service_requests_total[{RI}]))', "{{status}}")], unit="cps", w=12)
ts("Cache hit ratio", [('embedding_service_cache_hit_ratio', "hit ratio")], unit="percentunit", w=6)
ts("embedding-service in-flight", [('embedding_service_inflight', "inflight")], unit="short", w=6)
ts("TEI semaphore wait (avg)",
   [(f'1000*rate(embedding_service_tei_semaphore_wait_seconds_sum[{RI}])/clamp_min(rate(embedding_service_tei_semaphore_wait_seconds_count[{RI}]),1)', "ms wait")], unit="ms", w=8)
ts("TEI calls vs failures (indexer)",
   [(f'rate(article_search_indexer_embedding_tei_call_count_total[{RI}])', "tei calls/s"),
    (f'rate(article_search_indexer_embedding_tei_failure_count_total[{RI}])', "tei failures/s")], unit="cps", w=8)
ts("TEI backend in-flight / healthy",
   [('embedding_service_tei_backend_inflight', "inflight {{backend}}"),
    ('embedding_service_tei_backend_healthy', "healthy {{backend}}")], unit="short", w=8)

# ---- JVM (indexer) ----
row("JVM — indexer (CPU / heap / GC / threads)")
ts("Process CPU", [('process_cpu_usage{job="indexer"}*100', "process cpu %"),
                  ('system_cpu_count{job="indexer"}*100', "cores*100 (ceiling)")], unit="percent", w=12)
ts("Heap used vs max",
   [('sum(jvm_memory_used_bytes{job="indexer",area="heap"})', "heap used"),
    ('sum(jvm_memory_committed_bytes{job="indexer",area="heap"})', "heap committed"),
    ('sum(jvm_memory_max_bytes{job="indexer",area="heap"})', "heap max")], unit="bytes", w=12)
ts("GC pause time & overhead",
   [(f'rate(jvm_gc_pause_seconds_sum{{job="indexer"}}[{RI}])', "gc time s/s"),
    ('jvm_gc_overhead{job="indexer"}', "gc overhead")], unit="short", w=12)
ts("Threads (live / virtual / pinned)",
   [('jvm_threads_live_threads{job="indexer"}', "live"),
    ('jvm_threads_virtual_live_threads{job="indexer"}', "virtual live"),
    (f'rate(jvm_threads_virtual_pinned_seconds_count{{job="indexer"}}[{RI}])', "vthread pinned/s")], unit="short", w=12)

# ---- Host & hardware ----
row("Host & hardware (CPU / mem / disk / net / GPU / kvrocks)")
ts("Host CPU by mode",
   [(f'sum by (mode)(rate(node_cpu_seconds_total{{job="node",mode!="idle"}}[{RI}])) / scalar(count(count by (cpu)(node_cpu_seconds_total{{job="node"}})))', "{{mode}}")],
   unit="percentunit", w=12, stack="normal", fill=50)
ts("Load1 vs cores", [('node_load1{job="node"}', "load1"),
                     ('count(count by (cpu)(node_cpu_seconds_total{job="node"}))', "cores")], unit="short", w=12)
ts("Memory available", [('node_memory_MemAvailable_bytes{job="node"}', "available"),
                       ('node_memory_MemTotal_bytes{job="node"}', "total")], unit="bytes", w=12)
ts("Disk IO utilisation", [(f'rate(node_disk_io_time_seconds_total{{job="node"}}[{RI}])', "{{device}} busy")], unit="percentunit", w=12)
ts("Network throughput",
   [(f'rate(node_network_receive_bytes_total{{job="node",device!~"lo|veth.*|docker.*|br.*"}}[{RI}])', "rx {{device}}"),
    (f'rate(node_network_transmit_bytes_total{{job="node",device!~"lo|veth.*|docker.*|br.*"}}[{RI}])', "tx {{device}}")], unit="Bps", w=12)
ts("GPU utilisation & memory",
   [('DCGM_FI_DEV_GPU_UTIL', "gpu util %"),
    ('DCGM_FI_DEV_FB_USED/(DCGM_FI_DEV_FB_USED+DCGM_FI_DEV_FB_FREE)*100', "gpu mem %")], unit="percent", w=12)
ts("KVRocks ops/s & cache hit ratio",
   [(f'rate(redis_commands_processed_total{{job="kvrocks"}}[{RI}])', "ops/s"),
    (f'rate(redis_keyspace_hits_total{{job="kvrocks"}}[{RI}])/clamp_min(rate(redis_keyspace_hits_total{{job="kvrocks"}}[{RI}])+rate(redis_keyspace_misses_total{{job="kvrocks"}}[{RI}]),1)', "keyspace hit ratio")], unit="short", w=12)

dash = {"title": "Article Indexer — Bottleneck Analysis", "uid": "indexer-bottleneck",
        "schemaVersion": 39, "version": 1, "editable": True, "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"}, "timezone": "", "tags": ["indexer", "bottleneck", "v1-semantic"],
        "templating": {"list": []}, "panels": panels}
open("embedding-service/observability/grafana/dashboards/indexer-bottleneck.json", "w").write(json.dumps(dash, indent=2))
print(f"wrote dashboard: {len(panels)} panels ({sum(1 for p in panels if p['type']=='row')} rows)")
