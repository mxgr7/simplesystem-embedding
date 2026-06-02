#!/usr/bin/env python3
"""Tiny Prometheus exporter for Kafka topic/consumer-group offsets.

Exposes raw per-partition offsets so the Grafana dashboard can derive
producer rate, consumer rate, queue (topic) size, and consumer lag in PromQL:

  producer rate  = sum(rate(kafka_topic_partition_end_offset[$__rate_interval]))
  consumer rate  = sum(rate(kafka_group_partition_committed_offset[$__rate_interval]))
  queue size     = sum(kafka_topic_partition_end_offset - kafka_topic_partition_start_offset)
  consumer lag   = kafka_group_lag_total   (= sum(end) - sum(committed))

Reads SASL creds from .env (KAFKA_PROD_PORTAL_*) — no secrets on the cmdline.
Non-intrusive: watermarks via a throwaway consumer (no subscribe/join);
group offsets via AdminClient.list_consumer_group_offsets (OffsetFetch only).

Usage:
  uv run --with prometheus-client python scripts/kafka_lag_exporter.py \
     --topic prod.portal.marketplace.facts.articles.changed \
     --group article-indexer-v1-semantic --port 9145 --interval 15
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from confluent_kafka import Consumer, TopicPartition
from prometheus_client import Gauge, start_http_server


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def base_conf() -> dict:
    return {
        "bootstrap.servers": os.environ["KAFKA_PROD_PORTAL_BOOTSTRAP_SERVERS"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": os.environ["KAFKA_PROD_PORTAL_SASL_USERNAME"],
        "sasl.password": os.environ["KAFKA_PROD_PORTAL_SASL_PASSWORD"],
    }


END = Gauge("kafka_topic_partition_end_offset", "Log-end (high-watermark) offset", ["topic", "partition"])
START = Gauge("kafka_topic_partition_start_offset", "Earliest (low-watermark) offset", ["topic", "partition"])
COMMITTED = Gauge(
    "kafka_group_partition_committed_offset", "Committed offset for the group", ["group", "topic", "partition"]
)
LAG_TOTAL = Gauge("kafka_group_lag_total", "Total consumer lag (sum end - committed)", ["group", "topic"])
UP = Gauge("kafka_lag_exporter_up", "1 if the last topic-watermark poll succeeded")
GROUP_UP = Gauge("kafka_lag_exporter_group_up", "1 if the last committed-offset poll for this group succeeded", ["group"])


def poll_watermarks(consumer: Consumer, topic: str) -> tuple[list[int], dict[int, int]]:
    # Topic-level high/low watermarks — identical for every consumer group, so
    # fetched once per cycle and shared. cached=False forces a fresh broker query.
    md = consumer.list_topics(topic, timeout=20)
    parts = sorted(md.topics[topic].partitions.keys())
    end_by_p = {}
    for p in parts:
        lo, hi = consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10, cached=False)
        START.labels(topic, str(p)).set(lo)
        END.labels(topic, str(p)).set(hi)
        end_by_p[p] = hi
    return parts, end_by_p


def poll_committed(consumer: Consumer, topic: str, group: str, parts: list[int], end_by_p: dict[int, int]) -> None:
    # The consumer is configured with group.id=<group>; committed() issues an
    # OffsetFetch for that group WITHOUT subscribing/joining, so it never
    # becomes an active member and never disturbs the running indexer.
    committed = consumer.committed([TopicPartition(topic, p) for p in parts], timeout=30)
    lag = 0
    for tp in committed:
        if tp.offset is None or tp.offset < 0:
            continue
        COMMITTED.labels(group, topic, str(tp.partition)).set(tp.offset)
        lag += max(0, end_by_p.get(tp.partition, tp.offset) - tp.offset)
    LAG_TOTAL.labels(group, topic).set(lag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="prod.portal.marketplace.facts.articles.changed")
    ap.add_argument(
        "--group",
        action="append",
        help="consumer group to track (repeatable); defaults to article-indexer-v1-semantic",
    )
    ap.add_argument("--port", type=int, default=9145)
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    groups = args.group or ["article-indexer-v1-semantic"]
    load_env(Path(args.env))
    conf = base_conf()
    # One consumer per group: group.id must equal the group whose offsets we
    # OffsetFetch. Watermarks are topic-level, so we reuse the first consumer for them.
    consumers = {g: Consumer({**conf, "group.id": g, "enable.auto.commit": False}) for g in groups}
    wm_consumer = next(iter(consumers.values()))

    start_http_server(args.port)
    print(f"kafka_lag_exporter on :{args.port}  topic={args.topic} groups={groups} interval={args.interval}s")
    while True:
        try:
            parts, end_by_p = poll_watermarks(wm_consumer, args.topic)
            UP.set(1)
        except Exception as e:  # noqa: BLE001 - keep the exporter alive across transient broker errors
            UP.set(0)
            print(f"watermark poll error: {e}")
            time.sleep(args.interval)
            continue
        # Per-group isolation: a missing ACL / OffsetFetch error on one group
        # must not blank out the others' metrics.
        for g, c in consumers.items():
            try:
                poll_committed(c, args.topic, g, parts, end_by_p)
                GROUP_UP.labels(g).set(1)
            except Exception as e:  # noqa: BLE001
                GROUP_UP.labels(g).set(0)
                print(f"committed poll error group={g}: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
