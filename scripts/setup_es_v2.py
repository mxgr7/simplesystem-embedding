"""Create local-article-index-v2 cloned from v1, plus the TEST_PROFILE_18 additions.

Cloned verbatim from v1:
  - number_of_shards
  - index.analysis (all custom analyzers/normalizers/tokenizers/filters/char_filters)
  - all legacy mapping properties

Set for import (per BULK_IMPORT_TUNING.md):
  - number_of_replicas: 0
  - refresh_interval: -1
  - translog.durability: async, translog.flush_threshold_size: 2gb

Added vs v1:
  - _source.excludes: ["embeddings.vector"]
  - properties.embeddings.properties.inputHash (keyword, doc_values:false)
  - properties.embeddingModelVersion (keyword)

The script is idempotent — re-running deletes and recreates v2.
Pass --confirm-delete to allow dropping an existing v2.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import httpx


READONLY_SETTING_KEYS = {
    "creation_date",
    "creation_date_string",
    "history",
    "provided_name",
    "uuid",
    "version",
    "routing",
}


def fetch(client: httpx.Client, path: str) -> dict:
    r = client.get(path)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--src", default="local-article-index-v1")
    ap.add_argument("--dst", default="local-article-index-v2")
    ap.add_argument("--confirm-delete", action="store_true",
                    help="delete --dst first if it already exists")
    args = ap.parse_args()

    with httpx.Client(base_url=args.es, timeout=60.0) as cli:
        if cli.head(f"/{args.dst}").status_code == 200:
            if not args.confirm_delete:
                sys.exit(f"{args.dst} already exists; rerun with --confirm-delete")
            print(f"deleting existing {args.dst} ...", flush=True)
            r = cli.delete(f"/{args.dst}")
            r.raise_for_status()

        src_settings = fetch(cli, f"/{args.src}/_settings")[args.src]["settings"]["index"]
        src_mapping = fetch(cli, f"/{args.src}/_mapping")[args.src]["mappings"]

        idx_settings = {k: v for k, v in src_settings.items() if k not in READONLY_SETTING_KEYS}
        idx_settings["number_of_replicas"] = "0"
        idx_settings["refresh_interval"] = "-1"
        idx_settings["translog"] = {
            "durability": "async",
            "sync_interval": "120s",
            "flush_threshold_size": "2gb",
        }

        mappings = copy.deepcopy(src_mapping)
        mappings.setdefault("_source", {})["excludes"] = ["embeddings.vector"]
        props = mappings["properties"]
        emb = props.setdefault("embeddings", {"type": "nested", "properties": {}})
        emb.setdefault("properties", {})["inputHash"] = {
            "type": "keyword",
            "doc_values": False,
        }
        props["embeddingModelVersion"] = {"type": "keyword"}

        body = {"settings": {"index": idx_settings}, "mappings": mappings}
        print(f"creating {args.dst} (shards={idx_settings.get('number_of_shards')}, "
              f"replicas={idx_settings['number_of_replicas']}, "
              f"refresh={idx_settings['refresh_interval']}) ...", flush=True)
        r = cli.put(f"/{args.dst}", json=body)
        if r.status_code != 200:
            print("ERROR:", r.text, file=sys.stderr)
            r.raise_for_status()
        print("  acknowledged:", r.json().get("acknowledged"))

        # Verify the new bits made it in.
        m = fetch(cli, f"/{args.dst}/_mapping")[args.dst]["mappings"]
        print("\nverification:")
        print("  _source.excludes:",
              m.get("_source", {}).get("excludes"))
        print("  embeddings.inputHash:",
              m["properties"]["embeddings"]["properties"].get("inputHash"))
        print("  embeddingModelVersion:",
              m["properties"].get("embeddingModelVersion"))
        n_analyzers = len(src_settings.get("analysis", {}).get("analyzer", {}))
        print(f"  analyzers cloned: {n_analyzers}")
        s = fetch(cli, f"/{args.dst}/_settings")[args.dst]["settings"]["index"]
        print(f"  translog: {s.get('translog')}")


if __name__ == "__main__":
    main()
