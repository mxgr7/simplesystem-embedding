"""Create local-article-index-v5 for the staging-clone embedding import.

Mappings:  taken verbatim from target_mapping.json (the local-article-index-v4
           mapping — already includes nested `embeddings`, `embeddingModelVersion`,
           `catalogVersionIds`, `priceKeys`, and `_source.excludes:[embeddings.vector]`).
Settings:  index.analysis (+ other portable index settings) copied from the
           staging clone `stg-articles-v1-clone-20260516`, which carries every
           analyzer/normalizer the mapping references. Ingest tuning per
           BULK_IMPORT_TUNING.md: replicas 0, refresh -1, async translog.

Idempotent — pass --confirm-delete to drop an existing --dst.

Run:
    uv run --no-project python scripts/setup_es_v5.py [--confirm-delete]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

# Settings keys ES rejects on create (per-index identity / lifecycle state).
READONLY_SETTING_KEYS = {
    "creation_date",
    "creation_date_string",
    "history",
    "provided_name",
    "uuid",
    "version",
    "routing",  # tier allocation from the source cluster — not portable
}

DEFAULT_MAPPING = str(Path(__file__).resolve().parents[1] / "target_mapping.json")


def fetch(client: httpx.Client, path: str) -> dict:
    r = client.get(path)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default="http://localhost:9200")
    ap.add_argument("--settings-src", default="stg-articles-v1-clone-20260516",
                    help="index to copy index.analysis + portable settings from")
    ap.add_argument("--dst", default="local-article-index-v5")
    ap.add_argument("--mapping", default=DEFAULT_MAPPING,
                    help="path to target_mapping.json (mappings only)")
    ap.add_argument("--confirm-delete", action="store_true",
                    help="delete --dst first if it already exists")
    args = ap.parse_args()

    mapping_doc = json.loads(Path(args.mapping).read_text())
    mappings = mapping_doc["mappings"] if "mappings" in mapping_doc else mapping_doc

    with httpx.Client(base_url=args.es, timeout=60.0) as cli:
        if cli.head(f"/{args.dst}").status_code == 200:
            if not args.confirm_delete:
                sys.exit(f"{args.dst} already exists; rerun with --confirm-delete")
            print(f"deleting existing {args.dst} ...", flush=True)
            cli.delete(f"/{args.dst}").raise_for_status()

        src_idx = fetch(cli, f"/{args.settings_src}/_settings")
        src_settings = src_idx[args.settings_src]["settings"]["index"]

        idx_settings = {
            k: v for k, v in src_settings.items() if k not in READONLY_SETTING_KEYS
        }
        idx_settings["number_of_replicas"] = "0"
        idx_settings["refresh_interval"] = "-1"
        idx_settings["translog"] = {
            "durability": "async",
            "sync_interval": "120s",
            "flush_threshold_size": "2gb",
        }

        analyzers = sorted(src_settings.get("analysis", {}).get("analyzer", {}))
        normalizers = sorted(src_settings.get("analysis", {}).get("normalizer", {}))
        print(
            f"creating {args.dst} "
            f"(shards={idx_settings.get('number_of_shards')}, "
            f"replicas={idx_settings['number_of_replicas']}, "
            f"refresh={idx_settings['refresh_interval']}, "
            f"analyzers={len(analyzers)}, normalizers={len(normalizers)}) ...",
            flush=True,
        )

        body = {"settings": {"index": idx_settings}, "mappings": mappings}
        r = cli.put(f"/{args.dst}", json=body)
        if r.status_code != 200:
            print("ERROR:", r.text, file=sys.stderr)
            r.raise_for_status()
        print("  acknowledged:", r.json().get("acknowledged"))

        # Verify the embedding-relevant bits round-tripped.
        m = fetch(cli, f"/{args.dst}/_mapping")[args.dst]["mappings"]
        props = m["properties"]
        vec = props["embeddings"]["properties"]["vector"]
        print("\nverification:")
        print("  _source.excludes:     ", m.get("_source", {}).get("excludes"))
        print("  embeddings.vector:    ",
              {k: vec.get(k) for k in ("type", "dims", "similarity", "index")},
              vec.get("index_options"))
        print("  embeddings.inputHash: ",
              props["embeddings"]["properties"].get("inputHash"))
        print("  embeddingModelVersion:", props.get("embeddingModelVersion"))
        print("  catalogVersionIds:    ", props.get("catalogVersionIds"))
        print("  priceKeys:            ", props.get("priceKeys"))
        s = fetch(cli, f"/{args.dst}/_settings")[args.dst]["settings"]["index"]
        print("  translog:             ", s.get("translog"))
        h = fetch(cli, f"/_cluster/health/{args.dst}")
        print("  health:               ", h.get("status"),
              "shards", h.get("active_shards"))


if __name__ == "__main__":
    main()
