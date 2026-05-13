"""Create local-article-index-v3: full-precision fp32 HNSW alongside v2 (int8_hnsw).

Cloned from v2:
  - all settings (shards, analysis, translog tuning)
  - full mapping including _source.excludes:[embeddings.vector],
    embeddings.inputHash, embeddingModelVersion

Changed vs v2:
  - embeddings.vector.index_options.type: int8_hnsw -> hnsw (fp32 storage)

Does NOT touch v2. Pass --confirm-delete to drop an existing v3.

The data reindex (v2 -> v3) is a separate script — vectors come from the
Redis cache `tei:v2:<hash>` (original fp32 returned by TEI), not from v2's
int8-quantized doc-values, so we don't lose precision in the upgrade.
"""

from __future__ import annotations

import argparse
import copy
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
    ap.add_argument("--src", default="local-article-index-v2")
    ap.add_argument("--dst", default="local-article-index-v3")
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

        mappings = copy.deepcopy(src_mapping)
        vec = mappings["properties"]["embeddings"]["properties"]["vector"]
        prev = vec.get("index_options", {}).get("type")
        vec.setdefault("index_options", {})["type"] = "hnsw"

        body = {"settings": {"index": idx_settings}, "mappings": mappings}
        print(
            f"creating {args.dst} (shards={idx_settings.get('number_of_shards')}, "
            f"replicas={idx_settings.get('number_of_replicas')}, "
            f"refresh={idx_settings.get('refresh_interval')}, "
            f"vector codec: {prev} -> hnsw) ...",
            flush=True,
        )
        r = cli.put(f"/{args.dst}", json=body)
        if r.status_code != 200:
            print("ERROR:", r.text, file=sys.stderr)
            r.raise_for_status()
        print("  acknowledged:", r.json().get("acknowledged"))

        m = fetch(cli, f"/{args.dst}/_mapping")[args.dst]["mappings"]
        print("\nverification:")
        print("  _source.excludes:", m.get("_source", {}).get("excludes"))
        print("  embeddings.inputHash:",
              m["properties"]["embeddings"]["properties"].get("inputHash"))
        print("  embeddings.vector.index_options:",
              m["properties"]["embeddings"]["properties"]["vector"].get("index_options"))
        print("  embeddingModelVersion:",
              m["properties"].get("embeddingModelVersion"))


if __name__ == "__main__":
    main()
