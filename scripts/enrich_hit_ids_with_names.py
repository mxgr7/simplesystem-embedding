# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb", "requests"]
# ///
"""
Retrieve article names from Elasticsearch for every article_id in
experiment_hit_article_ids.csv and join them back into the file.

Design goals:
- Do NOT overload the production ES instance:
  * single worker (sequential), one connection
  * batched _mget (default 1000 ids/request) so it's few requests, not 787k
  * a polite sleep between requests + retry/backoff, and it honors HTTP 429
- Resumable: names are written incrementally to a side file; a re-run skips
  ids already resolved.
- DuckDB does the final join (ids+counts) x (names) -> output CSV.

Run:  uv run enrich_hit_ids_with_names.py
Test: uv run enrich_hit_ids_with_names.py --limit 2000
"""
import argparse, csv, os, sys, time
from pathlib import Path
import requests
import duckdb

HOME = Path.home()
ENV_FILE = HOME / "simplesystem-embedding" / ".env"
INDEX = "prod-article-index-v1-semantic-bf16"

def load_env(path):
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def extract_name(src):
    if not src:
        return ""
    offers = src.get("offers") or []
    for o in offers:
        n = (o or {}).get("name")
        if n:
            return n
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(HOME / "experiment_hit_article_ids.csv"))
    ap.add_argument("--names", default=str(HOME / "experiment_hit_id_names.csv"),
                    help="incremental id->name side file (resume source)")
    ap.add_argument("--out", default=str(HOME / "experiment_hit_article_ids_named.csv"))
    ap.add_argument("--batch", type=int, default=1000, help="ids per _mget request")
    ap.add_argument("--sleep", type=float, default=0.15, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0, help="only first N ids (testing)")
    args = ap.parse_args()

    env = load_env(ENV_FILE)
    es_url = env["ES_SEMANTIC2_URL"].rstrip("/")
    auth = (env["ES_SEMANTIC2_USERNAME"], env["ES_SEMANTIC2_PASSWORD"])
    mget_url = f"{es_url}/{INDEX}/_mget"
    params = {"_source": "offers.name"}

    # ---- read the id list via duckdb ----
    con = duckdb.connect()
    q = f"SELECT article_id FROM read_csv_auto('{args.infile}', header=true)"
    if args.limit:
        q += f" LIMIT {args.limit}"
    ids = [r[0] for r in con.execute(q).fetchall()]
    total = len(ids)
    print(f"[info] {total} ids to resolve from {args.infile}", file=sys.stderr)

    # ---- resume: skip ids already in names file ----
    done = set()
    names_path = Path(args.names)
    write_header = True
    if names_path.exists():
        with names_path.open() as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if row:
                    done.add(row[0])
        write_header = False
        print(f"[info] resume: {len(done)} ids already resolved, skipping them", file=sys.stderr)

    todo = [i for i in ids if i not in done]
    print(f"[info] {len(todo)} ids remaining", file=sys.stderr)

    session = requests.Session()
    session.auth = auth
    session.headers.update({"Content-Type": "application/json"})

    found_n = missing_n = 0
    t0 = time.time()
    with names_path.open("a", newline="") as out:
        w = csv.writer(out)
        if write_header:
            w.writerow(["article_id", "name", "found"])
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            # polite retry/backoff loop
            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = session.post(mget_url, params=params,
                                        json={"ids": chunk}, timeout=args.timeout)
                    if resp.status_code == 429:
                        wait = min(60, 2 ** attempt)
                        print(f"[warn] 429 throttled, sleeping {wait}s", file=sys.stderr)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt >= 6:
                        print(f"[error] giving up on batch at {start}: {e}", file=sys.stderr)
                        raise
                    wait = min(60, 2 ** attempt)
                    print(f"[warn] {e} -> retry in {wait}s", file=sys.stderr)
                    time.sleep(wait)

            for doc in resp.json().get("docs", []):
                fnd = bool(doc.get("found"))
                nm = extract_name(doc.get("_source")) if fnd else ""
                w.writerow([doc.get("_id"), nm, int(fnd)])
                if fnd:
                    found_n += 1
                else:
                    missing_n += 1

            out.flush()
            n_done = start + len(chunk)
            if (start // args.batch) % 20 == 0 or n_done >= len(todo):
                rate = n_done / max(1e-9, time.time() - t0)
                eta = (len(todo) - n_done) / max(1e-9, rate)
                print(f"[prog] {n_done}/{len(todo)}  found={found_n} missing={missing_n}"
                      f"  {rate:.0f} ids/s  eta {eta/60:.1f}m", file=sys.stderr)
            time.sleep(args.sleep)

    print(f"[info] fetch complete: found={found_n} missing={missing_n}", file=sys.stderr)

    # ---- final join via duckdb ----
    print(f"[info] joining names into {args.out} ...", file=sys.stderr)
    con.execute(f"""
        COPY (
          SELECT i.article_id,
                 i.hit_occurrences,
                 n.name,
                 COALESCE(n.found, 0) AS found
          FROM read_csv_auto('{args.infile}', header=true) i
          LEFT JOIN read_csv_auto('{args.names}', header=true) n
            ON i.article_id = n.article_id
          ORDER BY i.hit_occurrences DESC, i.article_id
        ) TO '{args.out}' (HEADER, DELIMITER ',');
    """)
    nrows = con.execute(f"SELECT count(*) FROM read_csv_auto('{args.out}', header=true)").fetchone()[0]
    print(f"[done] wrote {nrows} rows to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
