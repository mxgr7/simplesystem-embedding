"""Dump all `tei:v2:<hash>` keys from KVRocks to a sorted uint128 numpy
binary file. Snapshot is used by scripts/prewarm_stream.py --cached-snapshot
to skip the pipelined EXISTS round-trip for any hash already present at
snapshot time. Hashes added after the snapshot fall through to the regular
EXISTS path, so the snapshot is safe to use even mid-write.

Storage layout: dtype=[('h','<u8'),('l','<u8')] — first 16 bytes of the
32-char-hex content hash split into two big-endian u64s, then sorted
lexicographically. Lets workers do a vectorised np.searchsorted per chunk.

Run (on the box hosting KVRocks; no tunnel needed there):
    uv run python scripts/dump_kvrocks_keys.py \\
        --redis-host localhost --redis-port 6666 \\
        --out /data/kvrocks_keys.bin
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import redis


_KEY_DT = np.dtype([("h", "<u8"), ("l", "<u8")])
_PREFIX = b"tei:v2:"
_PLEN = len(_PREFIX)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6666)
    ap.add_argument("--out", required=True,
                    help="binary file path; raw uint128 array (h,l u64 pair)")
    ap.add_argument("--scan-count", type=int, default=20000,
                    help="SCAN COUNT hint per round-trip.")
    args = ap.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=False)
    total = r.dbsize()
    print(f"KVRocks DBSIZE: {total:,}", flush=True)
    # Allocate generously; trimmed at the end.
    cap = int(total * 1.05) + 10_000
    arr = np.empty(cap, dtype=_KEY_DT)
    idx = 0
    cursor = 0
    t0 = time.time()
    last_log = t0

    while True:
        cursor, keys = r.scan(cursor, match=_PREFIX + b"*",
                              count=args.scan_count)
        if keys:
            n = len(keys)
            if idx + n > cap:
                cap = max(cap * 2, idx + n + 10_000)
                arr = np.resize(arr, cap)
            for i, k in enumerate(keys):
                if not k.startswith(_PREFIX):
                    continue
                hx = k[_PLEN:]
                if len(hx) != 32:
                    continue
                try:
                    arr[idx]["h"] = int(hx[:16], 16)
                    arr[idx]["l"] = int(hx[16:32], 16)
                    idx += 1
                except (ValueError, TypeError):
                    pass
        now = time.time()
        if now - last_log >= 10.0:
            rate = idx / max(now - t0, 1e-3)
            print(f"  {idx:,} keys scanned ({rate:,.0f}/s), "
                  f"{now-t0:.0f}s elapsed", flush=True)
            last_log = now
        if cursor == 0:
            break

    arr = arr[:idx]
    print(f"sorting {idx:,} keys ...", flush=True)
    arr.sort()
    # Drop duplicates (shouldn't happen but cheap to ensure)
    uniq_mask = np.empty(len(arr), dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = arr[1:] != arr[:-1]
    arr = arr[uniq_mask]
    arr.tofile(args.out)
    el = time.time() - t0
    print(f"\nDONE: wrote {len(arr):,} unique keys "
          f"({arr.nbytes/1e9:.2f} GB) to {args.out} in {el:.0f}s "
          f"({len(arr)/max(el,1e-3):,.0f}/s)")


if __name__ == "__main__":
    main()
