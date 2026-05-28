"""Dump all `tei:v2:<hash>` keys from KVRocks to a raw 16-byte-per-key
binary file. Consumed by scripts/prewarm_stream.py --cached-snapshot,
which sorts the array in-process on load (the snapshot box is typically
CPU-constrained; sorting happens on the consumer side where there's
more CPU).

Storage: contiguous 16-byte blocks (the decoded SHA-256[:32] hex),
UNSORTED. Consumer is responsible for sorting before searchsorted.

Implementation notes:
- Uses hiredis (C parser) for the SCAN responses — Python-side per-key
  overhead is otherwise the dominant cost. ~10x faster than stdlib.
- Streams 16-byte binary directly via bytes.fromhex (no int parsing).
- Buffers ~1 MB before write to avoid syscall storms.

Run (on the box hosting KVRocks; no SSH tunnel needed there):
    uv run python scripts/dump_kvrocks_keys.py \\
        --redis-host localhost --redis-port 6666 \\
        --out /data/datasets/kvrocks_keys.bin
"""
from __future__ import annotations

import argparse
import os
import time

import redis


_PREFIX = b"tei:v2:"
_PLEN = len(_PREFIX)
_FLUSH_BYTES = 1 << 20  # 1 MB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6666)
    ap.add_argument("--out", required=True,
                    help="binary file path; raw uint128 array")
    ap.add_argument("--scan-count", type=int, default=50000,
                    help="SCAN COUNT hint per round-trip.")
    args = ap.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=False)
    total = r.dbsize()
    print(f"KVRocks DBSIZE: {total:,}", flush=True)

    tmp_path = args.out + ".tmp"
    buf = bytearray()
    n = 0
    cursor = 0
    t0 = time.time()
    last_log = t0

    with open(tmp_path, "wb") as fout:
        while True:
            cursor, keys = r.scan(cursor, match=_PREFIX + b"*",
                                  count=args.scan_count)
            for k in keys:
                if len(k) == _PLEN + 32 and k[:_PLEN] == _PREFIX:
                    try:
                        buf += bytes.fromhex(k[_PLEN:].decode("ascii"))
                        n += 1
                    except ValueError:
                        pass
            if len(buf) >= _FLUSH_BYTES:
                fout.write(buf)
                buf.clear()
            now = time.time()
            if now - last_log >= 5.0:
                rate = n / max(now - t0, 1e-3)
                print(f"  {n:,} keys ({rate:,.0f}/s) {now-t0:.0f}s elapsed",
                      flush=True)
                last_log = now
            if cursor == 0:
                break
        if buf:
            fout.write(buf)
    os.rename(tmp_path, args.out)

    el = time.time() - t0
    size = os.path.getsize(args.out)
    print(f"\nDONE: {n:,} keys in {el:.0f}s ({n/max(el,1e-3):,.0f}/s) "
          f"-> {args.out} ({size/1e9:.2f} GB, UNSORTED)", flush=True)


if __name__ == "__main__":
    main()
