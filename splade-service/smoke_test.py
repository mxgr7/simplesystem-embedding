import json
import time
import urllib.request

from constants import FIELD_ORDER


def request(url, value):
    body = json.dumps({"inputs": value}, ensure_ascii=False).encode()
    call = urllib.request.Request(
        f"{url.rstrip('/')}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(call, timeout=300) as response:
        result = json.load(response)[0]
    return result, time.perf_counter() - started


def main():
    row = {name: "" for name in FIELD_ORDER}
    row.update({
        "name": "Kühlschrank Pro",
        "manufacturer_name": "Müller",
        "ean": "4000123456789",
        "features_text": "Größe: 60 cm.",
    })
    value = "\x00".join(row[name] for name in FIELD_ORDER)
    first, first_seconds = request("http://127.0.0.1:8137", value)
    second, second_seconds = request("http://127.0.0.1:8137", value)
    assert first == second
    assert 0 < len(first) <= 256
    print(json.dumps({
        "nnz": len(first),
        "first_seconds": round(first_seconds, 4),
        "cached_seconds": round(second_seconds, 4),
        "max_weight": max(first.values()),
        "sample": dict(list(first.items())[:8]),
    }, indent=2))


if __name__ == "__main__":
    main()
