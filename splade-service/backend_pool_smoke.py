import json
import urllib.request

from constants import FIELD_ORDER


BASE = "http://127.0.0.1:8137"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    secondary = call("POST", "/admin/backends", {
        "url": "http://127.0.0.1:8139",
        "weight": 10,
        "max_concurrency": 1,
        "max_client_batch": 8,
    })
    secondary_id = secondary["id"]
    try:
        call("PATCH", "/admin/backends/b1", {"weight": 0})
        row = {name: "" for name in FIELD_ORDER}
        row["name"] = "Secondary backend routing probe unique 2026-07-25"
        value = "\x00".join(row[name] for name in FIELD_ORDER)
        vectors = call("POST", "/embed", {"inputs": value})
        assert len(vectors) == 1 and 0 < len(vectors[0]) <= 256
        print(json.dumps({
            "secondary": secondary,
            "nnz": len(vectors[0]),
            "pool": call("GET", "/admin/backends"),
        }, indent=2))
    finally:
        call("PATCH", "/admin/backends/b1", {"weight": 1})
        call("DELETE", f"/admin/backends/{secondary_id}")


if __name__ == "__main__":
    main()
