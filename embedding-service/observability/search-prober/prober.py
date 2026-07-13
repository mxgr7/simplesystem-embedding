#!/usr/bin/env python3
"""Synthetic search prober: health-check + cold-start warm-keeper for the semantic
search stack. Every PROBE_INTERVAL seconds it fires ONE real semantic query at the
target (preprod gateway end-to-end), rotating query string + company context so it
touches a representative spread of the hot vectors (keeps the ES page cache warm)
while also proving the whole path (gateway -> proxy -> query service -> TEI -> ES) is up.

Exposes Prometheus metrics on PROBE_PORT/metrics:
  probe_up{target}                          1 if last probe HTTP 200 else 0
  probe_search_latency_seconds{target}      histogram of client-side latency
  probe_requests_total{target,outcome}      success/error counter
  probe_hit_count{target}                   last response hitCount
  probe_last_status{target}                 last HTTP status code
  probe_mode_requests_total{target,mode}    by predicted routing mode

stdlib only (no pip). Reads contexts from PROBE_MAP (the company_context_map.json).
"""
import json, os, ssl, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

URL      = os.environ.get("PROBE_URL", "https://preprod-api.platform.simplesystem.com").rstrip("/")
AUTH     = os.environ.get("PROBE_AUTH", "")          # "user:password"
INTERVAL = float(os.environ.get("PROBE_INTERVAL", "10"))
PORT     = int(os.environ.get("PROBE_PORT", "9155"))
MAP      = os.environ.get("PROBE_MAP", "/app/company_context_map.json")
PROFILE  = os.environ.get("PROBE_PROFILE", "TEST_PROFILE_18")
TARGET   = os.environ.get("PROBE_TARGET_LABEL", "preprod")
TIMEOUT  = float(os.environ.get("PROBE_TIMEOUT", "30"))
SEARCH_PATH = "/article-features/search?page=1&pageSize=10"

# Varied multiword / non-id queries -> route to VECTOR_ONLY / HYBRID_RRF (exercise the
# vector path, which is what warms the bf16 vectors + is the real-user latency).
QUERIES = [
    "kaffeevollautomat für das büro", "edelstahl arbeitsplatte mit becken",
    "nitril handschuhe puderfrei", "monitor 27 zoll wqhd", "akku bohrschrauber set",
    "drehmomentschlüssel 1/2 zoll", "absturzsicherung dachhaken edelstahl",
    "thermopapierrolle bisphenol frei", "schweißerschutzvorhang grün",
    "laborwaschflasche ldpe spritzflasche", "kabelbinder uv beständig schwarz",
    "gehörschutzstöpsel snr 36db", "pneumatik steckverschraubung 8mm",
    "industriereiniger entfetter kanister", "magnetwinkel schweißen verstellbar",
    "warnschutzlatzhose orange", "edelstahlrohr v2a geschliffen",
    "sicherheitsschuh s3 esd", "gewindeschneider hss-g m6", "fettpresse pneumatisch",
]
BUCKETS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]

_lock = threading.Lock()
_m = {
    "up": 0, "last_status": 0, "hit_count": 0,
    "req": {"success": 0, "error": 0},
    "mode": {}, "buckets": {b: 0 for b in BUCKETS}, "inf": 0, "sum": 0.0, "count": 0,
    # server-side time from the X-Search-Ms response header (v0.0.5+): lets dashboards
    # split client-observed latency into server search vs gateway/network/serialization.
    "srv_buckets": {b: 0 for b in BUCKETS}, "srv_inf": 0, "srv_sum": 0.0, "srv_count": 0,
}

def _ssl_ctx():
    try:
        return ssl.create_default_context()
    except Exception:
        c = ssl.create_default_context(); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE; return c
CTX = _ssl_ctx()

def load_contexts():
    m = json.load(open(MAP))["companies"]
    out = []
    for cid, s in m.items():
        sas = s.get("selectedArticleSources", s)
        clean = {k: v for k, v in sas.items() if not k.startswith("_")}
        if clean.get("catalogVersionIdsOrderedByPreference"):
            out.append(clean)
    return out

def predict_mode(q):
    qs = q.strip()
    if " " in qs:
        return "VECTOR_ONLY"
    return "LEXICAL_ONLY" if qs.replace("-", "").isdigit() else "HYBRID_RRF"

def record(latency_s, status, ok, hit, mode, server_s=None):
    with _lock:
        _m["up"] = 1 if ok else 0
        _m["last_status"] = status
        _m["req"]["success" if ok else "error"] += 1
        _m["mode"][mode] = _m["mode"].get(mode, 0) + 1
        if ok:
            _m["hit_count"] = hit
            _m["sum"] += latency_s; _m["count"] += 1
            for b in BUCKETS:
                if latency_s <= b:
                    _m["buckets"][b] += 1
            _m["inf"] += 1
            if server_s is not None:
                _m["srv_sum"] += server_s; _m["srv_count"] += 1
                for b in BUCKETS:
                    if server_s <= b:
                        _m["srv_buckets"][b] += 1
                _m["srv_inf"] += 1

def probe_loop(contexts):
    hdr = {"Content-Type": "application/json"}
    if AUTH:
        import base64
        hdr["Authorization"] = "Basic " + base64.b64encode(AUTH.encode()).decode()
    i = 0
    while True:
        q = QUERIES[i % len(QUERIES)]
        ctx = contexts[i % len(contexts)]
        mode = predict_mode(q)
        body = {"searchMode": "HITS_ONLY", "searchArticlesBy": PROFILE,
                "selectedArticleSources": ctx, "queryString": q, "currency": "EUR",
                "coreSortimentOnly": False, "closedMarketplaceOnly": False,
                "maxDeliveryTime": 0, "explain": False, "summaries": []}
        req = urllib.request.Request(URL + SEARCH_PATH, data=json.dumps(body).encode(), headers=hdr)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
                dt = time.perf_counter() - t0
                data = json.load(r)
                hit = (data.get("metadata") or {}).get("hitCount", 0)
                srv = r.headers.get("X-Search-Ms")
                record(dt, r.status, True, hit, mode,
                       float(srv) / 1000.0 if srv else None)
        except urllib.error.HTTPError as e:
            record(time.perf_counter() - t0, e.code, False, 0, mode)
        except Exception:
            record(time.perf_counter() - t0, 0, False, 0, mode)
        i += 1
        time.sleep(INTERVAL)

def render():
    with _lock:
        L = f'target="{TARGET}"'
        out = []
        out += ["# TYPE probe_up gauge", f"probe_up{{{L}}} {_m['up']}"]
        out += ["# TYPE probe_last_status gauge", f"probe_last_status{{{L}}} {_m['last_status']}"]
        out += ["# TYPE probe_hit_count gauge", f"probe_hit_count{{{L}}} {_m['hit_count']}"]
        out += ["# TYPE probe_requests_total counter"]
        for o, v in _m["req"].items():
            out.append(f'probe_requests_total{{{L},outcome="{o}"}} {v}')
        out += ["# TYPE probe_mode_requests_total counter"]
        for mo, v in _m["mode"].items():
            out.append(f'probe_mode_requests_total{{{L},mode="{mo}"}} {v}')
        out += ["# TYPE probe_search_latency_seconds histogram"]
        for b in BUCKETS:
            out.append(f'probe_search_latency_seconds_bucket{{{L},le="{b}"}} {_m["buckets"][b]}')
        out.append(f'probe_search_latency_seconds_bucket{{{L},le="+Inf"}} {_m["inf"]}')
        out.append(f'probe_search_latency_seconds_sum{{{L}}} {_m["sum"]}')
        out.append(f'probe_search_latency_seconds_count{{{L}}} {_m["count"]}')
        out += ["# TYPE probe_search_server_seconds histogram"]
        for b in BUCKETS:
            out.append(f'probe_search_server_seconds_bucket{{{L},le="{b}"}} {_m["srv_buckets"][b]}')
        out.append(f'probe_search_server_seconds_bucket{{{L},le="+Inf"}} {_m["srv_inf"]}')
        out.append(f'probe_search_server_seconds_sum{{{L}}} {_m["srv_sum"]}')
        out.append(f'probe_search_server_seconds_count{{{L}}} {_m["srv_count"]}')
        return ("\n".join(out) + "\n").encode()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/metrics"):
            b = render(); self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok\n")
    def log_message(self, *a):  # silence access log
        pass

if __name__ == "__main__":
    contexts = load_contexts()
    print(f"prober: target={URL} interval={INTERVAL}s contexts={len(contexts)} queries={len(QUERIES)} profile={PROFILE}", flush=True)
    threading.Thread(target=probe_loop, args=(contexts,), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
