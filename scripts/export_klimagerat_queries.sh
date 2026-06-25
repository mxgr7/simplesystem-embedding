#!/usr/bin/env bash
# Export every search query (with hit count + add-to-cart count) for all sessions
# that searched "klimagerät" in the last 30 days. Data streams PostHog -> file via jq.
set -euo pipefail

ENV_FILE="$HOME/simplesystem-embedding/.env"
PROJECT_ID=74110
BASE_URL="https://eu.posthog.com"
OUT="${1:-$HOME/klimagerat_queries_cart.csv}"

API_KEY="$(grep -E '^POSTHOG_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
[ -n "$API_KEY" ] || { echo "POSTHOG_API_KEY not found in $ENV_FILE" >&2; exit 1; }

read -r -d '' SQL <<'HOGQL' || true
WITH target AS (
  SELECT DISTINCT properties.`$session_id` AS sid
  FROM events
  WHERE event='search_performed'
    AND lower(toString(properties.queryTerm))='klimagerät'
    AND properties.`$session_id` IS NOT NULL
    AND timestamp >= now()-INTERVAL 30 DAY
),
searches AS (
  SELECT properties.`$session_id` AS sid, properties.queryId AS qid,
         any(properties.queryTerm) AS term, min(timestamp) AS ts,
         max(toInt(toString(properties.searchResults.hitCount))) AS hits,
         any(toString(properties.`$feature/search-experiment`)) AS variant,
         count() AS n
  FROM events
  WHERE event='search_performed'
    AND properties.`$session_id` IN (SELECT sid FROM target)
    AND properties.queryId IS NOT NULL
    AND timestamp >= now()-INTERVAL 30 DAY
  GROUP BY sid, qid
),
carts AS (
  SELECT properties.queryId AS qid, count() AS cart_adds
  FROM events
  WHERE event='added_to_cart' AND properties.queryId IS NOT NULL
    AND timestamp >= now()-INTERVAL 30 DAY
  GROUP BY qid
)
SELECT s.sid AS session,
       formatDateTime(s.ts,'%Y-%m-%d %H:%i:%S') AS time,
       s.variant AS variant,
       s.term AS query,
       s.hits AS hitCount,
       s.n AS searches,
       coalesce(c.cart_adds,0) AS cart_adds
FROM searches s
LEFT JOIN carts c ON s.qid=c.qid
ORDER BY s.sid, s.ts
LIMIT 100000
HOGQL

# Build request body with jq so the SQL is JSON-escaped safely.
BODY="$(jq -n --arg q "$SQL" '{query:{kind:"HogQLQuery", query:$q}}')"

echo "Querying PostHog (project $PROJECT_ID)..." >&2
curl -sS --fail-with-body "$BASE_URL/api/projects/$PROJECT_ID/query/" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY" \
| jq -r '["session","time","variant","query","hitCount","searches","cart_adds"],
         (.results[] | [.[0], .[1], (.[2] // ""), (.[3] // "(no term)"), .[4], .[5], .[6]])
         | @csv' > "$OUT"

ROWS=$(( $(wc -l < "$OUT") - 1 ))
echo "Wrote $ROWS rows to $OUT" >&2
