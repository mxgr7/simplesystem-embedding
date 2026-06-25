#!/usr/bin/env bash
# Extract all distinct article IDs that appeared as search hits (searchResults.items[].id)
# during the currently running "Semantic Search PoC" experiment (flag search-experiment).
# IDs stream PostHog -> file via jq; they never pass through the model context.
set -euo pipefail

ENV_FILE="$HOME/simplesystem-embedding/.env"
PROJECT_ID=74110
BASE_URL="https://eu.posthog.com"
OUT="${1:-$HOME/experiment_hit_article_ids.csv}"
# Experiment start (running, no end). Overridable: ARG2 = ISO start.
START="${2:-2026-06-22 08:44:42}"

API_KEY="$(grep -E '^POSTHOG_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
[ -n "$API_KEY" ] || { echo "POSTHOG_API_KEY not found in $ENV_FILE" >&2; exit 1; }

# The query API caps each response at 50000 rows and forbids OFFSET with a
# personal API key, so paginate via keyset on the grouped article_id key:
# each page fetches rows with article_id > last-seen id.
PAGE=50000
echo '"article_id","hit_occurrences"' > "$OUT"
cursor=""
page_no=0
echo "Querying PostHog (project $PROJECT_ID, experiment since $START)..." >&2
while : ; do
  # escape single quotes in cursor for safe SQL embedding
  esc="${cursor//\'/\'\'}"
  read -r -d '' SQL <<HOGQL || true
SELECT
  JSONExtractString(arrayJoin(JSONExtractArrayRaw(ifNull(toString(properties.searchResults), '{}'), 'items')), 'id') AS article_id,
  count() AS hit_occurrences
FROM events
WHERE event = 'search_performed'
  AND properties.\`\$feature/search-experiment\` IN ('control','test2')
  AND timestamp >= toDateTime('$START')
GROUP BY article_id
HAVING article_id > '$esc'
ORDER BY article_id
LIMIT $PAGE
HOGQL

  BODY="$(jq -n --arg q "$SQL" '{query:{kind:"HogQLQuery", query:$q}}')"
  RESP="$(curl -sS --fail-with-body --max-time 600 "$BASE_URL/api/projects/$PROJECT_ID/query/" \
    -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$BODY")"

  n="$(printf '%s' "$RESP" | jq -r '.results | length')"
  [ "$n" = "0" ] && break
  printf '%s' "$RESP" | jq -r '.results[] | [.[0], .[1]] | @csv' >> "$OUT"
  cursor="$(printf '%s' "$RESP" | jq -r '.results[-1][0]')"
  page_no=$((page_no+1))
  echo "  page $page_no: +$n rows (cursor ...${cursor: -12})" >&2
  [ "$n" -lt "$PAGE" ] && break
done

ROWS=$(( $(wc -l < "$OUT") - 1 ))
echo "Wrote $ROWS distinct article IDs to $OUT" >&2
