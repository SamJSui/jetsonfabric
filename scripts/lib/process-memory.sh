#!/usr/bin/env bash

runtime_pid_for_port() {
  local port=$1
  ss -H -ltnp "sport = :$port" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

process_memory_json() {
  local pid=$1 model_path=$2
  local rollup="/proc/$pid/smaps_rollup"
  local mappings="/proc/$pid/smaps"
  [[ -r "$rollup" && -r "$mappings" ]] || {
    echo "cannot read process memory for pid $pid" >&2
    return 1
  }

  local rss_kb pss_kb model_rss_kb
  rss_kb="$(awk '$1 == "Rss:" { print $2 }' "$rollup")"
  pss_kb="$(awk '$1 == "Pss:" { print $2 }' "$rollup")"
  model_rss_kb="$(awk -v model="$model_path" '
    /^[[:xdigit:]]+-[[:xdigit:]]+[[:space:]]/ {
      in_model = index($0, model) > 0
      next
    }
    in_model && $1 == "Rss:" { rss += $2 }
    END { print rss + 0 }
  ' "$mappings")"

  jq -nc \
    --argjson pid "$pid" \
    --argjson rss_bytes "$((rss_kb * 1024))" \
    --argjson pss_bytes "$((pss_kb * 1024))" \
    --argjson model_mapping_rss_bytes "$((model_rss_kb * 1024))" \
    '{
      pid: $pid,
      rss_bytes: $rss_bytes,
      pss_bytes: $pss_bytes,
      model_mapping_rss_bytes: $model_mapping_rss_bytes
    }'
}

runtime_memory_for_port() {
  local port=$1 model_path=$2
  local pid
  pid="$(runtime_pid_for_port "$port")"
  [[ -n "$pid" ]] || {
    echo "could not identify runtime process listening on port $port" >&2
    return 1
  }
  process_memory_json "$pid" "$model_path"
}
