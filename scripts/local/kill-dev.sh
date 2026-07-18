#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NODE_PORT="${JF_NODE0_PORT:-19180}"
RUNTIME_PORT="${JF_RUNTIME_PORT:-19190}"
WORK_DIR="${JF_DEV_WORK_DIR:-$ROOT_DIR/.cache/jetsonfabric/dev}"
if [[ "$WORK_DIR" != /* ]]; then
  WORK_DIR="$ROOT_DIR/$WORK_DIR"
fi
NODE_PID_FILE="$WORK_DIR/node.pid"
RUNTIME_PID_FILE="$WORK_DIR/runtime.pid"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}
for command in pgrep ps seq ss grep; do
  require_command "$command"
done

pid_is_running() {
  local pid=${1:-}
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

cmdline_for_pid() {
  local pid=$1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline"
}

valid_node_pid() {
  local pid=$1 cmdline
  pid_is_running "$pid" || return 1
  cmdline="$(cmdline_for_pid "$pid")" || return 1
  [[ "$cmdline" == *"jetsonfabric-node"* &&
     "$cmdline" == *"--node-name jf-dev"* &&
     "$cmdline" == *"--listen 127.0.0.1:$NODE_PORT"* &&
     "$cmdline" == *"--data-dir $WORK_DIR/node"* ]]
}

valid_runtime_pid() {
  local pid=$1 cmdline
  pid_is_running "$pid" || return 1
  cmdline="$(cmdline_for_pid "$pid")" || return 1
  [[ "$cmdline" == *"jetsonfabric-runtime-worker"* &&
     "$cmdline" == *"--node-name jf-dev"* &&
     "$cmdline" == *"--listen 127.0.0.1:$RUNTIME_PORT"* ]]
}

read_recorded_pid() {
  local file=$1
  if [[ -f "$file" ]]; then
    tr -d '[:space:]' <"$file"
  fi
}

find_matching_node() {
  local pid
  for pid in $(pgrep -f 'jetsonfabric-node' 2>/dev/null || true); do
    if valid_node_pid "$pid"; then
      printf '%s\n' "$pid"
      return 0
    fi
  done
  return 1
}

find_matching_runtime() {
  local pid
  for pid in $(pgrep -f 'jetsonfabric-runtime-worker' 2>/dev/null || true); do
    if valid_runtime_pid "$pid"; then
      printf '%s\n' "$pid"
      return 0
    fi
  done
  return 1
}

NODE_PID="$(read_recorded_pid "$NODE_PID_FILE")"
RUNTIME_PID="$(read_recorded_pid "$RUNTIME_PID_FILE")"
if ! valid_node_pid "$NODE_PID"; then
  NODE_PID="$(find_matching_node || true)"
fi
if ! valid_runtime_pid "$RUNTIME_PID"; then
  RUNTIME_PID="$(find_matching_runtime || true)"
fi

if [[ -z "$NODE_PID" && -z "$RUNTIME_PID" ]]; then
  rm -f "$NODE_PID_FILE" "$RUNTIME_PID_FILE"
  echo "No active JetsonFabric development process was found."
  for port_spec in "node:$NODE_PORT" "runtime:$RUNTIME_PORT"; do
    label=${port_spec%%:*}
    port=${port_spec##*:}
    if ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .; then
      echo "Warning: $label port $port is occupied by an untracked process:" >&2
      ss -H -ltnp "sport = :$port" >&2 2>/dev/null || true
    fi
  done
  exit 0
fi

if [[ -n "$NODE_PID" ]]; then
  echo "Stopping JetsonFabric dev node PID $NODE_PID"
  node_pgid="$(ps -o pgid= -p "$NODE_PID" 2>/dev/null | tr -d ' ' || true)"
  if [[ "$node_pgid" == "$NODE_PID" ]]; then
    kill -TERM -- "-$NODE_PID" 2>/dev/null || true
  else
    kill -TERM "$NODE_PID" 2>/dev/null || true
  fi
fi
if [[ -n "$RUNTIME_PID" ]] && pid_is_running "$RUNTIME_PID"; then
  echo "Stopping JetsonFabric dev runtime PID $RUNTIME_PID"
  kill -TERM "$RUNTIME_PID" 2>/dev/null || true
fi

for _ in $(seq 1 50); do
  if ! pid_is_running "$NODE_PID" && ! pid_is_running "$RUNTIME_PID"; then
    break
  fi
  sleep 0.2
done

if pid_is_running "$NODE_PID"; then
  echo "Dev node did not exit after SIGTERM; sending SIGKILL" >&2
  node_pgid="$(ps -o pgid= -p "$NODE_PID" 2>/dev/null | tr -d ' ' || true)"
  if [[ "$node_pgid" == "$NODE_PID" ]]; then
    kill -KILL -- "-$NODE_PID" 2>/dev/null || true
  else
    kill -KILL "$NODE_PID" 2>/dev/null || true
  fi
fi
if pid_is_running "$RUNTIME_PID"; then
  echo "Dev runtime did not exit after SIGTERM; sending SIGKILL" >&2
  kill -KILL "$RUNTIME_PID" 2>/dev/null || true
fi

rm -f "$NODE_PID_FILE" "$RUNTIME_PID_FILE"

after_failure=0
for port_spec in "node:$NODE_PORT" "runtime:$RUNTIME_PORT"; do
  label=${port_spec%%:*}
  port=${port_spec##*:}
  if ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .; then
    echo "Warning: $label port $port remains occupied:" >&2
    ss -H -ltnp "sport = :$port" >&2 2>/dev/null || true
    after_failure=1
  fi
done

if [[ $after_failure -ne 0 ]]; then
  exit 1
fi

echo "JetsonFabric development processes stopped."
