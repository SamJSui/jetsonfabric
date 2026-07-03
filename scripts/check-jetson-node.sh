#!/usr/bin/env sh
set -eu

remote_host=
expected_hostname=${EXPECTED_HOSTNAME:-dopey}

usage() {
  cat <<USAGE
usage: $0 [--host USER@HOST] [--expected-hostname NAME]

Run a compact Jetson readiness check locally, or over SSH with --host.
USAGE
}

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      remote_host=$2
      shift 2
      ;;
    --expected-hostname)
      expected_hostname=$2
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "$remote_host" ]; then
  q_expected=$(shell_quote "$expected_hostname")
  exec ssh "$remote_host" "EXPECTED_HOSTNAME=$q_expected sh -s" < "$0"
fi

section() {
  printf '\n== %s ==\n' "$1"
}

run_optional() {
  label=$1
  shift
  printf '\n-- %s --\n' "$label"
  if "$@"; then
    return 0
  fi
  printf 'WARN: %s failed or unavailable\n' "$label" >&2
}

command_status() {
  name=$1
  if command -v "$name" >/dev/null 2>&1; then
    printf 'OK   %-18s %s\n' "$name" "$(command -v "$name")"
  else
    printf 'MISS %-18s\n' "$name"
  fi
}

section 'identity'
printf 'hostname: %s\n' "$(hostname)"
if [ "$(hostname)" = "$expected_hostname" ]; then
  printf 'hostname_check: OK expected %s\n' "$expected_hostname"
else
  printf 'hostname_check: WARN expected %s\n' "$expected_hostname"
fi
printf 'kernel: %s\n' "$(uname -a)"
printf 'arch: %s\n' "$(uname -m)"

section 'os and jetson release'
if [ -f /etc/os-release ]; then
  . /etc/os-release
  printf 'os: %s\n' "${PRETTY_NAME:-unknown}"
fi
if [ -f /etc/nv_tegra_release ]; then
  printf 'nv_tegra_release: '
  cat /etc/nv_tegra_release
else
  printf 'WARN: /etc/nv_tegra_release not found\n'
fi
run_optional 'nvidia jetpack package' sh -c 'dpkg-query -W nvidia-jetpack 2>/dev/null || true'

section 'hardware and power'
if [ -r /proc/device-tree/model ]; then
  printf 'model: %s\n' "$(tr -d '\000' < /proc/device-tree/model)"
fi
run_optional 'nvpmodel' sh -c 'command -v nvpmodel >/dev/null 2>&1 && nvpmodel -q'
run_optional 'jetson_clocks availability' sh -c 'command -v jetson_clocks >/dev/null 2>&1 && jetson_clocks --show'

section 'memory and disk'
run_optional 'memory' free -h
run_optional 'root disk' df -h /
run_optional 'block devices' lsblk

section 'network'
run_optional 'ip addresses' ip -brief addr
run_optional 'default route' sh -c 'ip route | grep default || true'
run_optional 'dns lookup' sh -c 'getent hosts github.com || true'

section 'runtime commands'
command_status tegrastats
command_status nvcc
command_status trtexec
command_status docker
command_status llama-server
command_status llama-cli
command_status ollama

section 'cuda and tensorrt hints'
run_optional 'cuda version' sh -c 'nvcc --version 2>/dev/null || true'
run_optional 'cuda path' sh -c 'ls -ld /usr/local/cuda* 2>/dev/null || true'
run_optional 'tensorrt packages' sh -c 'dpkg-query -W "*tensorrt*" 2>/dev/null | head -20 || true'

section 'services'
run_optional 'ssh service' sh -c 'systemctl is-active ssh 2>/dev/null || systemctl is-active sshd 2>/dev/null || true'
run_optional 'avahi service' sh -c 'systemctl is-active avahi-daemon 2>/dev/null || true'
run_optional 'docker service' sh -c 'systemctl is-active docker 2>/dev/null || true'

section 'tegrastats sample'
if command -v tegrastats >/dev/null 2>&1; then
  if command -v timeout >/dev/null 2>&1; then
    timeout 3s tegrastats || true
  else
    tegrastats | sed -n '1,3p'
  fi
else
  printf 'MISS tegrastats\n'
fi

command_status docker
command_status docker-compose
run_optional 'docker compose version' sh -c 'docker compose version 2>/dev/null || true'
run_optional 'nvidia container runtime' sh -c 'command -v nvidia-ctk >/dev/null 2>&1 && nvidia-ctk --version || true'
run_optional 'jetsonfabric dirs' sh -c 'ls -ld /etc/jetsonfabric /var/lib/jetsonfabric 2>/dev/null || true'

section 'summary'
printf 'Node readiness check complete for expected hostname %s. Review WARN/MISS lines before running JetsonFabric POC.\n' "$expected_hostname"
