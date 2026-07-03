#!/usr/bin/env sh
set -eu

host=dopey
control_url=
join_token=dev-token
node_name=dopey
remote_path=/tmp/jetsonfabric-agent
install_path=/usr/local/bin/jetsonfabric-agent
skip_build=
smoke_test=

usage() {
  cat <<USAGE
usage: $0 [--host USER@HOST] [--control-url URL] [--join-token TOKEN] [--node-name NAME] [--remote-path PATH] [--install-path PATH] [--skip-build] [--smoke-test]

Build and deploy the Linux arm64 JetsonFabric agent to a Jetson over SSH.

Examples:
  sh scripts/deploy-agent.sh --host samuel@dopey.local
  sh scripts/deploy-agent.sh --host dopey --control-url http://192.168.1.50:52415 --smoke-test
USAGE
}

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      host=$2
      shift 2
      ;;
    --control-url)
      control_url=$2
      shift 2
      ;;
    --join-token)
      join_token=$2
      shift 2
      ;;
    --node-name)
      node_name=$2
      shift 2
      ;;
    --remote-path)
      remote_path=$2
      shift 2
      ;;
    --install-path)
      install_path=$2
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --smoke-test)
      smoke_test=1
      shift
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

if [ -z "$host" ]; then
  printf '--host is required\n' >&2
  exit 2
fi

if [ -n "$smoke_test" ] && [ -z "$control_url" ]; then
  printf '--control-url is required with --smoke-test\n' >&2
  exit 2
fi

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
artifact="$repo_root/dist/jetsonfabric-agent-linux-arm64"

if [ -z "$skip_build" ]; then
  printf 'building JetsonFabric artifacts...\n'
  (cd "$repo_root" && sh scripts/build.sh)
fi

if [ ! -f "$artifact" ]; then
  printf 'missing agent artifact: %s\n' "$artifact" >&2
  printf 'run: sh scripts/build.sh\n' >&2
  exit 1
fi

printf 'copying %s to %s:%s...\n' "$artifact" "$host" "$remote_path"
scp "$artifact" "$host:$remote_path"

q_remote_path=$(shell_quote "$remote_path")
q_install_path=$(shell_quote "$install_path")

printf 'installing agent on %s as %s...\n' "$host" "$install_path"
ssh -t "$host" "sudo install -m 0755 $q_remote_path $q_install_path && $q_install_path --help >/dev/null"

if [ -n "$smoke_test" ]; then
  q_control_url=$(shell_quote "$control_url")
  q_join_token=$(shell_quote "$join_token")
  q_node_name=$(shell_quote "$node_name")
  printf 'running one-shot heartbeat smoke test from %s as node %s...\n' "$host" "$node_name"
  ssh "$host" "$q_install_path --control-url $q_control_url --join-token $q_join_token --node-name $q_node_name --once"
fi

printf 'deploy complete: %s -> %s:%s\n' "$artifact" "$host" "$install_path"
