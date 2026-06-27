#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
go_bin=${GO:-go}
dist_dir="$repo_root/dist"
go_cache="$repo_root/.cache/go-build"

mkdir -p "$dist_dir" "$go_cache"
export GOCACHE="$go_cache"

run_go() {
  "$go_bin" "$@"
}

build_target() {
  target_os=$1
  target_arch=$2
  output_name=$3
  package_path=$4

  GOOS=$target_os GOARCH=$target_arch run_go build -buildvcs=false -o "$dist_dir/$output_name" "$package_path"
}

run_go version
run_go test ./...

build_target linux amd64 jetsonfabric-control-linux-amd64 ./cmd/jetsonfabric-control
build_target linux amd64 jetsonfabric-agent-linux-amd64 ./cmd/jetsonfabric-agent
build_target linux amd64 jetsonfabric-bench-linux-amd64 ./cmd/jetsonfabric-bench
build_target linux arm64 jetsonfabric-control-linux-arm64 ./cmd/jetsonfabric-control
build_target linux arm64 jetsonfabric-agent-linux-arm64 ./cmd/jetsonfabric-agent
build_target linux arm64 jetsonfabric-bench-linux-arm64 ./cmd/jetsonfabric-bench

printf 'Built JetsonFabric artifacts in %s\n' "$dist_dir"
