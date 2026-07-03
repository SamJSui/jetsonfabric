#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
build_dir="$repo_root/runtime/build"
dist_dir="$repo_root/dist"

cmake_bin=${CMAKE:-cmake}

mkdir -p "$dist_dir"

"$cmake_bin" -S "$repo_root/runtime" -B "$build_dir"
"$cmake_bin" --build "$build_dir"

cp "$build_dir/jetsonfabric-runtime-worker" "$dist_dir/jetsonfabric-runtime-worker"
printf 'Built runtime artifact at %s\n' "$dist_dir/jetsonfabric-runtime-worker"
