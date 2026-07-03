#!/usr/bin/env sh
set -eu

prefix=${JETSONFABRIC_PREFIX:-/opt/jetsonfabric}
etc_dir=${JETSONFABRIC_ETC:-/etc/jetsonfabric}
state_dir=${JETSONFABRIC_STATE:-/var/lib/jetsonfabric}

sudo mkdir -p "$etc_dir" \
  "$state_dir/models" \
  "$state_dir/plans" \
  "$state_dir/benchmarks" \
  "$state_dir/logs"

sudo chown -R "${USER:-$(id -un)}:${USER:-$(id -un)}" "$state_dir"

printf 'JetsonFabric node layout ready:\n'
printf '  repo:       %s\n' "$prefix"
printf '  config:     %s\n' "$etc_dir"
printf '  state:      %s\n' "$state_dir"
printf '\nExpected private files:\n'
printf '  %s/node.env\n' "$etc_dir"
printf '  %s/runtime-assignment.json\n' "$etc_dir"