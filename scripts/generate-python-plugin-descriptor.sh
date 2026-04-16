#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

protoc \
  -I proto \
  --include_imports \
  --descriptor_set_out=plugin_api/python/eov_extension.desc \
  proto/eov_extension.proto

printf '%s\n' "Wrote plugin_api/python/eov_extension.desc"