#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/../..
source .venv/bin/activate
python3 packaging/flatpak/flatpak-builder-tools/cargo/flatpak-cargo-generator.py \
  Cargo.lock \
  -o packaging/flatpak/cargo-sources.json
rm -rf .flatpak-builder build
flatpak-builder --force-clean build packaging/flatpak/io.eosin.eov.yml