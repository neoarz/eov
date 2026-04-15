#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_SRC="$REPO_ROOT/example_plugin"
PLUGIN_DEST="$HOME/.eov/plugins/example_plugin"
PYTHON_PLUGIN_SRC="$REPO_ROOT/example_plugin_python"
PYTHON_PLUGIN_DEST="$HOME/.eov/plugins/example_plugin_python"

echo "Building eov and example plugin..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml"

echo "Installing Rust example plugin to $PLUGIN_DEST..."
rm -rf "$PLUGIN_DEST"
mkdir -p "$PLUGIN_DEST"
cp "$PLUGIN_SRC/plugin.toml" "$PLUGIN_DEST/"
cp -r "$PLUGIN_SRC/ui" "$PLUGIN_DEST/"
cp "$REPO_ROOT/target/debug/libexample_plugin.so" "$PLUGIN_DEST/"

echo "Installing Python example plugin to $PYTHON_PLUGIN_DEST..."
rm -rf "$PYTHON_PLUGIN_DEST"
mkdir -p "$PYTHON_PLUGIN_DEST"
cp "$PYTHON_PLUGIN_SRC/plugin.toml" "$PYTHON_PLUGIN_DEST/"
cp "$PYTHON_PLUGIN_SRC/plugin.py" "$PYTHON_PLUGIN_DEST/"
cp -r "$PYTHON_PLUGIN_SRC/ui" "$PYTHON_PLUGIN_DEST/"

# Create a venv inside the plugin directory and install slint into it.
if [ ! -d "$PYTHON_PLUGIN_DEST/.venv" ]; then
    echo "Creating Python venv for plugin..."
    python3 -m venv "$PYTHON_PLUGIN_DEST/.venv"
    "$PYTHON_PLUGIN_DEST/.venv/bin/pip" install --quiet slint
fi

echo "Launching eov..."
exec "$REPO_ROOT/target/debug/eov" "$@"
