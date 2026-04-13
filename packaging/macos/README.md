# macOS packaging

This directory contains the source-controlled macOS packaging entry point for `eov`.

The script builds a `.app` bundle, copies the checked-in macOS icon asset, bundles non-system dynamic libraries, ad-hoc signs the bundle, and produces a zipped release artifact.

## Requirements

- macOS
- Xcode command line tools (`codesign`, `install_name_tool`, `otool`)
- Homebrew `openslide` installed, or `OPENSLIDE_LIB_DIR` set
- Rust toolchain with Cargo

## Usage

Build a zipped `.app` bundle:

```bash
./packaging/macos/build.sh
```

Build for a specific target triple:

```bash
RUST_TARGET=aarch64-apple-darwin ./packaging/macos/build.sh
RUST_TARGET=x86_64-apple-darwin ./packaging/macos/build.sh
```

Useful environment overrides:

- `VERSION`
- `RUST_TARGET`
- `ARCH_LABEL`
- `ARCHIVE_BASENAME`
- `DIST_DIR`
- `ICNS_SOURCE`
- `CODESIGN_IDENTITY`
