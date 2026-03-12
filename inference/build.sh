#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Build the Stylift WASM inference engine
# Requires: Emscripten SDK (emcc) — https://emscripten.org
#
# Usage:
#   ./build.sh           # release build  → ../demo/stylift.js + .wasm
#   ./build.sh --debug   # debug build    → adds -g, disables -O3
# ─────────────────────────────────────────────────────────────
set -euo pipefail

OUT_DIR="../demo"
OUT_NAME="stylift"

OPT_FLAGS="-O3 -msimd128"
if [[ "${1:-}" == "--debug" ]]; then
    OPT_FLAGS="-O0 -g"
    echo "→ Debug build"
else
    echo "→ Release build"
fi

mkdir -p "$OUT_DIR"

emcc \
    attention.cpp \
    mlp.cpp       \
    forward.cpp   \
    generate.cpp  \
    bindings.cpp  \
    \
    $OPT_FLAGS                          \
    -std=c++17                          \
    \
    -s WASM=1                           \
    -s MODULARIZE=1                     \
    -s EXPORT_NAME="StyliftModule"      \
    -s ALLOW_MEMORY_GROWTH=1            \
    -s INITIAL_MEMORY=67108864          \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString"]' \
    -s EXPORTED_FUNCTIONS='[           \
        "_wasm_model_load",            \
        "_wasm_prefill",               \
        "_wasm_decode_step",           \
        "_wasm_generate",              \
        "_wasm_seq_free",              \
        "_malloc",                     \
        "_free"                        \
    ]'                                  \
    \
    -o "$OUT_DIR/$OUT_NAME.js"

echo "✓ Built → $OUT_DIR/$OUT_NAME.js + $OUT_DIR/$OUT_NAME.wasm"