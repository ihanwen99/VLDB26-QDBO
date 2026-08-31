#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/cpp_embed.cpp"
OUT="$SCRIPT_DIR/cpp_embed_lib.so"

if command -v clang++ >/dev/null 2>&1; then
  CXX=clang++
elif command -v g++ >/dev/null 2>&1; then
  CXX=g++
else
  echo "Neither clang++ nor g++ found." >&2
  exit 1
fi

"$CXX" -O3 -std=c++17 -shared -fPIC -o "$OUT" "$SRC"

echo "Built $OUT"
