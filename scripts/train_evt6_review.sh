#!/usr/bin/env bash
# Backward-compatible alias. The canonical paper command is maintained once.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train_paper_evt6.sh" "$@"
