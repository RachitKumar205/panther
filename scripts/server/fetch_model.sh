#!/usr/bin/env bash
# Download one model at its pinned Hugging Face revision. Needs the network, so it runs as a CPU job:
#
#   deploy/server/panther.sh job fetch-1.5b scripts/server/fetch_model.sh Qwen2.5-1.5B-Instruct
set -euo pipefail

MODEL="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${PANTHER_DATA:-$ROOT/data}"
PY="${PANTHER_PY:?run inside the Panther image}"

read -r REPO REV < <("$PY" - "$MODEL" <<'EOF'
import sys
from panther.models.inventory import load_registry
m = next(m for m in load_registry()["models"] if m["name"] == sys.argv[1])
print(m["repo"], m["revision"])
EOF
)

HF_DIR="$DATA/hf/$MODEL"
"$PANTHER_HF" download "$REPO" --revision "$REV" --local-dir "$HF_DIR"
echo "$REPO@$REV" > "$HF_DIR/.panther-revision"
echo "fetched $REPO@$REV -> $HF_DIR"
