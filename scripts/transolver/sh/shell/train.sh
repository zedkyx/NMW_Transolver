#!/bin/bash
# 统一训练入口: bash scripts/transolver/sh/shell/train.sh [--dry_run] [--start_from ID] [--ddp]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${TRANSOLVER_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CONFIG="${SCRIPT_DIR}/../../configs/train.yaml"

source "${CONDA_BASE:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-upt}"

cd "$REPO_ROOT"
export TRANSOLVER_ROOT="$REPO_ROOT"
exec python "${SCRIPT_DIR}/../../tools/launch/train_launcher.py" --config "$CONFIG" "$@"
