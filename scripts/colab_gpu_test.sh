#!/usr/bin/env bash
# Build + validate the modern_fm CUDA backend on a Google Colab GPU VM.
#
# Automates docs/cuda_validation_runbook.md: the dev machine (macOS) has no
# NVIDIA GPU and CI has no GPU runner, so `backend="cuda"` is validated here —
# provision a GPU VM via the Colab CLI, upload the committed working tree,
# build the extension with `--features cuda-backend` (rustup + maturin on the
# VM), run tests/test_cuda_parity.py + the full suite + benchmarks/bench_cuda.py,
# pull back a markdown report, then tear the VM down.
# (Pattern borrowed from repleafgbm-bench/scripts/colab_gpu_test.sh.)
#
# Requires the Colab CLI:  uv tool install google-colab-cli
#
# Usage:
#   bash scripts/colab_gpu_test.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
#                                  [--full-bench]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="modernfm-gpu"
KEEP=0
FULL_BENCH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --full-bench) FULL_BENCH=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if ! command -v colab >/dev/null 2>&1; then
    echo "error: the 'colab' CLI is not installed (uv tool install google-colab-cli)" >&2
    exit 1
fi
if ! git diff --quiet HEAD; then
    echo "error: tracked files differ from HEAD — commit/stash first ('git archive HEAD' would omit them)." >&2
    git status --short >&2
    exit 1
fi

DATE="$(date +%F)"
REPORT_OUT="/tmp/${DATE}-modernfm-cuda-validation.md"
TARBALL="$(mktemp -t modernfm-XXXXXX).tar.gz"
cleanup_local() { rm -f "$TARBALL"; }
trap cleanup_local EXIT

echo ">> archiving committed tree (git archive HEAD) -> $TARBALL"
git archive --format=tar.gz -o "$TARBALL" HEAD

echo ">> provisioning $GPU VM (session: $SESSION)"
colab new -s "$SESSION" --gpu "$GPU"

stop_vm() { [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap 'cleanup_local; stop_vm' EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/modern_fm.tar.gz

echo ">> building + validating on the GPU (rust build takes a few minutes)"
MODERNFM_FULL_BENCH="$FULL_BENCH" colab exec -s "$SESSION" --timeout 3600 -f scripts/colab_remote_test.py

echo ">> downloading report -> $REPORT_OUT"
colab download -s "$SESSION" /content/modernfm_gpu_report.md "$REPORT_OUT"

echo ">> done. report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
