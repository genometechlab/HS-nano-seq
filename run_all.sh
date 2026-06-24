#!/usr/bin/env bash
#
# One-shot reproduction:
#   1. download + build reference + align   (data_preparation.sh)
#   2. trim signals -> HDF5                  (signal_trimming.py)   all handles
#   3. inference -> prediction pickle        (model_inference.py)   all handles
#   4. confusion-matrix figures              (replicates results.ipynb)
#
# Stages run sequentially: stage 2 completes for every handle before stage 3 begins.
# Confusion-matrix PNGs are written to the project root.
#
# Usage:  bash run_all.sh [options]        (or: sbatch run_all.sh [options])
#         bash run_all.sh --help

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults (match the per-script defaults). Override via CLI flags below.
# --------------------------------------------------------------------------- #
THREADS=32          # data prep (samtools / minimap2)
BATCH_SIZE=512      # inference
NUM_WORKERS=4       # inference DataLoader workers
FORCE_CPU=0         # inference: run on CPU even if a GPU is present
TNRE=12             # trimming: NRE-side nt kept
TRE=12              # trimming: RE-side nt kept
MIN_LEN=18          # trimming: min trimmed signal length
SKIP_PREP=0         # skip stage 1 (reuse already-downloaded/aligned data)
SKIP_TRIM=0         # skip stage 2 (reuse existing .h5 signal files)
SKIP_INFER=0        # skip stage 3 (reuse existing prediction pickles)

usage() {
    cat <<'USAGE'
Usage: bash run_all.sh [options]

  --threads N        threads for data prep (samtools / minimap2)   [default 32]
  --batch-size N     inference batch size                          [default 512]
  --num-workers N    inference DataLoader workers                  [default 4]
  --force-cpu        run inference on CPU even if a GPU exists      [default off]
  --tnre N           NRE-side adapter nt kept (trimming)           [default 12]
  --tre N            RE-side adapter nt kept (trimming)            [default 12]
  --min-len N        minimum trimmed signal length (trimming)      [default 18]
  --skip-prep        skip stage 1 (reuse downloaded/aligned data)
  --skip-trimming    skip stage 2 (reuse existing .h5 signal files)
  --skip-inference   skip stage 3 (reuse existing prediction pickles)
  -h, --help         show this help and exit
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)        THREADS="$2";     shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";  shift 2 ;;
        --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
        --force-cpu)      FORCE_CPU=1;       shift ;;
        --tnre)           TNRE="$2";        shift 2 ;;
        --tre)            TRE="$2";         shift 2 ;;
        --min-len)        MIN_LEN="$2";     shift 2 ;;
        --skip-prep)      SKIP_PREP=1;       shift ;;
        --skip-trimming)  SKIP_TRIM=1;       shift ;;
        --skip-inference) SKIP_INFER=1;      shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Config -- edit handles / paths here if yours differ.
# (The plotting block at the bottom has its own copy of the handles; keep both
#  in sync.)
# --------------------------------------------------------------------------- #
CHECKPOINT="Data/synthetics/model_weights/synthetics_classifier.pth"
SIGNALS_DIR="Data/synthetics/signals"
PREDS_DIR="Data/results"

HANDLES=(
    "9fea1263-540e-4cc6-ad49-9e80276dca2b"
    "266e1025-91a2-48d9-a168-6386828065f6"
)

# per-handle POD5 directory
declare -A POD5_DIR=(
    ["9fea1263-540e-4cc6-ad49-9e80276dca2b"]="Data/synthetics/03_10_26_NBD114_TT_PH009_fig004_BC1_16/20260310_1620_1G_PBG61569_9fea1263"
    ["266e1025-91a2-48d9-a168-6386828065f6"]="Data/synthetics/03_10_26_NBD114_TT_PH009_fig004_BC1_16/20260310_1512_1G_PBG61569_266e1025"
)

# inference flags assembled from the options above
INFER_FLAGS=(--batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
if [[ "$FORCE_CPU" -eq 1 ]]; then
    INFER_FLAGS+=(--force-cpu)
fi

echo "config: threads=$THREADS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS \
force_cpu=$FORCE_CPU tnre=$TNRE tre=$TRE min_len=$MIN_LEN \
skip_prep=$SKIP_PREP skip_trim=$SKIP_TRIM skip_infer=$SKIP_INFER"

# --------------------------------------------------------------------------- #
# 1. Data preparation
# --------------------------------------------------------------------------- #
if [[ "$SKIP_PREP" -eq 1 ]]; then
    echo "==> [1/4] data preparation (skipped)"
else
    echo "==> [1/4] data preparation"
    THREADS="$THREADS" bash data_preparation.sh
fi

# --------------------------------------------------------------------------- #
# 2. Signal trimming -- every handle
# --------------------------------------------------------------------------- #
if [[ "$SKIP_TRIM" -eq 1 ]]; then
    echo "==> [2/4] trimming (skipped)"
else
    mkdir -p "$SIGNALS_DIR"
    for h in "${HANDLES[@]}"; do
        echo "==> [2/4] trimming   ($h)"
        python signal_trimming.py \
            --bam-dir  "Data/synthetics/aligned_bams/$h" \
            --pod5-dir "${POD5_DIR[$h]}" \
            --out      "$SIGNALS_DIR/$h.h5" \
            --tnre "$TNRE" \
            --tre "$TRE" \
            --min-len "$MIN_LEN"
    done
fi

# --------------------------------------------------------------------------- #
# 3. Inference -- every handle
# --------------------------------------------------------------------------- #
if [[ "$SKIP_INFER" -eq 1 ]]; then
    echo "==> [3/4] inference (skipped)"
else
    mkdir -p "$PREDS_DIR"
    for h in "${HANDLES[@]}"; do
        echo "==> [3/4] inference  ($h)"
        python model_inference.py \
            --h5            "$SIGNALS_DIR/$h.h5" \
            --model-weights "$CHECKPOINT" \
            --out           "$PREDS_DIR/$h.pkl" \
            "${INFER_FLAGS[@]}"
    done
fi

# --------------------------------------------------------------------------- #
# 4. Confusion-matrix figures (replicates results.ipynb), saved to project root
# --------------------------------------------------------------------------- #
echo "==> [4/4] confusion-matrix figures"
python - <<'PY'
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: write files, never open a window
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SPLIT_RIDS = "Data/synthetics/split_dict.pkl"
PREDS_DIR  = "Data/results"
NUM_CLASSES = 16
labels = list(range(NUM_CLASSES))
tick_labels = [f"BC{c+1:02d}" for c in labels]   # 0->BC01 ... 15->BC16

# (handle, splits)
#   training experiment -> train/val/test panels from split_dict
#   held-out experiment -> a single 'test' panel over all its reads
EXPERIMENTS = [
    ("9fea1263-540e-4cc6-ad49-9e80276dca2b", ["train", "val", "test"]),
    ("266e1025-91a2-48d9-a168-6386828065f6", None),
]


def draw(ax, gt, pr, name, missing):
    acc = (gt == pr).mean() if len(gt) else float("nan")
    cm = confusion_matrix(gt, pr, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    recall = np.divide(cm, row_sums, where=row_sums != 0)
    recall[row_sums[:, 0] == 0] = 0

    im = ax.imshow(recall, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels); ax.set_yticks(labels)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("pred"); ax.set_ylabel("gt")
    ax.set_title(f"{name}  n={len(gt)}  acc={acc:.3f}" +
                 (f"  (missing {missing})" if missing else ""))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{recall[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7,
                    color="white" if recall[i, j] > 0.5 else "black")
    return im


split_rids = None
try:
    with open(SPLIT_RIDS, "rb") as fh:
        split_rids = pickle.load(fh)
except FileNotFoundError:
    print(f"[warn] {SPLIT_RIDS} not found; experiments fall back to all reads")

for handle, splits in EXPERIMENTS:
    with open(f"{PREDS_DIR}/{handle}.pkl", "rb") as fh:
        preds = pickle.load(fh)

    panels = splits if (splits and split_rids) else ["test"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 13 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, name in zip(axes, panels):
        gts, prs, missing = [], [], 0
        rids = split_rids[name] if (splits and split_rids) else preds.keys()
        for rid in rids:
            rec = preds.get(rid)
            if rec is None:
                missing += 1
                continue
            gts.append(rec["gt"]); prs.append(rec["pred"])
        im = draw(ax, np.array(gts), np.array(prs), name, missing)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = f"confusion_matrix_{handle}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
PY

echo "==> done. Confusion-matrix figures written to the project root."