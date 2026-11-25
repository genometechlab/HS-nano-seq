#!/usr/bin/env bash
#
# One-shot reproduction, for every run in experiments.sh:
#   Stage 1. download + build reference + align   (1_data_preparation.sh)
#   Stage 2. trim signals -> HDF5                  (2_signal_trimming.py)
#   Stage 3. inference -> prediction pickle        (3_model_inference.py)
#   Stage 4. confusion-matrix figures -> figures/  (4_results.py)
#
# Stages run sequentially: a stage completes for every run before the next begins.
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
TNRE=12             # trimming: NRE-side nt kept (the trained model needs 12)
TRE=12              # trimming: RE-side nt kept  (the trained model needs 12)
MIN_LEN=18          # trimming: min trimmed signal length
TRIM_WORKERS=16     # trimming: worker processes
SKIP_PREP=0         # skip stage 1 (reuse already-downloaded/aligned data)
SKIP_TRIM=0         # skip stage 2 (reuse existing .h5 signal files)
SKIP_INFER=0        # skip stage 3 (reuse existing prediction pickles)

usage() {
    cat <<'USAGE'
Usage: bash run_all.sh [options]

  --only "KEYS"       only these <exp>/<run> keys, e.g. "PH009/266e1025 PH015/78073ab4"
                      (default: every run in experiments.sh)
  --data PATH         data root                                      [default ./Data]
  --threads N         threads for data prep (samtools / minimap2)    [default 32]
  --tnre N            NRE-side adapter nt kept (trimming)            [default 12]
  --tre N             RE-side adapter nt kept (trimming)             [default 12]
                      NOTE: the trained model only supports --tnre 12 --tre 12
  --min-len N         minimum trimmed signal length (trimming)       [default 18]
  --trim-workers N    trimming worker processes                      [default 16]
  --batch-size N      inference batch size                           [default 512]
  --num-workers N     inference DataLoader workers                   [default 4]
  --force-cpu         run inference on CPU even if a GPU exists       [default off]
  --skip-prep         skip stage 1 (reuse downloaded/aligned data)
  --skip-trimming     skip stage 2 (reuse existing .h5 signal files)
  --skip-inference    skip stage 3 (reuse existing prediction pickles)
  -h, --help          show this help and exit

Stage 4 (figures) always runs. To only list the downloads, use
  bash 1_data_preparation.sh --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)           export ONLY="$2"; shift 2 ;;
        --data)           export DATA="$2"; shift 2 ;;
        --threads)        THREADS="$2";      shift 2 ;;
        --tnre)           TNRE="$2";         shift 2 ;;
        --tre)            TRE="$2";          shift 2 ;;
        --min-len)        MIN_LEN="$2";      shift 2 ;;
        --trim-workers)   TRIM_WORKERS="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";   shift 2 ;;
        --num-workers)    NUM_WORKERS="$2";  shift 2 ;;
        --force-cpu)      FORCE_CPU=1;        shift ;;
        --skip-prep)      SKIP_PREP=1;        shift ;;
        --skip-trimming)  SKIP_TRIM=1;        shift ;;
        --skip-inference) SKIP_INFER=1;       shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# A dry run would stop data prep after listing downloads, and stage 2 would then
# run on missing BAMs.
if [[ "${DRYRUN:-0}" == 1 ]]; then
    echo "DRYRUN is not supported by run_all.sh; use: bash 1_data_preparation.sh --dry-run" >&2
    exit 1
fi

if [[ "$TNRE" != 12 || "$TRE" != 12 ]]; then
    echo "WARNING: --tnre $TNRE --tre $TRE: the trained model only supports 12/12;" \
         "its predictions on these signals are not meaningful." >&2
fi

# --------------------------------------------------------------------------- #
# Config: experiments, runs and paths all come from experiments.sh.
# --------------------------------------------------------------------------- #
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # sbatch runs a spooled copy of the script
source ./experiments.sh

# inference flags assembled from the options above
INFER_FLAGS=(--batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
if [[ "$FORCE_CPU" -eq 1 ]]; then
    INFER_FLAGS+=(--force-cpu)
fi

# <exp>/<run> for every run, e.g. PH009/9fea1263, and its first barcode
RUN_KEYS=()
declare -A RUN_FIRST_BC
for line in "${RUNS[@]}"; do
    parse_run "$line"
    RUN_KEYS+=("$EXP/$RUN")
    RUN_FIRST_BC["$EXP/$RUN"]="$FIRST_BC"
done

echo "config: threads=$THREADS tnre=$TNRE tre=$TRE min_len=$MIN_LEN \
trim_workers=$TRIM_WORKERS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS \
force_cpu=$FORCE_CPU skip_prep=$SKIP_PREP skip_trim=$SKIP_TRIM skip_infer=$SKIP_INFER"
echo "data:   $DATA"
echo "runs:   ${RUN_KEYS[*]}"

# --------------------------------------------------------------------------- #
# Stage 1. Data preparation
# --------------------------------------------------------------------------- #
if [[ "$SKIP_PREP" -eq 1 ]]; then
    echo "==> Stage 1/4: data preparation (skipped)"
else
    echo "==> Stage 1/4: data preparation"
    bash 1_data_preparation.sh --threads "$THREADS" --data "$DATA" \
        ${ONLY:+--only "$ONLY"}
fi

# --------------------------------------------------------------------------- #
# Stage 2. Signal trimming -- every run
# --------------------------------------------------------------------------- #
if [[ "$SKIP_TRIM" -eq 1 ]]; then
    echo "==> Stage 2/4: signal trimming (skipped)"
else
    for key in "${RUN_KEYS[@]}"; do
        echo "==> Stage 2/4: signal trimming ($key)"
        python 2_signal_trimming.py \
            --bam-dir   "$PROCESSED/$key/aligned" \
            --pod5-dir  "$RAW/$key/pod5" \
            --out       "$PROCESSED/$key/signals.h5" \
            --first-bc  "${RUN_FIRST_BC[$key]}" \
            --num-classes "$N_BARCODES" \
            --tnre "$TNRE" \
            --tre "$TRE" \
            --min-len "$MIN_LEN" \
            --workers "$TRIM_WORKERS"
    done
fi

# --------------------------------------------------------------------------- #
# Stage 3. Inference -- every run
# --------------------------------------------------------------------------- #
if [[ "$SKIP_INFER" -eq 1 ]]; then
    echo "==> Stage 3/4: inference (skipped)"
else
    for key in "${RUN_KEYS[@]}"; do
        echo "==> Stage 3/4: inference ($key)"
        python 3_model_inference.py \
            --h5            "$PROCESSED/$key/signals.h5" \
            --model-weights "$CHECKPOINT" \
            --out           "$PROCESSED/$key/preds.pkl" \
            "${INFER_FLAGS[@]}"
    done
fi

# --------------------------------------------------------------------------- #
# Stage 4. Confusion-matrix figures
# --------------------------------------------------------------------------- #
echo "==> Stage 4/4: confusion-matrix figures"
python 4_results.py --data "$DATA" --out-dir figures

echo "==> done. Figures in figures/"
