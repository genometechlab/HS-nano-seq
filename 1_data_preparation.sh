#!/usr/bin/env bash
#
# End-to-end synthetic-read pipeline, for every run listed in experiments.sh:
#   1. download POD5 + MinKNOW reports + the 16 basecalled barcode BAMs per run,
#      plus the trained model and split_dict, from the public S3 bucket
#   2. write + index the DNA_HS_DNA reference (.fa)
#   3. align the 16 barcodes of each run (preserving tags) -> sorted, indexed BAMs
#
# Downloads are include-only: anything in the bucket not listed here (fast5/,
# PH016, the merged run-level BAMs, unused barcodes, ...) is never fetched.
#
# Requires: awscli, samtools, minimap2 on PATH.
#
# Usage:  bash 1_data_preparation.sh [options]     (bash 1_data_preparation.sh --help)
# Each option can also be given as an environment variable (THREADS, DATA, DRYRUN,
# ONLY); a flag overrides the variable.

set -euo pipefail

THREADS="${THREADS:-32}"
DRYRUN="${DRYRUN:-0}"

usage() {
    cat <<'USAGE'
Usage: bash 1_data_preparation.sh [options]

  --only "KEYS"   only these <exp>/<run> keys, e.g. "PH009/266e1025"  [default: all runs]
  --data PATH     data root                                            [default ./Data]
  --threads N     threads for samtools / minimap2                      [default 32]
  --dry-run       list what would be downloaded, then stop (nothing is downloaded)
  -h, --help      show this help and exit
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)     export ONLY="$2"; shift 2 ;;
        --data)     export DATA="$2"; shift 2 ;;
        --threads)  THREADS="$2";     shift 2 ;;
        --dry-run)  DRYRUN=1;          shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # sbatch runs a spooled copy of the script
source ./experiments.sh

# --size-only: never re-download a file already present at the right size (raw
# data is immutable; mtimes of moved/copied files would otherwise trigger re-syncs)
AWS_FLAGS=(--no-sign-request --size-only --only-show-errors)
if [[ "$DRYRUN" -eq 1 ]]; then
    AWS_FLAGS=(--no-sign-request --size-only --dryrun)
fi

# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
echo "  -> prep 1/3: download from S3$([[ "$DRYRUN" -eq 1 ]] && echo " (dry run)")"

aws s3 sync "$S3_BASE/$S3_MODEL_WEIGHTS/" "$MODEL_DIR/" "${AWS_FLAGS[@]}"
aws s3 sync "$S3_BASE/$(dirname "$S3_SPLIT_DICT")/" "$MODEL_DIR/" "${AWS_FLAGS[@]}" \
    --exclude "*" --include "$(basename "$S3_SPLIT_DICT")"

for line in "${RUNS[@]}"; do
    parse_run "$line"
    run_dir="$RAW/$EXP/$RUN"
    echo "       $EXP/$RUN"

    # MinKNOW run folder (pod5/ + reports), minus the redundant fast5/ copy
    aws s3 sync "$S3_BASE/$S3_RUN/" "$run_dir/" "${AWS_FLAGS[@]}" \
        --exclude "fast5/*"

    # this run's 16 barcode BAMs only
    includes=()
    for ((i = 0; i < N_BARCODES; i++)); do
        includes+=(--include "${HANDLE}_*_barcode$(printf "%02d" $((FIRST_BC + i))).bam")
    done
    aws s3 sync "$S3_BASE/$S3_BAM/" "$run_dir/basecalled/" "${AWS_FLAGS[@]}" \
        --exclude "*" "${includes[@]}"
done

if [[ "$DRYRUN" -eq 1 ]]; then
    echo "  -> dry run: stopping before reference + alignment"
    exit 0
fi

# --------------------------------------------------------------------------- #
# 2. Reference (.fa) + index
# --------------------------------------------------------------------------- #
echo "  -> prep 2/3: write + index reference"
mkdir -p "$REF_DIR"
cat > "$REF" << 'FASTA'
>DNA_HS_DNA
TACGAGCGGTTGAGAAAGCTACGCCCGCGAATTACAGAGCGCACGGTTTTATTCTTACTTCTTGCCTCGGTGCGAACTGTATGAACTACTACTCCAGTGCTACGAGTTGCCGAAGAACTGA
FASTA
samtools faidx "$REF"          # creates DNA_HS_DNA.fa.fai
# (minimap2 builds its index on the fly for a reference this small; no .mmi needed)

# --------------------------------------------------------------------------- #
# 3. Align the 16 barcodes of each run
# --------------------------------------------------------------------------- #
echo "  -> prep 3/3: align barcodes per run (threads=$THREADS)"

for line in "${RUNS[@]}"; do
    parse_run "$line"
    in_dir="$RAW/$EXP/$RUN/basecalled"
    out_dir="$PROCESSED/$EXP/$RUN/aligned"
    mkdir -p "$out_dir"
    echo "       $EXP/$RUN  ->  $out_dir"

    for ((i = 0; i < N_BARCODES; i++)); do
        bc=$(printf "%02d" $((FIRST_BC + i)))

        # exactly one input BAM per barcode; every run must have all 16, so a
        # missing or ambiguous one fails loudly
        matches=("$in_dir/${HANDLE}"_*_barcode"${bc}".bam)
        if [[ ${#matches[@]} -ne 1 || ! -f "${matches[0]}" ]]; then
            echo "  !! $EXP/$RUN barcode${bc}: expected 1 BAM in $in_dir, found ${#matches[@]}" >&2
            exit 1
        fi
        in="${matches[0]}"
        out="$out_dir/bc${bc}.bam"     # class $i

        # the index is written last, so its presence means the BAM is complete
        if [[ -f "$out.bai" ]]; then
            echo "         barcode${bc} (class $i): already aligned, skipping"
            continue
        fi

        echo "         barcode${bc} (class $i): $(basename "$in") -> $(basename "$out")"

        samtools fastq -@ "$THREADS" -T'*' "$in" \
            | minimap2 -ax map-ont -t "$THREADS" -y "$REF" - \
            | samtools view -@ "$THREADS" -h -F 4 -b - \
            | samtools sort -@ "$THREADS" -o "$out" -

        samtools index "$out"
    done
done

echo "  -> prep done. Aligned BAMs in $PROCESSED/<exp>/<run>/aligned/"
