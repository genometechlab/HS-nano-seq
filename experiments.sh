# Shared experiment / run table, sourced by 1_data_preparation.sh and run_all.sh.
#
# One line per sequencing run:
#   EXP | HANDLE | S3 run folder (MinKNOW output, holds pod5/) | S3 basecalled folder | FIRST_BC
#
#   EXP       experiment name; local folder under Data/raw/ and Data/processed/
#   HANDLE    full run UUID; prefix of the basecalled BAM filenames. The local run
#             folder is its first 8 characters (e.g. 9fea1263).
#   S3 paths  relative to $S3_BASE
#   FIRST_BC  first of the 16 barcodes used: barcode FIRST_BC+i  ->  class i (i = 0..15)
#
# To add a run, append a line. Data layout:
#   Data/raw/<EXP>/<run>/{pod5/, basecalled/, MinKNOW reports}     (downloads only)
#   Data/processed/<EXP>/<run>/{aligned/bcNN.bam, signals.h5, preds.pkl}

S3_BASE="s3://gtl-public-data/miten/ryan"

PH009="03_10_26_NBD114_TT_PH009_fig004_BC1_16"
PH015="09_04_26_NBD114_PH015_PH016"

RUNS=(
    "PH009|9fea1263-540e-4cc6-ad49-9e80276dca2b|$PH009/$PH009/20260310_1620_1G_PBG61569_9fea1263|$PH009/basecalled/$PH009|1"
    "PH009|266e1025-91a2-48d9-a168-6386828065f6|$PH009/$PH009/20260310_1512_1G_PBG61569_266e1025|$PH009/basecalled/$PH009|1"
    "PH015|cd8c3e45-6d9d-4771-ae15-76e33aa0196e|$PH015/09_04_26_NBD114_PH015/20260904_1603_2E_PBK04830_cd8c3e45|$PH015/basecalled/09_04_26_NBD114_PH015|9"
    "PH015|78073ab4-0a86-4068-90c9-7eb0cca37ed3|$PH015/09_04_26_NBD114_PH015/20260908_1237_2E_PBK04830_78073ab4|$PH015/basecalled/09_04_26_NBD114_PH015|9"
)
N_BARCODES=16

# Optional subset, e.g. ONLY="PH009/266e1025" (space-separated <exp>/<run> keys).
if [[ -n "${ONLY:-}" ]]; then
    _kept=()
    for _line in "${RUNS[@]}"; do
        IFS='|' read -r _exp _handle _ <<< "$_line"
        [[ " $ONLY " == *" $_exp/${_handle:0:8} "* ]] && _kept+=("$_line")
    done
    if [[ ${#_kept[@]} -eq 0 ]]; then
        echo "ONLY=\"$ONLY\" matches no run in experiments.sh" >&2
        exit 1
    fi
    RUNS=("${_kept[@]}")
    unset _kept _line _exp _handle
fi

# Trained model + its train/val/test read split (both come from the PH009 prefix).
S3_MODEL_WEIGHTS="$PH009/model_weights"
S3_SPLIT_DICT="$PH009/split_dict.pkl"
# Run the model was trained on: plotted as train/val/test panels via split_dict.
TRAIN_RUN="PH009/9fea1263"

DATA="${DATA:-./Data}"
RAW="$DATA/raw"
PROCESSED="$DATA/processed"
MODEL_DIR="$DATA/model"
REF_DIR="$DATA/references"
REF="$REF_DIR/DNA_HS_DNA.fa"
CHECKPOINT="$MODEL_DIR/synthetics_classifier.pth"
SPLIT_DICT="$MODEL_DIR/split_dict.pkl"

# Split a RUNS line into EXP HANDLE RUN S3_RUN S3_BAM FIRST_BC.
parse_run() {
    IFS='|' read -r EXP HANDLE S3_RUN S3_BAM FIRST_BC <<< "$1"
    RUN="${HANDLE:0:8}"
}
