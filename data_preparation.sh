#!/usr/bin/env bash
#
# End-to-end synthetic-read pipeline:
#   1. download basecalled BAMs from the public S3 bucket
#   2. write + index the DNA_HS_DNA reference (.fa)
#   3. align barcode01..16 (preserving tags) -> sorted, indexed BAMs
#
# Requires: awscli, samtools, minimap2 on PATH.

set -euo pipefail

THREADS=32

DATA="./Data"
SYN="$DATA/synthetics"
REF_DIR="$DATA/references"
REF="$REF_DIR/DNA_HS_DNA.fa"
BASECALLED="$SYN/basecalled/03_10_26_NBD114_TT_PH009_fig004_BC1_16/"        # all experiments' BAMs live flat in here
ALIGNED="$SYN/aligned_bams"         # outputs go to aligned_bams/<handle>/

# Experiment handles. Each handle must be a substring of the BAM filenames for
# that experiment (it's used both to select that experiment's BAMs and as the
# output subfolder name). Fill these in:
HANDLES=(
    "9fea1263-540e-4cc6-ad49-9e80276dca2b"   # <-- experiment 1 handle
    "266e1025-91a2-48d9-a168-6386828065f6"   # <-- experiment 2 handle
)

# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
echo "==> [1/3] downloading from S3"
mkdir -p "$SYN"
aws s3 sync \
    s3://gtl-public-data/miten/ryan/03_10_26_NBD114_TT_PH009_fig004_BC1_16/ \
    "$SYN" \
    --exclude "03_10_26_NBD114_TT_PH009_fig004_BC1_16_dorado_1.0.0.moves.bam" \
    --no-sign-request

# --------------------------------------------------------------------------- #
# 2. Reference (.fa) + index
# --------------------------------------------------------------------------- #
echo "==> [2/3] writing + indexing reference"
mkdir -p "$REF_DIR"
cat > "$REF" << 'EOF'
>DNA_HS_DNA
TACGAGCGGTTGAGAAAGCTACGCCCGCGAATTACAGAGCGCACGGTTTTATTCTTACTTCTTGCCTCGGTGCGAACTGTATGAACTACTACTCCAGTGCTACGAGTTGCCGAAGAACTGA
EOF
samtools faidx "$REF"          # creates DNA_HS_DNA.fa.fai
# (minimap2 builds its index on the fly for a reference this small; no .mmi needed)

# --------------------------------------------------------------------------- #
# 3. Align bc01..bc16, per experiment handle
# --------------------------------------------------------------------------- #
echo "==> [3/3] aligning barcodes 01..16 per experiment"

for handle in "${HANDLES[@]}"; do
    if [[ -z "$handle" ]]; then
        echo "  !! empty handle in HANDLES array -- fill it in; skipping"
        continue
    fi

    out_dir="$ALIGNED/$handle"
    mkdir -p "$out_dir"
    echo "  == experiment: $handle  ->  $out_dir"

    for i in $(seq 1 16); do
        bc=$(printf "%02d" "$i")

        # this experiment's BAM for this barcode: name contains both handle and barcodeNN
        in=$(ls "$BASECALLED"/*"${handle}"*_barcode"${bc}".bam 2>/dev/null | head -n1 || true)
        if [[ -z "$in" ]]; then
            echo "     -- bc${bc}: no input BAM found, skipping"
            continue
        fi

        # output name: <uuid>_bc01.bam  (strip the _SQK...barcodeNN part, append _bcNN)
        stem=$(basename "$in" | sed "s/_SQK.*$//")
        out="$out_dir/${stem}_bc${bc}.bam"

        echo "     -- bc${bc}: $(basename "$in") -> $(basename "$out")"

        samtools fastq -@ "$THREADS" -T'*' "$in" \
            | minimap2 -ax map-ont -t "$THREADS" -y "$REF" - \
            | samtools view -@ "$THREADS" -h -F 4 -b - \
            | samtools sort -@ "$THREADS" -o "$out" -

        samtools index "$out"
    done
done

echo "==> done. Aligned BAMs in $ALIGNED/<handle>/"