# HS-nano-seq — HS chain classification

Classify single nanopore reads of **DNA–HS–DNA chimeras** into one of 16 synthetic
heparan sulfate (HS) standards (fourteen 9-mers + two 18-mers) directly from the
ionic-current signal.

Each HS chain is flanked by two 60 bp DNA adapters and translocated through a
motor-driven nanopore. After basecalling and alignment to the adapter reference,
the HS-associated current segment plus the 12 nt HS–DNA junction window on each
side is isolated and fed to a transformer classifier. This repo covers the
16-way synthetic-standard classification.

## Repository structure

```
.
├── Data/                    # Will be available after running the data_preparation script
│   ├── references/          # DNA_HS_DNA.fa (concatenated NRE+RE adapters) + index
│   ├── synthetics/          # downloaded BAMs, aligned BAMs, Pod5s, etc.
│   └── model_weights/       # trained model weights (.pth)
├── data_preperation.sh      # download → build reference → align barcodes
├── signal_trimming.py       # aligned BAM + POD5 → trimmed signals in HDF5
├── model/                   # Model related scripts
│   ├── factory/
│   ├── transformer_multitask_zam_cls_only.py
│   └── __init__.py
├── model_inference.ipynb    # run a trained model on trimmed signals → predictions
└── results.ipynb            # confusion matrices / per-class recall from predictions
```

## Pipeline

The workflow runs in three stages.

**1. Data preparation** — `data_preperation.sh`

Downloads the basecalled, per-barcode BAMs from the public S3 bucket and the trained
model checkpoint (into `Data/model_weights/`), writes the `DNA_HS_DNA` reference (the
two DNA adapter sequences, 60 bp each) and indexes it, then aligns each barcode
(bc01–bc16) with `minimap2`.

```bash
bash data_preperation.sh
```

Aligned BAMs land in `Data/synthetics/aligned_bams/<handle>/`, where `<handle>` is
the experiment id.

**2. Signal trimming** — `signal_trimming.py`

For each aligned BAM (one barcode = one label) and the shared POD5 dataset,
maps each read's HS-flanking reference window back to raw signal via the basecaller
move table, trims the signal, and writes the result into a single HDF5 file.

```bash
python signal_trimming.py \
    --bam-dir  Data/synthetics/aligned_bams/<handle> \
    --pod5-dir Data/synthetics/<pod5_dir> \
    --out      Data/synthetics/signals/standards.h5 \
    --tnre 12 \
    --tre 12
```

`--tnre`/`--tre` set how many nucleotides of NRE/RE adapter to keep on each side of
the HS segment (default 12, matching the junction window in the paper).

The output HDF5 has four datasets, all aligned by read position `i`:

```
signals   int16   1D   all reads' trimmed signals, concatenated
offsets   int64   1D   length n_reads + 1; read i == signals[offsets[i]:offsets[i+1]]
read_ids  vlenstr 1D   length n_reads
labels    int64   1D   length n_reads; 0-based label (0..15) by sorted BAM order
```

Labels are **0-based**: barcode BC01 → class 0, …, BC16 → class 15.

**3. Inference & results** — `model_inference.ipynb`, `results.ipynb`

`model_inference.ipynb` streams **every read** in an HDF5 file through the trained
checkpoint (`Data/model_weights/`) and writes a prediction dict to a pickle:

```python
{ read_id (str): {"gt": int, "pred": int}, ... }   # gt / pred are 0-based (0..15)
```

`results.ipynb` loads the prediction pickle and the train/val/test split, then plots
a confusion matrix per split. Matrices are **row-normalized** (each row ÷ its total,
so the diagonal is per-class recall) and annotated with recall and raw count per
cell. Axes are labeled BC01–BC16 (class 0 → BC01, … class 15 → BC16).

## Requirements

- CLI tools: `awscli`, `samtools`, `minimap2`
- Python: `pod5`, `pysam`, `h5py`, `numpy`, `tqdm`, `torch`, `scikit-learn`, `matplotlib`

## Data & citation

TODO