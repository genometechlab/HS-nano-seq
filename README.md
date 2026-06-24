# HS-nano-seq — HS chain classification

Classify single nanopore reads of **DNA–HS–DNA chimeras** into one of 16 synthetic
heparan sulfate (HS) standards (fourteen 9-mers + two 18-mers) directly from the
ionic-current signal.

Each HS chain is flanked by two 60 bp DNA adapters and translocated through a
motor-driven nanopore. After basecalling and alignment to the adapter reference,
the HS-associated current segment plus the 12 nt HS–DNA junction window on each
side is isolated and fed to a transformer classifier. This repo covers the
16-way synthetic-standard classification.

To reproduce the exact results reported in the paper, run the three pipeline stages below in order.

> [!WARNING]
> You can reproduce everything in one shot with `bash run_all.sh` — it runs all four
> stages in order (download/align → trim → infer → figures) and writes the
> confusion-matrix figures to the project root. It accepts options such as
> `--threads`, `--batch-size`, `--num-workers`, `--force-cpu`, `--skip-prep`,
> `--skip-trimming`, and `--skip-inference` (see `bash run_all.sh --help`).
> Otherwise, follow the stages below manually.

## Repository structure

```
.
├── Data/                    # populated by data_preparation.sh
│   ├── references/          # DNA_HS_DNA.fa (concatenated NRE+RE adapters) + index
│   ├── results/             # per-experiment prediction pickles (written by inference)
│   └── synthetics/          # downloaded BAMs, aligned BAMs, Pod5s, model_weights/, etc.
│       └── model_weights/   # trained model checkpoint (.pth)
├── run_all.sh               # one-shot: runs every stage + writes figures
├── data_preparation.sh      # download → build reference → align barcodes
├── signal_trimming.py       # aligned BAM + POD5 → trimmed signals in HDF5
├── model/                   # Model related scripts
│   ├── factory/
│   ├── transformer_multitask_zam_cls_only.py
│   └── __init__.py
├── model_inference.py       # run a trained model on trimmed signals → predictions
└── results.ipynb            # confusion matrices / per-class recall from predictions
```

## Pipeline

### 1. Data preparation — `data_preparation.sh`

Downloads the basecalled per-barcode BAMs, the shared POD5 dataset, and the trained
model checkpoint from the public S3 bucket into `Data/synthetics/`, writes the two 60 bp DNA adapters as the
`DNA_HS_DNA` reference into `Data/references/` and indexes it, then aligns each
barcode (bc01–bc16) of every experiment with `minimap2`, preserving the `mv` and `ts`
tags that stage 2 relies on. Aligned BAMs land in `Data/synthetics/aligned_bams/<handle>/`.

Configure it by editing the variables at the top of the script before running:

```
THREADS   threads passed to samtools / minimap2
```

```bash
sbatch data_preparation.sh
```

---

### 2. Signal trimming — `signal_trimming.py`

Takes one experiment's aligned BAMs (one barcode = one label) plus the shared POD5
dataset, maps each read's HS-flanking reference window back to raw signal via the
basecaller move table, trims the signal, and writes everything into a single HDF5
file.

```
--bam-dir    (required)  directory of per-barcode BAMs for one handle; one file = one label
--pod5-dir   (required)  shared POD5 dataset directory, read recursively
--out        (required)  output .h5 path
--tnre       (int, 12)   nucleotides of NRE (left) adapter to keep beside the HS segment
--tre        (int, 12)   nucleotides of RE (right) adapter to keep beside the HS segment
--min-len    (int, 18)   minimum trimmed signal length to keep; set this >= the model's
                         chunk_size, or short reads survive here and crash stage 3
--chunk      (int, 1e6)  HDF5 signal chunk size in samples; also caps peak RAM for the
                         signal buffer (~2 bytes per sample)
```

```bash
# handle XXXX
python signal_trimming.py \
    --bam-dir  Data/synthetics/aligned_bams/9fea1263-540e-4cc6-ad49-9e80276dca2b \
    --pod5-dir Data/synthetics/03_10_26_NBD114_TT_PH009_fig004_BC1_16/20260310_1620_1G_PBG61569_9fea1263 \
    --out      Data/synthetics/signals/9fea1263-540e-4cc6-ad49-9e80276dca2b.h5 \
    --tnre 12 \
    --tre 12

# handle YYYYYY
python signal_trimming.py \
    --bam-dir  Data/synthetics/aligned_bams/266e1025-91a2-48d9-a168-6386828065f6 \
    --pod5-dir Data/synthetics/03_10_26_NBD114_TT_PH009_fig004_BC1_16/20260310_1512_1G_PBG61569_266e1025 \
    --out      Data/synthetics/signals/266e1025-91a2-48d9-a168-6386828065f6.h5 \
    --tnre 12 \
    --tre 12
```

The output HDF5 has four datasets, all aligned by read position `i`:

```
signals   int16   1D   all reads' trimmed signals, concatenated
offsets   int64   1D   length n_reads + 1; read i == signals[offsets[i]:offsets[i+1]]
read_ids  vlenstr 1D   length n_reads
labels    int64   1D   length n_reads; 0-based label (0..15) by sorted BAM order
```

Labels are 0-based and positional (sorted BAM filename order): BC01 → class 0, …,
BC16 → class 15. Make sure all 16 `*_bcNN.bam` files are present in the handle
directory before trimming, or the mapping shifts.

---

### 3. Inference — `model_inference.py`

Streams every read in an HDF5 file through the trained checkpoint and writes a
prediction dict to a pickle, `{ read_id: {"gt": int, "pred": int} }` (gt/pred are
0-based, 0..15). The architecture is read from the checkpoint's own `config`, so
there are no architecture flags.

```
--h5             (required)        path to the HDF5 signal file from stage 2
--model-weights  (required)        self-describing .pth (state_dict + config)
--out            (preds.pkl)       output pickle path
--batch-size     (int, 512)        reads per forward pass
--num-workers    (int, 4)          DataLoader worker processes
--force-cpu      (flag)            run on CPU even if a GPU is available
```

```bash
# handle XXXX
python model_inference.py \
    --h5            Data/synthetics/signals/9fea1263-540e-4cc6-ad49-9e80276dca2b.h5 \
    --model-weights Data/synthetics/model_weights/synthetics_classifier.pth \
    --out           Data/results/9fea1263-540e-4cc6-ad49-9e80276dca2b.pkl

# handle YYYYYY
python model_inference.py \
    --h5            Data/synthetics/signals/266e1025-91a2-48d9-a168-6386828065f6.h5 \
    --model-weights Data/synthetics/model_weights/synthetics_classifier.pth \
    --out           Data/results/266e1025-91a2-48d9-a168-6386828065f6.pkl
```

---

### 4. Results — `results.ipynb`

Loads the prediction pickles from `Data/results/` and plots a confusion matrix per
experiment. For the experiment the model was trained on, the train/val/test split in
`Data/synthetics/split_dict.pkl` is used to draw one panel per subset; the held-out
experiment is plotted as a single test panel over all its reads. Matrices are
row-normalized (each row ÷ its total, so the diagonal is per-class recall) and
annotated with recall and raw count per cell; axes are labeled BC01–BC16 (class 0 →
BC01, … class 15 → BC16).

Running `run_all.sh` produces the same figures non-interactively as
`confusion_matrix_<handle>.png` in the project root.

## Requirements

- CLI tools: `awscli`, `samtools`, `minimap2`
- Python: `pod5`, `pysam`, `h5py`, `numpy`, `tqdm`, `torch`, `scikit-learn`, `matplotlib`

## Data & citation

TODO