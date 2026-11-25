# HS-nano-seq — HS chain classification

Classify single nanopore reads of **DNA–HS–DNA chimeras** into one of 16 synthetic
heparan sulfate (HS) standards (fourteen 9-mers + two 18-mers) directly from the
ionic-current signal.

Each HS chain is flanked by two DNA adapters (61 nt NRE + 60 nt RE) and translocated through a
motor-driven nanopore. After basecalling and alignment to the adapter reference,
the HS-associated current segment plus the 12 nt HS–DNA junction window on each
side is isolated and fed to a transformer classifier. This repo covers the
16-way synthetic-standard classification.

To reproduce the exact results reported in the paper, run the four pipeline stages below in order.

> [!WARNING]
> You can reproduce everything in one shot with `bash run_all.sh` — it runs all four
> stages in order (download/align → trim → infer → figures) for every run and writes the
> confusion-matrix figures to `figures/`. It accepts options such as `--only`, `--data`,
> `--threads`, `--trim-workers`, `--batch-size`, `--num-workers`, `--force-cpu`,
> `--skip-prep`, `--skip-trimming`, and `--skip-inference` (see `bash run_all.sh --help`).
> Otherwise, follow the stages below manually; the scripts are numbered in run order.
>
> `--tnre` / `--tre` are exposed for experimentation only: the trained model supports
> **only `--tnre 12 --tre 12`** (the default), and both scripts warn on anything else.

## Experiments

Two experiments, two sequencing runs each. All of them are listed in
`experiments.sh`, which every script reads; to add a run, add one line there.

| Experiment | Run        | Barcodes → classes      | Role                                  |
|------------|------------|-------------------------|---------------------------------------|
| PH009      | `9fea1263` | barcode01–16 → 0–15     | training run (train/val/test split)   |
| PH009      | `266e1025` | barcode01–16 → 0–15     | held-out                              |
| PH015      | `cd8c3e45` | barcode09–24 → 0–15     | held-out                              |
| PH015      | `78073ab4` | barcode09–24 → 0–15     | held-out                              |

Runs are named by the first 8 characters of their MinKNOW run UUID.

## Repository structure

```
.
├── Data/                          # populated by the pipeline (git-ignored)
│   ├── references/                # DNA_HS_DNA.fa (NRE+RE adapters, 121 nt) + index
│   ├── model/                     # synthetics_classifier.pth, split_dict.pkl
│   ├── raw/<exp>/<run>/           # downloads only — never written by later stages
│   │   ├── pod5/                  # raw signal
│   │   ├── basecalled/            # the 16 per-barcode BAMs of this run
│   │   └── *.txt|csv|html|…       # MinKNOW reports (sequencing_summary, throughput, …)
│   └── processed/<exp>/<run>/     # everything regenerated; rm -rf to reset
│       ├── aligned/bcNN.bam       # stage 1 (NN = barcode number)
│       ├── signals.h5             # stage 2
│       └── preds.pkl              # stage 3
├── experiments.sh                 # experiment / run table + data paths (shared config)
├── run_all.sh                     # one-shot: runs every stage + writes figures/
├── 1_data_preparation.sh          # download → build reference → align barcodes
├── 2_signal_trimming.py           # aligned BAM + POD5 → trimmed signals in HDF5
├── model/                         # model definition
│   ├── factory/
│   ├── zap_model.py
│   └── __init__.py
├── 3_model_inference.py           # run a trained model on trimmed signals → predictions
├── 4_results.py                   # confusion-matrix figures + accuracy summary → figures/
└── 4_results.ipynb                # the same figures inline, one section per run
```

## Pipeline

### 1. Data preparation — `1_data_preparation.sh`

For every run in `experiments.sh`, downloads from the public S3 bucket into
`Data/raw/<exp>/<run>/`: the run's POD5 files, its MinKNOW reports, and the 16
basecalled per-barcode BAMs. It also fetches the trained checkpoint and
`split_dict.pkl` into `Data/model/`. Downloads are include-only, so nothing else in
the bucket (fast5 copies, PH016, merged run-level BAMs, unused barcodes) is fetched;
a fresh download is ~950 GB.

It then writes the two DNA adapters (61 nt NRE + 60 nt RE, 121 nt total) as the
`DNA_HS_DNA` reference into `Data/references/`, indexes it, and aligns each run's 16
barcodes with `minimap2`, preserving the `mv` and `ts` tags that stage 2 relies on.
Aligned BAMs land in `Data/processed/<exp>/<run>/aligned/`. A missing barcode BAM is
an error, because it would shift every later label.

```
--only "KEYS"   only these <exp>/<run> keys, e.g. "PH009/266e1025"   (default: all runs)
--data PATH     data root                                             (default ./Data)
--threads N     threads passed to samtools / minimap2                 (default 32)
--dry-run       list what would be downloaded, then stop
```

Each option can also be set as an environment variable (`ONLY`, `DATA`, `THREADS`,
`DRYRUN=1`); a flag overrides it. Re-running is cheap: the S3 sync skips files already
present, and barcodes whose aligned BAM is already indexed are not re-aligned.

```bash
bash 1_data_preparation.sh --dry-run      # check what will be downloaded
bash 1_data_preparation.sh --threads 32
```

---

### 2. Signal trimming — `2_signal_trimming.py`

Takes one run's aligned BAMs (one barcode = one label) plus that run's POD5
files, maps each read's HS-flanking reference window back to raw signal via the
basecaller move table, trims the signal, and writes everything into a single HDF5
file. Reverse-strand alignments are skipped (their query coordinates refer to the
reverse-complemented read, which the move table doesn't follow); the per-BAM log
reports how many were dropped.

```
--bam-dir    (required)  directory of bcNN.bam files for one run; one file = one class
--pod5-dir   (required)  POD5 directory of that run, read recursively
--out        (required)  output .h5 path
--tnre       (int, 12)   nucleotides of NRE (left) adapter to keep beside the HS segment
--tre        (int, 12)   nucleotides of RE (right) adapter to keep beside the HS segment
                         (the trained model only supports --tnre 12 --tre 12)
--first-bc   (int, 1)    barcode number of class 0 (PH009: 1, PH015: 9)
--num-classes (int, 16)  number of classes
--min-len    (int, 18)   minimum trimmed signal length to keep; set this >= the model's
                         chunk_size, or short reads survive here and crash stage 3
--chunk      (int, 1e6)  HDF5 signal chunk size in samples; also caps peak RAM for the
                         signal buffer (~2 bytes per sample)
--workers    (int, 16)   worker processes (default min(16, CPUs))
```

It runs in two parallel phases: first each BAM is read to work out every read's trim
window in signal samples, then the POD5 files are streamed once, batch by batch, and
the windows are cut out (PH009/9fea1263, ~17 M reads: about 6 min with 16 workers).
Reads are stored in POD5 order, not grouped by barcode.

```bash
# one run; repeat for PH009/266e1025, PH015/cd8c3e45 and PH015/78073ab4 (PH015: --first-bc 9)
python 2_signal_trimming.py \
    --bam-dir  Data/processed/PH009/9fea1263/aligned \
    --pod5-dir Data/raw/PH009/9fea1263/pod5 \
    --out      Data/processed/PH009/9fea1263/signals.h5 \
    --first-bc 1
```

The output HDF5 has four datasets, all aligned by read position `i`:

```
signals   int16   1D   all reads' trimmed signals, concatenated
offsets   int64   1D   length n_reads + 1; read i == signals[offsets[i]:offsets[i+1]]
read_ids  vlenstr 1D   length n_reads
labels    int64   1D   length n_reads; 0-based label (0..15) by sorted BAM order
```

Labels come from the barcode number in each BAM's filename, never from file order:
class = NN − `--first-bc` (PH009: bc01–bc16 → 0–15 with `--first-bc 1`; PH015:
bc09–bc24 → 0–15 with `--first-bc 9`). A barcode outside that range or a duplicate is
an error; a missing one only gives a warning, since it can no longer shift the others.
`run_all.sh` passes the right `--first-bc` for each run from `experiments.sh`.

---

### 3. Inference — `3_model_inference.py`

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
# one run; repeat for the other three
python 3_model_inference.py \
    --h5            Data/processed/PH009/9fea1263/signals.h5 \
    --model-weights Data/model/synthetics_classifier.pth \
    --out           Data/processed/PH009/9fea1263/preds.pkl
```

---

### 4. Results — `4_results.py` / `4_results.ipynb`

`4_results.py` loads each run's `Data/processed/<exp>/<run>/preds.pkl` (runs come from
`experiments.sh`, so `ONLY=...` works here too), writes one confusion matrix per run
to `figures/confusion_matrix_<exp>_<run>.png`, and prints an accuracy table. Runs
without predictions yet are skipped with a warning.

```bash
python 4_results.py                      # all runs -> figures/
ONLY=PH015/78073ab4 python 4_results.py  # one run
```

`4_results.ipynb` shows the same figures inline, one section per run; it imports its
plotting from `4_results.py`, so the two never disagree.

For the run the model was trained on (PH009/9fea1263), the train/val/test split in
`Data/model/split_dict.pkl` is used to draw one panel per subset; every other run is
test-only and plotted as a single panel over all its reads. Matrices are
row-normalized (each row ÷ its total, so the diagonal is per-class recall) and
annotated with recall and raw count per cell. Axes show the class as its PH009 barcode
name (class 0 → BC01, … class 15 → BC16); each title also gives the run's own barcode
range (PH015: barcode09–24).

`run_all.sh` runs `4_results.py` as its stage 4. To redraw the figures from existing
predictions: `bash run_all.sh --skip-prep --skip-trimming --skip-inference`, or simply
`python 4_results.py`.

## Requirements

- CLI tools: `awscli`, `samtools`, `minimap2`
- Python: `pod5`, `pysam`, `h5py`, `numpy`, `tqdm`, `torch`, `scikit-learn`, `matplotlib`

## Data & citation

TODO
