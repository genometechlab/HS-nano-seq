#!/usr/bin/env python3
"""
Parallel extraction of trimmed nanopore signals to HDF5.

For each aligned barcode BAM in --bam-dir (one file == one class), match its
reads against the run's POD5 files in --pod5-dir, trim each read's signal to the
reference window defined by --tnre / --tre, and write everything into one HDF5
file. The trained model only supports --tnre 12 --tre 12.

Two phases:
  1. BAM phase (one worker per BAM): for every primary read, work out the
     trim window in signal samples from the alignment + move table + ts tag
     alone -- no signal is touched.
  2. POD5 phase (one task per POD5 batch): stream every batch once, in file
     order, match its reads against the BAM reads by read id (vectorized),
     and cut the window out of each matching read's signal. A single writer
     (the main process) appends results to the HDF5 file.

Each read's length comes from the POD5 metadata (num_samples), so a signal is
decompressed only once, and only for reads that are kept. Reads are stored in
POD5 order, not grouped by barcode.

Labels come from the barcode number in each BAM's filename (bcNN.bam or
<anything>_bcNN.bam): label = NN - --first-bc, which must fall in
0..--num-classes-1. PH009 uses barcode01-16 (--first-bc 1), PH015 barcode09-24
(--first-bc 9).

HDF5 layout
-----------
    signals     : int16 1D, all reads' signals concatenated
    offsets     : int64, length n_reads + 1; read i == signals[offsets[i]:offsets[i+1]]
    read_ids    : vlen str, length n_reads
    labels      : int64,   length n_reads; 0-based class per read (barcode - first_bc)
"""

import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import re
import sys
import uuid

import h5py
import numpy as np
import pod5
import pysam
from tqdm import tqdm


META_CHUNK = 65536


def extract_ts_tag(bam_read: pysam.AlignedSegment) -> int:
    """Extract the ts:i tag from a read."""
    for tag, val, dtype in bam_read.get_tags(with_value_type=True):
        if tag == "ts" and dtype == "i":
            return val

    sam_string = bam_read.to_string()
    for field in sam_string.split("\t"):
        if field.startswith("ts:i:"):
            try:
                return int(field.split(":")[2])
            except ValueError:
                continue

    raise ValueError(f"Read {bam_read.query_name} has no valid ts:i tag")


# --------------------------------------------------------------------------- #
# HDF5 writing (bounded memory: chunk-aligned signal buffer)
# --------------------------------------------------------------------------- #
class _ChunkBuffer:
    """
    Buffers signal samples and flushes only in whole-chunk multiples, so every
    write to the (chunked) HDF5 dataset lands on a chunk boundary -- no partial
    read-modify-write. Peak memory is ~one chunk regardless of dataset size.
    """

    def __init__(self, ds, chunk: int):
        self.ds = ds
        self.chunk = chunk
        self.parts: list[np.ndarray] = []  # accumulated arrays, total < ~2*chunk
        self.pending = 0                   # samples in self.parts
        self.written = 0                   # samples already in ds (multiple of chunk until close)

    def add(self, arr: np.ndarray) -> None:
        self.parts.append(arr)
        self.pending += len(arr)
        if self.pending >= self.chunk:
            self._flush_full()

    def _flush_full(self) -> None:
        buf = np.concatenate(self.parts)
        n_full = (len(buf) // self.chunk) * self.chunk  # chunk-aligned amount
        if n_full:
            self.ds.resize((self.written + n_full,))
            self.ds[self.written:self.written + n_full] = buf[:n_full]
            self.written += n_full
        remainder = buf[n_full:]
        self.parts = [remainder] if len(remainder) else []
        self.pending = len(remainder)

    def close(self) -> None:
        # final flush: the one (and only) partial chunk in the whole dataset
        if self.pending:
            buf = np.concatenate(self.parts)
            self.ds.resize((self.written + len(buf),))
            self.ds[self.written:self.written + len(buf)] = buf
            self.written += len(buf)
        self.parts = []
        self.pending = 0


def _append(ds, data) -> None:
    """Append a 1D array to a resizable HDF5 dataset (used for tiny per-read metadata)."""
    n = ds.shape[0]
    ds.resize((n + len(data),))
    ds[n:] = data


# per-read status codes from the BAM phase
CANDIDATE, REVERSE, NON_ALIGNED = 0, 1, 2
NO_END = -1   # window runs to the end of the read: resolved with num_samples

# Read table shared with POD5 workers. Set in the parent before the pool forks;
# plain numpy arrays, so copy-on-write pages are never touched (no refcounts).
_TABLE = None
_READERS = {}


# --------------------------------------------------------------------------- #
# BAM phase
# --------------------------------------------------------------------------- #
def _uuid_words(read_id: str):
    """Read id -> two big-endian uint64 words of its 16 UUID bytes."""
    b = uuid.UUID(read_id).bytes
    return int.from_bytes(b[:8], "big"), int.from_bytes(b[8:], "big")


def bam_windows(task):
    """
    One BAM -> arrays over its primary reads:
        hi, lo        uint64  read-id words
        status        int8    CANDIDATE / REVERSE / NON_ALIGNED
        start, end    int64   trim window in samples (NO_END = to num_samples)
    """
    label, bam_path, tnre, tre = task
    # reference trim window, inclusive: NRE ends at 60, RE starts at 61
    ref_start = 60 - tnre + 1
    ref_end = 61 + tre - 1

    hi, lo, status, start, end = [], [], [], [], []
    with pysam.AlignmentFile(bam_path) as bam_fh:
        for bam_read in bam_fh.fetch(until_eof=True):
            if bam_read.is_supplementary or bam_read.is_secondary:
                continue
            h, l = _uuid_words(bam_read.query_name)
            hi.append(h); lo.append(l)

            # query positions of reverse-strand alignments index the reverse-complemented
            # read, but the move table follows the original read; they can't be mapped
            if bam_read.is_reverse:
                status.append(REVERSE); start.append(0); end.append(0)
                continue

            ref_to_query = {r: q for q, r in bam_read.get_aligned_pairs(matches_only=True)}
            if ref_start not in ref_to_query or ref_end not in ref_to_query:
                status.append(NON_ALIGNED); start.append(0); end.append(0)
                continue

            # base -> first signal sample; the trailing NO_END stands for num_samples
            move_table = bam_read.get_tag("mv")
            stride, moves = move_table[0], np.asarray(move_table[1:], dtype=np.int8)
            base_to_signal = np.append(
                np.flatnonzero(moves == 1) * stride + extract_ts_tag(bam_read), NO_END
            )
            status.append(CANDIDATE)
            start.append(int(base_to_signal[ref_to_query[ref_start]]))
            end.append(int(base_to_signal[ref_to_query[ref_end] + 1]))

    return (
        label,
        np.array(hi, dtype=np.uint64), np.array(lo, dtype=np.uint64),
        np.array(status, dtype=np.int8),
        np.array(start, dtype=np.int64), np.array(end, dtype=np.int64),
    )


def build_table(results):
    """Concatenate per-BAM arrays into one table sorted by read id (hi, lo)."""
    labels = np.concatenate([np.full(len(r[1]), r[0], dtype=np.int64) for r in results])
    hi, lo, status, start, end = (np.concatenate([r[k] for r in results]) for k in range(1, 6))
    order = np.lexsort((lo, hi))
    table = {"hi": hi[order], "lo": lo[order], "label": labels[order],
             "status": status[order], "start": start[order], "end": end[order]}
    dup = (np.diff(table["hi"]) == 0) & (np.diff(table["lo"]) == 0)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} read ids appear in more than one BAM / twice")
    return table


# --------------------------------------------------------------------------- #
# POD5 phase
# --------------------------------------------------------------------------- #
def _reader(path):
    if path not in _READERS:
        _READERS[path] = pod5.Reader(path)
    return _READERS[path]


def _lookup(hi, lo):
    """Indices into _TABLE for (hi, lo) pairs, -1 where the read isn't in any BAM."""
    t_hi, t_lo = _TABLE["hi"], _TABLE["lo"]
    left = np.searchsorted(t_hi, hi, side="left")
    right = np.searchsorted(t_hi, hi, side="right")
    idx = np.full(len(hi), -1, dtype=np.int64)

    # common case: the high word is unique in the table -> check the low word
    single = (right - left) == 1
    hit = single.copy()
    hit[single] = t_lo[left[single]] == lo[single]
    idx[hit] = left[hit]

    # rare case: several reads share the high word -> search the low word among them
    for i in np.flatnonzero((right - left) > 1):
        j = left[i] + np.searchsorted(t_lo[left[i]:right[i]], lo[i])
        if j < right[i] and t_lo[j] == lo[i]:
            idx[i] = j
    return idx


def pod5_batch(task):
    """
    One POD5 batch -> (found, idx, lengths, signal)
        found    table indices of every BAM read seen in this batch (for stats)
        idx      table indices of the kept reads, in batch order
        lengths  trimmed length of each kept read
        signal   all kept reads' trimmed signals, concatenated (int16)
    """
    path, batch_index, min_len = task
    batch = _reader(path).get_batch(batch_index)

    col = batch.read_id_column                       # FixedSizeBinary(16): raw UUID bytes
    raw = np.frombuffer(col.buffers()[1], dtype=np.uint8)
    words = raw[col.offset * 16:(col.offset + len(col)) * 16].view(">u8").reshape(-1, 2)
    table_idx = _lookup(words[:, 0].astype(np.uint64), words[:, 1].astype(np.uint64))

    found = table_idx[table_idx >= 0]
    keep_idx, lengths, parts = [], [], []
    for row in np.flatnonzero(table_idx >= 0):
        t = table_idx[row]
        if _TABLE["status"][t] != CANDIDATE:
            continue
        read = batch.get_read(row)
        start, end = int(_TABLE["start"][t]), int(_TABLE["end"][t])
        if start == NO_END:
            start = read.num_samples
        if end == NO_END:
            end = read.num_samples
        if start >= end:
            continue
        trimmed = np.asarray(read.signal[start:end], dtype=np.int16)
        if len(trimmed) < min_len:
            continue
        parts.append(trimmed)
        keep_idx.append(t)
        lengths.append(len(trimmed))

    signal = np.concatenate(parts) if parts else np.empty(0, dtype=np.int16)
    return (found, np.array(keep_idx, dtype=np.int64),
            np.array(lengths, dtype=np.int64), signal)


def _words_to_id(hi, lo) -> str:
    return str(uuid.UUID(bytes=int(hi).to_bytes(8, "big") + int(lo).to_bytes(8, "big")))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(bam_dir, pod5_dir, out_path, tnre, tre, min_len, chunk, workers,
        first_bc, num_classes):
    global _TABLE

    bam_paths = sorted(glob.glob(osp.join(bam_dir, "*.bam")))
    if not bam_paths:
        raise SystemExit(f"No .bam files found in {bam_dir}")
    pod5_paths = sorted(glob.glob(osp.join(pod5_dir, "**", "*.pod5"), recursive=True))
    if not pod5_paths:
        raise SystemExit(f"No .pod5 files found in {pod5_dir}")

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if tnre != 12 or tre != 12:
        print(f"WARNING: --tnre {tnre} --tre {tre}: the trained model only supports 12/12;"
              " its predictions on these signals are not meaningful.", file=sys.stderr)

    # label from the barcode number in the filename, never from file order
    names, bam_labels = [], []
    for p in bam_paths:
        name = osp.splitext(osp.basename(p))[0]
        m = re.search(r"bc(\d+)$", name)
        if m is None:
            raise SystemExit(f"{p}: filename must end in bcNN.bam to give its barcode")
        label = int(m.group(1)) - first_bc
        if not 0 <= label < num_classes:
            raise SystemExit(f"{p}: barcode {m.group(1)} is outside "
                             f"{first_bc}..{first_bc + num_classes - 1} (--first-bc {first_bc})")
        if label in bam_labels:
            raise SystemExit(f"{p}: barcode {m.group(1)} appears in more than one BAM")
        names.append(name); bam_labels.append(label)
    missing = sorted(set(range(num_classes)) - set(bam_labels))
    if missing:
        print(f"WARNING: no BAM for class(es) {missing} "
              f"(barcodes {[c + first_bc for c in missing]})", file=sys.stderr)

    ctx = mp.get_context("fork")

    # ---- 1. BAM phase ------------------------------------------------------ #
    with ctx.Pool(min(workers, len(bam_paths))) as pool:
        tasks = [(label, p, tnre, tre) for label, p in zip(bam_labels, bam_paths)]
        results = list(tqdm(pool.imap(bam_windows, tasks), total=len(tasks),
                            desc="BAMs", unit="bam"))
    _TABLE = build_table(results)
    del results
    n_labels = num_classes

    # ---- 2. POD5 phase ----------------------------------------------------- #
    batch_tasks = []
    for path in pod5_paths:
        with pod5.Reader(path) as r:
            batch_tasks += [(path, b, min_len) for b in range(r.batch_count)]

    found_per_label = np.zeros(n_labels, dtype=np.int64)
    reverse_per_label = np.zeros(n_labels, dtype=np.int64)
    non_aligned_per_label = np.zeros(n_labels, dtype=np.int64)
    valid_per_label = np.zeros(n_labels, dtype=np.int64)

    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(out_path, "w") as h5, ctx.Pool(workers) as pool:
        signals_ds = h5.create_dataset(
            "signals", shape=(0,), maxshape=(None,), dtype=np.int16, chunks=(chunk,),
        )
        offsets_ds = h5.create_dataset(
            "offsets", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=(META_CHUNK,),
        )
        offsets_ds[0] = 0
        ids_ds = h5.create_dataset(
            "read_ids", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=(META_CHUNK,),
        )
        labels_ds = h5.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(META_CHUNK,),
        )

        sig_buf = _ChunkBuffer(signals_ds, chunk)
        cursor = 0
        for found, idx, lengths, signal in tqdm(
            pool.imap(pod5_batch, batch_tasks), total=len(batch_tasks),
            desc="POD5 batches", unit="batch",
        ):
            st = _TABLE["status"][found]
            lab = _TABLE["label"][found]
            found_per_label += np.bincount(lab, minlength=n_labels)
            reverse_per_label += np.bincount(lab[st == REVERSE], minlength=n_labels)
            non_aligned_per_label += np.bincount(lab[st == NON_ALIGNED], minlength=n_labels)
            if not len(idx):
                continue

            labels = _TABLE["label"][idx]
            valid_per_label += np.bincount(labels, minlength=n_labels)
            sig_buf.add(signal)
            ends = cursor + np.cumsum(lengths)
            cursor = int(ends[-1])
            _append(offsets_ds, ends)
            _append(ids_ds, np.array(
                [_words_to_id(h, l) for h, l in zip(_TABLE["hi"][idx], _TABLE["lo"][idx])],
                dtype=object,
            ))
            _append(labels_ds, labels)

        sig_buf.close()
        n_reads = int(valid_per_label.sum())

    # per-BAM stats (counts over reads found in POD5)
    for label, name in sorted(zip(bam_labels, names)):
        print(
            f"[{name}] label={label} "
            f"reverse={reverse_per_label[label]}/{found_per_label[label]} "
            f"non_aligned={non_aligned_per_label[label]}/{found_per_label[label]} "
            f"valid={valid_per_label[label]}"
        )

    if n_reads == 0:
        raise SystemExit("No valid reads extracted; nothing to write.")

    n_bams = int((valid_per_label > 0).sum())
    print(f"\nDone -> {out_path}")
    print(f"  {n_reads} reads across {n_bams} labels (labels are 0..{n_labels - 1})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract trimmed nanopore signals from BAM+POD5 into one HDF5 file (parallel)."
    )
    p.add_argument("--bam-dir", required=True,
                   help="Directory of bcNN.bam files; each file is one class.")
    p.add_argument("--pod5-dir", required=True,
                   help="POD5 dataset directory (read recursively).")
    p.add_argument("--out", required=True,
                   help="Output .h5 path.")
    p.add_argument("--tnre", type=int, default=12,
                   help="Left (non-reference-end) trim offset. Default 12; the trained "
                        "model only supports 12.")
    p.add_argument("--tre", type=int, default=12,
                   help="Right (reference-end) trim offset. Default 12; the trained "
                        "model only supports 12.")
    p.add_argument("--first-bc", type=int, default=1,
                   help="Barcode number of class 0 (PH009: 1, PH015: 9). Default 1.")
    p.add_argument("--num-classes", type=int, default=16,
                   help="Number of classes; barcodes first-bc .. first-bc+N-1. Default 16.")
    p.add_argument("--min-len", type=int, default=18,
                   help="Minimum trimmed signal length to keep. Default 18.")
    p.add_argument("--chunk", type=int, default=1_000_000,
                   help="HDF5 signal chunk size in samples. Default 1,000,000.")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                   help="Worker processes for the BAM and POD5 phases. Default min(16, CPUs).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        bam_dir=args.bam_dir,
        pod5_dir=args.pod5_dir,
        out_path=args.out,
        tnre=args.tnre,
        tre=args.tre,
        min_len=args.min_len,
        chunk=args.chunk,
        workers=args.workers,
        first_bc=args.first_bc,
        num_classes=args.num_classes,
    )
