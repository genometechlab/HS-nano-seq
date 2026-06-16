#!/usr/bin/env python3
"""
Single-pass extraction of trimmed nanopore signals to HDF5.

For each BAM file in --bam-dir (one file == one label), match reads against the
shared POD5 dataset in --pod5-dir, trim each read's signal to the reference
window defined by --tnre / --tre, and stream the result straight into one HDF5
file..

HDF5 layout
-----------
    signals     : int16 1D, all reads' signals concatenated
    offsets     : int64, length n_reads + 1; read i == signals[offsets[i]:offsets[i+1]]
    read_ids    : vlen str, length n_reads
    labels      : int64,   length n_reads; 0-based label per read (0..N-1 by sorted BAM order)

NOTE labels are 0-based (enumerate starts at 0)
"""

import argparse
import glob
import os
import os.path as osp

import h5py
import numpy as np
import pod5
import pysam
from tqdm import tqdm


META_CHUNK = 65536


# --------------------------------------------------------------------------- #
# BAM / POD5 helpers
# --------------------------------------------------------------------------- #
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


def moves_to_base_indices(
    moves: np.ndarray, stride: int, ts: int, ns: int
) -> np.ndarray:
    """Convert a move array to base position indices in the signal."""
    moves_array = np.array(moves, dtype="int8")
    move_positions = np.where(moves_array == 1)[0] * stride + ts
    return np.append(move_positions, ns)


def get_query_to_signal(
    bam_read: pysam.AlignedSegment, pod_read: pod5.ReadRecord
) -> np.ndarray:
    ts = extract_ts_tag(bam_read)
    ns = pod_read.signal.shape[0]
    move_table = bam_read.get_tag("mv")
    stride, moves = move_table[0], move_table[1:]
    return moves_to_base_indices(moves, stride, ts, ns)


def get_ref_to_query(bam_read: pysam.AlignedSegment) -> dict[int, int]:
    """Returns {ref_pos: query_pos} for each aligned (non-insertion) base."""
    ref_to_query: dict[int, int] = {}
    for q, r in bam_read.get_aligned_pairs():
        if q is None or r is None:
            continue  # deletion in query or insertion in ref
        ref_to_query[r] = q
    return ref_to_query


def get_bam_reads(bam_path: str) -> list[pysam.AlignedSegment]:
    bam_fh = pysam.AlignmentFile(bam_path)
    return [
        bam_read
        for bam_read in bam_fh.fetch(until_eof=True)
        if not (bam_read.is_supplementary or bam_read.is_secondary)
    ]


def fetch_pod_records(
    pod5_reader: pod5.DatasetReader, read_ids: list[str]
) -> dict[str, pod5.ReadRecord]:
    res: dict[str, pod5.ReadRecord] = {}
    for pod_read in pod5_reader.reads(selection=read_ids):
        res[str(pod_read.read_id)] = pod_read
    return res


# --------------------------------------------------------------------------- #
# Per-BAM processing
# --------------------------------------------------------------------------- #
def process_bam(
    bam_path: str,
    pod5_reader: pod5.DatasetReader,
    tnre: int,
    tre: int,
    min_len: int,
):
    """
    Returns (read_ids, signals, stats) for one BAM.
    signals is a list of int16 arrays aligned with read_ids.
    """
    bam_reads = get_bam_reads(bam_path)
    read_ids = [r.query_name for r in bam_reads]
    pod_map = fetch_pod_records(pod5_reader, read_ids)

    name = osp.splitext(osp.basename(bam_path))[0]

    # reference trim boundaries (both exclusive ends of the kept window)
    ref_start = 60 - tnre + 1
    ref_end = 61 + tre - 1

    out_ids: list[str] = []
    out_signals: list[np.ndarray] = []
    total = non_aligned = valid = 0

    for bam_read in tqdm(
        bam_reads, desc=name, unit="read",
        position=1, leave=False,
    ):
        rid = bam_read.query_name
        pod_read = pod_map.get(rid)
        if pod_read is None:
            continue
        total += 1

        query_to_signal = get_query_to_signal(bam_read, pod_read)
        ref_to_query = get_ref_to_query(bam_read)

        if ref_start not in ref_to_query or ref_end not in ref_to_query:
            non_aligned += 1
            continue

        q_start = ref_to_query[ref_start]
        q_end = ref_to_query[ref_end]
        signal_start = query_to_signal[q_start]
        signal_end = query_to_signal[q_end + 1]

        if signal_start >= signal_end:
            continue

        trimmed = np.asarray(
            pod_read.signal[signal_start:signal_end], dtype=np.int16
        )
        if len(trimmed) < min_len:
            continue

        valid += 1
        out_ids.append(rid)
        out_signals.append(trimmed)

    stats = {"total": total, "non_aligned": non_aligned, "valid": valid}
    return out_ids, out_signals, stats


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


def run(bam_dir, pod5_dir, out_path, tnre, tre, min_len, chunk):
    bam_paths = sorted(glob.glob(osp.join(bam_dir, "*.bam")))
    if not bam_paths:
        raise SystemExit(f"No .bam files found in {bam_dir}")

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    str_dt = h5py.special_dtype(vlen=str)

    with pod5.DatasetReader(pod5_dir, recursive=True) as pod5_reader, \
         h5py.File(out_path, "w") as h5:

        signals_ds = h5.create_dataset(
            "signals", shape=(0,), maxshape=(None,),
            dtype=np.int16, chunks=(chunk,),
        )
        offsets_ds = h5.create_dataset(
            "offsets", shape=(1,), maxshape=(None,), dtype=np.int64,
            chunks=(META_CHUNK,),
        )
        offsets_ds[0] = 0
        ids_ds = h5.create_dataset(
            "read_ids", shape=(0,), maxshape=(None,), dtype=str_dt,
            chunks=(META_CHUNK,),
        )
        labels_ds = h5.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(META_CHUNK,),
        )

        sig_buf = _ChunkBuffer(signals_ds, chunk)
        cursor = 0       # running sample offset for the offsets table
        n_reads = 0
        n_bams = 0

        for label, bam_path in enumerate(
            tqdm(bam_paths, desc="BAMs", position=0, leave=True), start=0
        ):
            name = osp.splitext(osp.basename(bam_path))[0]

            read_ids, sigs, stats = process_bam(
                bam_path, pod5_reader, tnre, tre, min_len
            )
            tqdm.write(
                f"[{name}] label={label} "
                f"non_aligned={stats['non_aligned']}/{stats['total']} "
                f"valid={stats['valid']}"
            )
            if not sigs:
                continue
            n_bams += 1

            for s in sigs:
                sig_buf.add(s)

            lengths = np.fromiter((len(s) for s in sigs), dtype=np.int64, count=len(sigs))
            ends = cursor + np.cumsum(lengths)
            cursor = int(ends[-1])
            _append(offsets_ds, ends)
            _append(ids_ds, np.array(read_ids, dtype=object))
            _append(labels_ds, np.full(len(read_ids), label, dtype=np.int64))
            n_reads += len(read_ids)

            del read_ids, sigs  # release this BAM's signals before the next one

        sig_buf.close()

    if n_reads == 0:
        raise SystemExit("No valid reads extracted; nothing to write.")

    print(f"\nDone -> {out_path}")
    print(f"  {n_reads} reads across {n_bams} labels (labels are 0..{n_bams - 1})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract trimmed nanopore signals from BAM+POD5 into one HDF5 file."
    )
    p.add_argument("--bam-dir", required=True,
                   help="Directory of .bam files; each file is one label.")
    p.add_argument("--pod5-dir", required=True,
                   help="Shared POD5 dataset directory (read recursively).")
    p.add_argument("--out", required=True,
                   help="Output .h5 path.")
    p.add_argument("--tnre", type=int, default=12,
                   help="Left (non-reference-end) trim offset. Default 12.")
    p.add_argument("--tre", type=int, default=12,
                   help="Right (reference-end) trim offset. Default 12.")
    p.add_argument("--min-len", type=int, default=18,
                   help="Minimum trimmed signal length to keep. Default 18.")
    p.add_argument("--chunk", type=int, default=1_000_000,
                   help="HDF5 signal chunk size in samples; also caps peak RAM "
                        "for the signal buffer (~2 bytes each, int16). Default 1,000,000.")
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
    )