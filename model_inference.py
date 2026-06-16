#!/usr/bin/env python3
"""
Inference over every read in an HDF5 signal file.

Streams all reads from the HDF5 (no train/val/test split), runs them through a
trained model checkpoint, and writes a pickle dict:

    { read_id (str): {"gt": int, "pred": int}, ... }
    
gt / pred are 0-based class indices.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import time
import pickle
import argparse
from types import SimpleNamespace

import numpy as np
import h5py
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

sys.path.append("../")

from model import TransformerZAM_multitask_CLSOnly


# --------------------------------------------------------------------------- #
# Preprocessing (identical to training)
# --------------------------------------------------------------------------- #
class SequenceStandardizer:
    def __init__(self):
        self.mean_array = None
        self.std_array = None

    def fit(self, signal_arrays):
        total_sum = np.zeros((signal_arrays[0].shape[1],))
        total_sq_sum = np.zeros((signal_arrays[0].shape[1],))
        total_count = 0
        for signal_array in signal_arrays:
            total_sum += np.nansum(signal_array, axis=0)
            total_sq_sum += np.nansum(signal_array**2, axis=0)
            total_count += signal_array.shape[0]
        self.mean_array = total_sum / total_count
        variance = (total_sq_sum / total_count) - (self.mean_array**2)
        self.std_array = np.sqrt(variance + 1e-8)
        self.mean_array = self.mean_array.reshape(1, -1)
        self.std_array = self.std_array.reshape(1, -1)

    def transform(self, signal_arrays):
        if self.mean_array is None or self.std_array is None:
            raise ValueError("SequenceStandardizer has not been fitted yet.")
        transformed_arrays = []
        for signal_array in signal_arrays:
            standardized_array = (signal_array - self.mean_array) / self.std_array
            transformed_arrays.append(np.nan_to_num(standardized_array))
        return transformed_arrays

    def fit_transform(self, signal_arrays):
        self.fit(signal_arrays)
        return self.transform(signal_arrays)


def local_standardizer(signal):
    standardizer = SequenceStandardizer()
    return standardizer.fit_transform([signal.reshape(-1, 1)])[0].flatten()


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def apply_rolling_window(window_size, step_size, array):
    if array.shape[0] < window_size:
        raise ValueError("Array size must be greater than window size.")
    shape = (array.size - window_size + 1, window_size)
    strides = array.strides * 2
    windows = np.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    )[0::step_size]
    return windows


def load_signal(signal, window_size, step_size, max_seq_len):
    signal_len = signal.shape[0]
    signal = signal[: (signal_len // window_size) * window_size]
    signal = apply_rolling_window(window_size, step_size, signal)
    if max_seq_len and len(signal) > max_seq_len:
        signal = signal[:max_seq_len]
    return signal


# --------------------------------------------------------------------------- #
# Dataset over ALL reads (no split)
# --------------------------------------------------------------------------- #
class HDF5SignalsData(data.Dataset):
    def __init__(self, indices, hdf5_path, transform,
                 window_size, step_size, max_seq_len, signal_dtype):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.window_size = window_size
        self.step_size = step_size
        self.max_seq_len = max_seq_len
        self.signal_dtype = signal_dtype
        self.hdf5_path = hdf5_path
        self._h5 = None
        with h5py.File(hdf5_path, "r") as f:
            self.offsets = f["offsets"][:]
            self.labels = f["labels"][:]

    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, i):
        idx = self.indices[i]
        label = self.labels[idx]
        f = self._get_file()
        start, end = self.offsets[idx], self.offsets[idx + 1]
        signal = f["signals"][start:end].astype(self.signal_dtype)
        signal = self.transform(signal).astype(self.signal_dtype)
        signal = load_signal(signal, self.window_size, self.step_size, self.max_seq_len)
        return {
            "inputs": {"signal": signal, "length": signal.shape[0]},
            "labels": {"classification": label},
            "metadata": {"read_id": int(idx), "num_tokens": signal.shape[0]},
        }

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def collate_fn(batch):
    signals = [torch.tensor(item["inputs"]["signal"]) for item in batch]
    lengths = torch.tensor([item["inputs"]["length"] for item in batch])
    read_ids = [item["metadata"]["read_id"] for item in batch]

    padded_signals = pad_sequence(signals, batch_first=True, padding_value=-1)

    out = {
        "inputs": {
            "signal": padded_signals,
            "length": lengths,
        },
        "metadata": {
            "read_id": read_ids,
            "num_tokens": lengths,
        },
    }
    if "labels" in batch[0] and "classification" in batch[0]["labels"]:
        out["labels"] = {
            "classification": torch.tensor(
                [item["labels"]["classification"] for item in batch]
            )
        }
    return out


# --------------------------------------------------------------------------- #
# Model / checkpoint
# --------------------------------------------------------------------------- #
REQUIRED_CONFIG_KEYS = [
    "chunk_size", "max_seq_len", "num_classes", "nhead", "num_layers",
    "dim_feedforward", "hidden_size", "dropout", "positional_encoding_type",
    "float_dtype",
]


def load_checkpoint(weights_path, device):
    """Load a self-describing checkpoint {'state_dict':..., 'config':...}."""
    obj = torch.load(weights_path, map_location=device)
    if not (isinstance(obj, dict) and "config" in obj and "state_dict" in obj):
        raise SystemExit(
            f"{weights_path} is not a self-describing checkpoint.\n"
            f"Run upgrade_checkpoint.py on it first to attach the architecture "
            f"config, then point --model_weights at the upgraded file."
        )
    config = obj["config"]
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise SystemExit(f"Checkpoint config is missing keys: {missing}")
    return obj["state_dict"], SimpleNamespace(**config)


def init_model(cfg):
    model = TransformerZAM_multitask_CLSOnly(
        input_size=cfg.chunk_size,
        max_seq_len=cfg.max_seq_len,
        num_classes=cfg.num_classes,
        num_heads=cfg.nhead,
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        dim_feedforward=cfg.dim_feedforward,
        dropout_rate_transformer=cfg.dropout,
        positional_encoding_type=cfg.positional_encoding_type,
    )
    if cfg.float_dtype == "float64":
        model = model.double()
    return model


def load_state(model, state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})", flush=True)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})", flush=True)
    return model


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(args):
    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Forcing CPU." if args.force_cpu else "CUDA not available; using CPU.",
              flush=True)
    else:
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)

    state_dict, cfg = load_checkpoint(args.model_weights, device)
    print("Architecture (from checkpoint):", flush=True)
    for k in REQUIRED_CONFIG_KEYS:
        print(f"    {k}: {getattr(cfg, k)}", flush=True)

    print("Reading read_ids + label count from HDF5 ...", flush=True)
    with h5py.File(args.h5, "r") as f:
        n = f["labels"].shape[0]
        raw_ids = f["read_ids"][:]
    print(f"  {n} reads total", flush=True)

    def to_str(x):
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

    all_idx = np.arange(n, dtype=np.int64)
    dataset = HDF5SignalsData(
        indices=all_idx,
        hdf5_path=args.h5,
        transform=local_standardizer,
        window_size=cfg.chunk_size,
        step_size=cfg.chunk_size,
        max_seq_len=cfg.max_seq_len,
        signal_dtype=np.float32,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    model = init_model(cfg).to(device)
    model = load_state(model, state_dict)
    model.eval()

    results = {}
    t0 = time.time()
    done = 0
    n_batches = len(loader)

    for bi, batch in enumerate(loader):
        inputs = batch["inputs"]
        signal = inputs["signal"].to(device, non_blocking=True)
        length = inputs.get("length", None)
        if isinstance(length, torch.Tensor):
            length = length.to(device, non_blocking=True)

        # NOTE: match this call to however EpochRunner invokes the model.
        outputs = model(signal=signal, length=length)
        logits = outputs["classification"]
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        gts = batch["labels"]["classification"]
        gts = gts.cpu().numpy() if isinstance(gts, torch.Tensor) else np.asarray(gts)

        idxs = batch["metadata"]["read_id"]
        idxs = idxs.cpu().numpy() if isinstance(idxs, torch.Tensor) else np.asarray(idxs)

        for idx, gt, pred in zip(idxs, gts, preds):
            rid = to_str(raw_ids[int(idx)])
            results[rid] = {"gt": int(gt), "pred": int(pred)}

        done += len(idxs)
        if bi % 50 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("inf")
            print(f"  batch {bi+1}/{n_batches}  reads {done}/{n}  "
                  f"{rate:,.0f} reads/s  ETA {eta/60:.1f} min", flush=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(results, fh)

    correct = sum(1 for v in results.values() if v["gt"] == v["pred"])
    acc = correct / len(results) if results else 0.0
    print(f"\nDone -> {args.out}", flush=True)
    print(f"  {len(results)} reads, overall accuracy {acc:.4f}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference over every read in an HDF5 file and dump "
                    "{read_id: {gt, pred}} to a pickle. Architecture is read "
                    "from the checkpoint."
    )
    p.add_argument("--h5", required=True, help="Path to the HDF5 signal file.")
    p.add_argument("--model-weights", required=True,
                   help="Path to a self-describing .pth (state_dict + config). "
                        "Use upgrade_checkpoint.py to create one.")
    p.add_argument("--out", default="inference_preds.pkl",
                   help="Output pickle path. Default: inference_preds.pkl")
    p.add_argument("--force-cpu", action="store_true",
                   help="Run on CPU even if a GPU is available.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())