"""Train the lightweight 2D‑CNN on the paper's room split.

Paper split:
  - Train:   Meeting Room + Lecture Room    (same‑environment)
  - Val:     Left Home Lab                   (unseen environment)
  - Test:    Right Home Lab                  (NLoS / through‑wall)

Preprocessing: per‑subcarrier Z‑score normalisation (fit on training set only).
Augmentations: asymmetric signal mixing + temporal shadowing.

Usage:
    python train.py                           # train from scratch
    python train.py --epochs 200 --lr 0.001   # custom params
    python train.py --eval-only               # evaluate existing checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.cnn2d_detector import LightweightFallCNN, count_parameters
from app.services.csi_preprocessor import CsiZScoreNormalizer
from app.services.augmentations import CSIAugmentation

# ── Paths ───────────────────────────────────────────────────────────────
DATA_DIR = Path("data/ENetFall_dataset_trained_networks")
_DEFAULT_OUTPUT_DIR = Path("data/checkpoints")
_DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZER_DIR = _DEFAULT_OUTPUT_DIR / "normalizer"

# These are set dynamically based on --output / --log-file args
MODEL_PATH: Path = _DEFAULT_OUTPUT_DIR / "lightweight_2dcnn_best.pth"
STATS_PATH: Path = _DEFAULT_OUTPUT_DIR / "training_results.json"
LOG_FILE_PATH: Path | None = None

# ── Room split (paper Table setup) ──────────────────────────────────────
TRAIN_DATASETS = [
    "dataset_meeting_room.mat",
    "dataset_lecture_room.mat",
]
VAL_DATASETS = [
    "dataset_home_lab(L).mat",
]
TEST_DATASETS = [
    "dataset_home_lab(R).mat",
]

ROOM_LABELS: dict[str, str] = {
    "dataset_meeting_room.mat": "meeting_room",
    "dataset_lecture_room.mat": "lecture_room",
    "dataset_home_lab(L).mat": "home_lab_left",
    "dataset_home_lab(R).mat": "home_lab_right",
}


# ── Helpers ─────────────────────────────────────────────────────────────

def load_mat_files(dataset_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load raw CSI data & labels from .mat files.

    Returns:
        data:   [N, 625, 90] float32
        labels: [N] int64  (0 = non‑fall, 1 = fall)
        rooms:  [N] str    room names
    """
    parts_data, parts_labels, parts_rooms = [], [], []
    for name in dataset_names:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        mat = sio.loadmat(path)
        d = np.asarray(mat["dataset_CSI_t"], dtype=np.float32)
        l = np.asarray(mat["dataset_labels"]).reshape(-1).astype(np.int64)
        parts_data.append(d)
        parts_labels.append(l)
        parts_rooms.extend([ROOM_LABELS.get(name, name)] * d.shape[0])
    return (
        np.concatenate(parts_data, axis=0),
        np.concatenate(parts_labels, axis=0),
        parts_rooms,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    return {
        "accuracy": (tp + tn) / n if n > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": n,
        "fall_pred_pct": (tp + fp) / n * 100 if n > 0 else 0.0,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Log file setup ─────────────────────────────────────────────────
    global LOG_FILE_PATH
    log_fh = None
    if args.log_file:
        LOG_FILE_PATH = Path(args.log_file)
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(LOG_FILE_PATH, "w", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    # ── Output paths ───────────────────────────────────────────────────
    global MODEL_PATH, STATS_PATH
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH = out.parent / f"{out.stem}_best.pth"
        STATS_PATH = out
    else:
        out = _DEFAULT_OUTPUT_DIR / "training_results.json"
        MODEL_PATH = _DEFAULT_OUTPUT_DIR / "lightweight_2dcnn_best.pth"
        STATS_PATH = out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────
    _log("\n═══ Loading data ═══")
    X_train, y_train, _ = load_mat_files(TRAIN_DATASETS)
    X_val, y_val, rooms_val = load_mat_files(VAL_DATASETS)
    X_test, y_test, rooms_test = load_mat_files(TEST_DATASETS)

    _log(f"  Train: {X_train.shape}, falls={int(y_train.sum())}/{len(y_train)}")
    _log(f"  Val:   {X_val.shape}, falls={int(y_val.sum())}/{len(y_val)}")
    _log(f"  Test:  {X_test.shape}, falls={int(y_test.sum())}/{len(y_test)}")

    # ── 2. Fit Z‑score normaliser on TRAINING data only ──────────────
    _log("\n═══ Preprocessing ═══")
    normalizer = CsiZScoreNormalizer.fit_on_numpy(X_train)
    normalizer.save(NORMALIZER_DIR)
    _log(f"  Z-score stats saved to {NORMALIZER_DIR}")

    # Apply normalisation
    X_train_n = normalizer.normalize_numpy(X_train)
    X_val_n = normalizer.normalize_numpy(X_val)
    X_test_n = normalizer.normalize_numpy(X_test)

    # Add channel dim: [N, 625, 90] → [N, 1, 625, 90]
    X_train_t = torch.from_numpy(X_train_n).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val_n).unsqueeze(1)
    X_test_t = torch.from_numpy(X_test_n).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train).long()
    y_val_t = torch.from_numpy(y_val).long()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── 3. Model ──────────────────────────────────────────────────────
    _log("\n═══ Model ═══")
    model = LightweightFallCNN(dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    _log(f"  LightweightFallCNN: {n_params:,} params ({n_params / 1e6:.2f} M)")

    # Class-balanced loss: pos_weight = #neg / #pos ≈ 185/154 ≈ 1.2
    # Prevents model from always predicting the majority (non-fall) class
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    _log(f"  pos_weight={n_neg/n_pos:.3f}  (neg={n_neg}, pos={n_pos})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )

    # Data augmentation
    augmentation = CSIAugmentation(
        p_mix=args.p_mix,
        p_shadow=args.p_shadow,
        p_stretch=args.p_stretch,
        p_noise=args.p_noise,
    )

    # ── 4. Training loop ──────────────────────────────────────────────
    _log(f"\n═══ Training ({args.epochs} epochs) ═══")
    best_val_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    # Save initial model so there is always a checkpoint
    torch.save(model.state_dict(), MODEL_PATH)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Apply augmentations
            batch_x = augmentation(batch_x, batch_y)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(1)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total

        # ── Validate ──
        val_metrics = _evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["f1"])

        # ── Log ──
        elapsed = time.time() - t0
        marker = "  <- best" if val_metrics["f1"] > best_val_f1 else ""
        _log(
            f"  Epoch {epoch:3d} | "
            f"train loss={train_loss_avg:.4f} acc={train_acc:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} "
            f"({elapsed:.1f}s){marker}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss_avg, 6),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_f1": round(val_metrics["f1"], 4),
        })

        # ── Save best ──
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            _log(f"  -> Saved best model to {MODEL_PATH}")

        # Early stopping
        if epoch - best_epoch >= args.early_stop_patience:
            _log(f"\n  Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            break

    # ── 5. Final test evaluation ────────────────────────────────────
    _log(f"\n═══ Final Evaluation (best model from epoch {best_epoch}) ═══")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    results: dict[str, Any] = {
        "model": "LightweightFallCNN",
        "params": n_params,
        "train_datasets": TRAIN_DATASETS,
        "val_datasets": VAL_DATASETS,
        "test_datasets": TEST_DATASETS,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "config": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "p_mix": args.p_mix,
            "p_shadow": args.p_shadow,
            "epochs_completed": epoch,
        },
        "val": {},
        "test": {},
        "per_room_test": {},
    }

    # Overall test
    _log("\n-- Overall Test (Right Home Lab / NLoS) --")
    test_loss, y_true_all, y_pred_all, y_conf_all = _evaluate_detailed(
        model, test_loader, device
    )
    test_metrics = compute_metrics(y_true_all, y_pred_all)
    results["test"] = {k: v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)}
    _print_metrics(test_metrics, "  ", log_fn=_log)

    # Per‑room on test set
    _log("\n-- Per-Room Breakdown --")
    for room in sorted(set(rooms_test)):
        indices = [i for i, r in enumerate(rooms_test) if r == room]
        r_true = np.array([y_test[i] for i in indices])
        r_pred = np.array([y_pred_all[i] for i in indices])
        rm = compute_metrics(r_true, r_pred)
        results["per_room_test"][room] = {k: v for k, v in rm.items() if not isinstance(v, np.ndarray)}
        _log(f"\n  {room}:")
        _log(f"    Samples={rm['total']}, Fall: true={rm['tp']+rm['fn']}, pred={rm['tp']+rm['fp']}")
        _log(f"    Acc={rm['accuracy']:.4f}  Prec={rm['precision']:.4f}  Rec={rm['recall']:.4f}  F1={rm['f1']:.4f}")
        _log(f"    TN={rm['tn']}  FP={rm['fp']}  FN={rm['fn']}  TP={rm['tp']}")

        # Highlight NLoS precision
        if "right" in room.lower() or "home_lab(R)" in room.lower():
            nlos_precision = rm["precision"]
            _log(f"  ╔══════════════════════════════════════════╗")
            _log(f"  ║  NLoS PRECISION: {nlos_precision:.1%}                  ║")
            _log(f"  ║  Paper B0 NLoS: 66% -> Target: 82%+     ║")
            _log(f"  ╚══════════════════════════════════════════╝")

    # ── Save results ──
    STATS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _log(f"\nResults saved to {STATS_PATH}")

    # ── Final comparison table ──
    _log("\n═══ Paper Comparison ═══")
    _log(f"  {'Scenario':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    _log(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    _log(f"  {'Paper B0 (NLoS)':<22} {'78.0%':>6} {'66.0%':>6} {'95.0%':>6} {'78.0%':>6}")
    _log(f"  {'Paper 2D-CNN (NLoS)':<22} {'86.0%':>6} {'82.0%':>6} {'86.0%':>6} {'84.0%':>6}")
    _log(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    _log(f"  {'Ours (NLoS Test)':<22} "
         f"{test_metrics['accuracy']:.1%}  "
         f"{test_metrics['precision']:.1%}  "
         f"{test_metrics['recall']:.1%}  "
         f"{test_metrics['f1']:.1%}")

    # ── Cleanup ──
    if log_fh is not None:
        log_fh.close()


# ── Evaluation helpers ─────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x).squeeze(1)  # logits
        if criterion is not None:
            total_loss += criterion(outputs, batch_y.float()).item() * batch_x.size(0)
        all_preds.append((torch.sigmoid(outputs) >= 0.5).long().cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    m = compute_metrics(y_true, y_pred)
    if criterion is not None:
        m["loss"] = total_loss / len(y_true)
    return m


@torch.no_grad()
def _evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (loss, y_true, y_pred, y_conf)."""
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x).squeeze(1)
        total_loss += criterion(outputs, batch_y.float()).item() * batch_x.size(0)
        probs = torch.sigmoid(outputs)
        all_preds.append((probs >= 0.5).long().cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())
        all_confs.append(probs.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_confs),
    )


def _print_metrics(m: dict[str, Any], prefix: str = "", log_fn=None) -> None:
    p = log_fn if log_fn else print
    p(f"{prefix}Accuracy:  {m['accuracy']:.4f}")
    p(f"{prefix}Precision: {m['precision']:.4f}")
    p(f"{prefix}Recall:    {m['recall']:.4f}")
    p(f"{prefix}F1:        {m['f1']:.4f}")
    p(f"{prefix}Confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight 2D-CNN for CSI fall detection (paper replication)"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--p-mix", type=float, default=0.5,
                        help="Asymmetric signal mixing probability")
    parser.add_argument("--p-shadow", type=float, default=0.5,
                        help="Temporal shadowing probability")
    parser.add_argument("--p-stretch", type=float, default=0.2,
                        help="Time stretching probability")
    parser.add_argument("--p-noise", type=float, default=0.2,
                        help="Gaussian noise probability")
    parser.add_argument("--lr-patience", type=int, default=20,
                        help="LR scheduler patience")
    parser.add_argument("--early-stop-patience", type=int, default=80,
                        help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0=disabled)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for training_results.json (also sets model .pth path)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path for training log file")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
