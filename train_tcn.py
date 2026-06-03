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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import settings
from app.services.augmentations import CSIAugmentation
from app.services.csi_preprocessor import CsiZScoreNormalizer
from app.services.tcn_detector import TemporalConvTransformer, count_parameters


DATA_DIR = Path("data/ENetFall_dataset_trained_networks")
_DEFAULT_OUTPUT_DIR = Path("data/checkpoints")
_DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH: Path = _DEFAULT_OUTPUT_DIR / "tcn_transformer_best.pth"
STATS_PATH: Path = _DEFAULT_OUTPUT_DIR / "tcn_training_results.json"
LOG_FILE_PATH: Path | None = None

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


def load_mat_files(dataset_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": n,
        "fall_pred_pct": (tp + fp) / n * 100 if n > 0 else 0.0,
    }


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (probs >= threshold).astype(np.int64)
    return compute_metrics(y_true.astype(np.int64), y_pred)


def find_best_threshold_for_f1(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.05, 0.95, 91, dtype=np.float32)
    best_thr = 0.5
    best = {"f1": -1.0}
    for thr in thresholds:
        m = metrics_from_probs(y_true, probs, float(thr))
        if m["f1"] > best["f1"]:
            best_thr = float(thr)
            best = m
    return best_thr, best


def find_threshold_for_high_precision(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.05, 0.99, 95, dtype=np.float32)
    candidates: list[tuple[float, dict[str, float]]] = []
    for thr in thresholds:
        m = metrics_from_probs(y_true, probs, float(thr))
        if m["precision"] >= target_precision and m["recall"] >= min_recall:
            candidates.append((float(thr), m))

    if candidates:
        best_thr, best_m = max(candidates, key=lambda t: (t[1]["recall"], t[1]["f1"], t[0]))
        return best_thr, best_m

    best_thr = 0.5
    best_m = {"precision": -1.0, "recall": -1.0, "f1": -1.0}
    for thr in thresholds:
        m = metrics_from_probs(y_true, probs, float(thr))
        if m["precision"] > best_m["precision"] or (
            m["precision"] == best_m["precision"] and (m["recall"], m["f1"]) > (best_m["recall"], best_m["f1"])
        ):
            best_thr = float(thr)
            best_m = m
    return best_thr, best_m


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
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x).squeeze(1)
        if criterion is not None:
            total_loss += criterion(logits, batch_y.float()).item() * batch_x.size(0)
        probs = torch.sigmoid(logits)
        all_preds.append((probs >= 0.5).long().cpu().numpy())
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
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x).squeeze(1)
        total_loss += criterion(logits, batch_y.float()).item() * batch_x.size(0)
        probs = torch.sigmoid(logits)
        all_preds.append((probs >= 0.5).long().cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())
        all_confs.append(probs.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_confs),
    )


def train(args: argparse.Namespace) -> None:
    log_fh = None
    if args.log_file:
        global LOG_FILE_PATH
        LOG_FILE_PATH = Path(args.log_file)
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(LOG_FILE_PATH, "w", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    global MODEL_PATH, STATS_PATH
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH = out.parent / f"{out.stem}_best.pth"
        STATS_PATH = out
    else:
        MODEL_PATH = _DEFAULT_OUTPUT_DIR / "tcn_transformer_best.pth"
        STATS_PATH = _DEFAULT_OUTPUT_DIR / "tcn_training_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    _log("\n═══ Loading data ═══")
    X_train, y_train, _ = load_mat_files(TRAIN_DATASETS)
    X_val, y_val, rooms_val = load_mat_files(VAL_DATASETS)
    X_test, y_test, rooms_test = load_mat_files(TEST_DATASETS)
    _log(f"  Train: {X_train.shape}, falls={int(y_train.sum())}/{len(y_train)}")
    _log(f"  Val:   {X_val.shape}, falls={int(y_val.sum())}/{len(y_val)}")
    _log(f"  Test:  {X_test.shape}, falls={int(y_test.sum())}/{len(y_test)}")

    _log("\n═══ Preprocessing ═══")
    normalizer = CsiZScoreNormalizer.fit_on_numpy(X_train)
    normalizer.save(settings.TCN_NORMALIZER_DIR)
    if args.output:
        (Path(args.output).parent / "normalizer").mkdir(parents=True, exist_ok=True)
        normalizer.save(Path(args.output).parent / "normalizer")
    _log(f"  Z-score stats saved to {settings.TCN_NORMALIZER_DIR}")

    X_train_n = normalizer.normalize_numpy(X_train)
    X_val_n = normalizer.normalize_numpy(X_val)
    X_test_n = normalizer.normalize_numpy(X_test)

    X_train_t = torch.from_numpy(X_train_n).transpose(1, 2).contiguous()
    X_val_t = torch.from_numpy(X_val_n).transpose(1, 2).contiguous()
    X_test_t = torch.from_numpy(X_test_n).transpose(1, 2).contiguous()
    y_train_t = torch.from_numpy(y_train).long()
    y_val_t = torch.from_numpy(y_val).long()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    _log("\n═══ Model ═══")
    model = TemporalConvTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = count_parameters(model)
    _log(f"  TemporalConvTransformer: {n_params:,} params ({n_params / 1e6:.2f} M)")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    _log(f"  pos_weight={n_neg/n_pos:.3f}  (neg={n_neg}, pos={n_pos})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)

    augmentation = CSIAugmentation(
        p_mix=args.p_mix,
        p_shadow=args.p_shadow,
        p_stretch=args.p_stretch,
        p_noise=args.p_noise,
    )

    _log(f"\n═══ Training ({args.epochs} epochs) ═══")
    best_val_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    torch.save(model.state_dict(), MODEL_PATH)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_x_aug = augmentation(batch_x.transpose(1, 2), batch_y).transpose(1, 2).contiguous()
            optimizer.zero_grad()
            logits = model(batch_x_aug).squeeze(1)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total

        val_metrics = _evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["f1"])

        elapsed = time.time() - t0
        marker = "  <- best" if val_metrics["f1"] > best_val_f1 else ""
        _log(
            f"  Epoch {epoch:3d} | "
            f"train loss={train_loss_avg:.4f} acc={train_acc:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} "
            f"({elapsed:.1f}s){marker}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss_avg, 6),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_metrics["accuracy"], 4),
                "val_precision": round(val_metrics["precision"], 4),
                "val_recall": round(val_metrics["recall"], 4),
                "val_f1": round(val_metrics["f1"], 4),
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            _log(f"  -> Saved best model to {MODEL_PATH}")

        if epoch - best_epoch >= args.early_stop_patience:
            _log(f"\n  Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            break

    _log(f"\n═══ Final Evaluation (best model from epoch {best_epoch}) ═══")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    val_loss, y_val_true, _, y_val_probs = _evaluate_detailed(model, val_loader, device)
    test_loss, y_test_true, _, y_test_probs = _evaluate_detailed(model, test_loader, device)

    if args.threshold_objective == "precision":
        decision_threshold, val_metrics = find_threshold_for_high_precision(
            y_val_true,
            y_val_probs,
            target_precision=float(args.target_precision),
            min_recall=float(args.min_recall),
        )
    else:
        decision_threshold, val_metrics = find_best_threshold_for_f1(y_val_true, y_val_probs)
    test_metrics = metrics_from_probs(y_test_true, y_test_probs, decision_threshold)

    results: dict[str, Any] = {
        "model": "TemporalConvTransformer",
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
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "p_mix": args.p_mix,
            "p_shadow": args.p_shadow,
            "p_stretch": args.p_stretch,
            "p_noise": args.p_noise,
            "threshold_objective": args.threshold_objective,
            "target_precision": args.target_precision,
            "min_recall": args.min_recall,
            "epochs_completed": epoch,
        },
        "decision_threshold": round(float(decision_threshold), 4),
        "val": {k: float(v) for k, v in val_metrics.items()},
        "val_loss": float(val_loss),
        "test": {k: float(v) for k, v in test_metrics.items()},
        "per_room_test": {},
        "history": history,
        "test_loss": float(test_loss),
    }

    for room in sorted(set(rooms_test)):
        indices = [i for i, r in enumerate(rooms_test) if r == room]
        r_true = np.array([y_test[i] for i in indices], dtype=np.int64)
        r_probs = np.array([y_test_probs[i] for i in indices], dtype=np.float32)
        rm = metrics_from_probs(r_true, r_probs, decision_threshold)
        results["per_room_test"][room] = {k: float(v) for k, v in rm.items()}

    STATS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _log(f"\nResults saved to {STATS_PATH}")

    if log_fh is not None:
        log_fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN+Transformer for CSI fall detection")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--p-mix", type=float, default=0.5)
    parser.add_argument("--p-shadow", type=float, default=0.5)
    parser.add_argument("--p-stretch", type=float, default=0.2)
    parser.add_argument("--p-noise", type=float, default=0.2)
    parser.add_argument("--lr-patience", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold-objective", type=str, default="f1", choices=["f1", "precision"])
    parser.add_argument("--target-precision", type=float, default=0.8)
    parser.add_argument("--min-recall", type=float, default=0.2)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
