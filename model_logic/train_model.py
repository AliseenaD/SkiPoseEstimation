#!/usr/bin/env python3
"""
train_model.py

Trains a Bidirectional LSTM to classify ski ability level
(beginner / intermediate / advanced) from pose feature sequences.

Inputs  (must exist before running):
  output/X.npy          (N, 30, 27)  float32 — LSTM windows
  output/y.npy          (N,)         int64   — class labels 0/1/2
  output/features.csv                        — frame-level data (used for video split)

Outputs written to output/:
  ski_classifier.pt             full model state dict
  scaler_mean.npy               per-feature mean  (apply before inference)
  scaler_std.npy                per-feature std   (apply before inference)
  training_history.npz          loss / accuracy curves for plotting
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("output")

WINDOW_SIZE = 30          # must match extract_features.py
STRIDE      = 15          # must match extract_features.py

BATCH_SIZE  = 16
MAX_EPOCHS  = 200
LR          = 1e-3
ES_PATIENCE = 20          # epochs without val_loss improvement before stopping
LR_PATIENCE = 10          # epochs before halving the learning rate

TRAIN_RATIO = 0.70        # fraction of *videos* assigned to train
VAL_RATIO   = 0.15        # fraction of *videos* assigned to val
                          # test gets the remainder

SEED        = 42

LABEL_MAP   = {"beginner": 0, "intermediate": 1, "advanced": 2}
CLASS_NAMES = ["beginner", "intermediate", "advanced"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")


# ── Reconstruct video → window mapping ─────────────────────────────────────────
def assign_windows_to_videos(csv_path: Path) -> list[tuple[str, int]]:
    """
    Returns (video_name, label_id) for every window in X.npy, in the same
    order that extract_features.py wrote them.  Reconstructs the sliding-window
    logic from the frame indices stored in the CSV so we can do a video-level
    split without re-running extraction.
    """
    df = pd.read_csv(csv_path)
    video_windows: list[tuple[str, int]] = []

    for level, label in LABEL_MAP.items():
        level_df = df[df["level"] == level]
        for video in sorted(level_df["video"].unique()):
            frame_indices = sorted(
                level_df[level_df["video"] == video]["frame_idx"].values
            )
            if not frame_indices:
                continue
            total_frames = max(frame_indices) + 1
            valid = np.zeros(total_frames, dtype=bool)
            for fi in frame_indices:
                valid[fi] = True
            for start in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
                if valid[start : start + WINDOW_SIZE].all():
                    video_windows.append((video, label))

    return video_windows


# ── Video-level stratified split ────────────────────────────────────────────────
def video_level_split(
    video_windows: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits window indices into train / val / test so that every window from
    the same video lands in the same split (no leakage).  Stratified by class.
    """
    video_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, (video, _) in enumerate(video_windows):
        video_to_idx[video].append(idx)

    video_label: dict[str, int] = {}
    for video, label in video_windows:
        video_label.setdefault(video, label)

    class_to_videos: dict[int, list[str]] = defaultdict(list)
    for video, label in video_label.items():
        class_to_videos[label].append(video)

    rng = random.Random(SEED)
    train_idx, val_idx, test_idx = [], [], []

    print("\nSplit breakdown (by video):")
    for label in sorted(class_to_videos):
        videos = list(class_to_videos[label])
        rng.shuffle(videos)

        n       = len(videos)
        n_train = max(1, round(n * TRAIN_RATIO))
        n_val   = max(1, round(n * VAL_RATIO))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        for v in videos[:n_train]:
            train_idx.extend(video_to_idx[v])
        for v in videos[n_train : n_train + n_val]:
            val_idx.extend(video_to_idx[v])
        for v in videos[n_train + n_val :]:
            test_idx.extend(video_to_idx[v])

        print(
            f"  {CLASS_NAMES[label]:12s}  "
            f"train={n_train:2d} ({n_train/n:.0%})  "
            f"val={n_val:2d}  "
            f"test={n - n_train - n_val:2d}"
        )

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx,   dtype=np.int64),
        np.array(test_idx,  dtype=np.int64),
    )


# ── Model ───────────────────────────────────────────────────────────────────────
class SkiClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.

    Architecture:
      Input (batch, 30, 27)
      → BiLSTM(32)          — hidden dim 32 per direction → 64 total
      → Dropout(0.5)
      → Linear(64 → 32) + ReLU
      → Dropout(0.4)
      → Linear(32 → 3)      — raw logits (CrossEntropyLoss handles softmax)
    """
    def __init__(self, n_features: int = 27, n_classes: int = 3) -> None:
        super().__init__()
        self.lstm    = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
        self.drop1   = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64, 32)
        self.relu    = nn.ReLU()
        self.drop2   = nn.Dropout(0.4)
        self.fc2     = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)          # h: (2, batch, 32)
        h = torch.cat([h[0], h[1]], dim=1) # (batch, 64) — fwd + bwd last hidden
        x = self.drop1(h)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)                 # logits (batch, 3)


# ── Training helpers ─────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """One forward pass over a DataLoader. Backprop only when optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)

    return total_loss / n, correct / n


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data …")
    X = np.load(OUTPUT_DIR / "X.npy")   # (N, 30, 27)
    y = np.load(OUTPUT_DIR / "y.npy")   # (N,)

    video_windows = assign_windows_to_videos(OUTPUT_DIR / "features.csv")

    if len(video_windows) != len(X):
        raise ValueError(
            f"Window count mismatch: CSV-derived {len(video_windows)} vs "
            f"X.npy {len(X)}.\nRe-run extract_features.py to regenerate outputs."
        )

    n_windows, n_frames, n_feat = X.shape
    print(f"  {n_windows} windows  ×  {n_frames} frames  ×  {n_feat} features")
    for lv, lb in LABEL_MAP.items():
        print(f"  {lv:12s}  {int((y == lb).sum())} windows")
    print(f"\nDevice: {DEVICE}")

    # ── Video-level split ─────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = video_level_split(video_windows)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(
        f"\n  train: {len(X_train):4d} windows  "
        f"val: {len(X_val):4d} windows  "
        f"test: {len(X_test):4d} windows"
    )

    # ── Feature standardisation (fit on train only) ───────────────────────
    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std  = (X_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    np.save(OUTPUT_DIR / "scaler_mean.npy", mean)
    np.save(OUTPUT_DIR / "scaler_std.npy",  std)
    print("Scaler saved  →  output/scaler_mean.npy  +  output/scaler_std.npy")

    # ── Class weights ─────────────────────────────────────────────────────
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    weights = len(y_train) / (len(CLASS_NAMES) * counts.clip(min=1))
    class_weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("\nClass weights (train):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}  weight={weights[i]:.3f}  ({counts[i]} windows)")

    # ── DataLoaders ───────────────────────────────────────────────────────
    def make_loader(Xd, yd, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(Xd, dtype=torch.float32),
            torch.tensor(yd, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model     = SkiClassifier(n_features=n_feat).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel  →  {total_params:,} trainable parameters")

    # ── Training loop ─────────────────────────────────────────────────────
    print("\nTraining …\n")
    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_imp  = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None)

        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        improved = va_loss < best_val_loss
        if improved:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        marker = " *" if improved else ""
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}  "
            f"lr={current_lr:.2e}{marker}"
        )

        if epochs_no_imp >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {ES_PATIENCE} epochs).")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Save artefacts ────────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features":       n_feat,
            "n_classes":        len(CLASS_NAMES),
            "class_names":      CLASS_NAMES,
        },
        OUTPUT_DIR / "ski_classifier.pt",
    )
    np.savez(OUTPUT_DIR / "training_history.npz", **{k: np.array(v) for k, v in history.items()})

    print("\n── Saved artefacts ──────────────────────────────────────────────")
    print("  output/ski_classifier.pt")
    print("  output/scaler_mean.npy / scaler_std.npy")
    print("  output/training_history.npz")

    # ── Test-set evaluation ───────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEVICE)).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    test_acc = (y_pred == y_true).mean()
    print(f"\n── Test set evaluation ──────────────────────────────────────────")
    print(f"  Accuracy: {test_acc:.4f}")

    print("\n  Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"    {name:12s}  {acc:.3f}  ({int(mask.sum())} windows)")

    print("\n  Confusion matrix  (rows = true, cols = predicted):")
    print(f"  {'':12s}  " + "  ".join(f"{n:^12s}" for n in CLASS_NAMES))
    for i, true_name in enumerate(CLASS_NAMES):
        row = [int(((y_true == i) & (y_pred == j)).sum()) for j in range(len(CLASS_NAMES))]
        print(f"  {true_name:12s}  " + "  ".join(f"{v:^12d}" for v in row))

    # ── Validation confusion matrix ───────────────────────────────────────
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
            val_labels.append(yb)
    plot_confusion_matrix(
        torch.cat(val_labels).numpy(),
        torch.cat(val_preds).numpy(),
        title="Validation confusion matrix",
        save_name="confusion_matrix_val.png",
    )

    # ── Test confusion matrix ─────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        title="Test confusion matrix",
        save_name="confusion_matrix_test.png",
    )

    # ── Training curves ───────────────────────────────────────────────────
    plot_history(history)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_name: str) -> None:
    import matplotlib.pyplot as plt

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate each cell with its count
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    out_path = OUTPUT_DIR / save_name
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved  →  {out_path}")
    plt.show()


def plot_history(history: dict) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nTraining curves saved  →  {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
