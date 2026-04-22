#!/usr/bin/env python3
"""
train_model_kfold.py

5-fold cross-validation at the video level for the Bidirectional LSTM ski
classifier.  Every video gets a turn as the held-out fold, so validation
metrics are stable despite the small dataset (~48 videos).

Inputs  (must exist before running):
  output/X.npy          (N, 30, 27)  float32 — LSTM windows
  output/y.npy          (N,)         int64   — class labels 0/1/2
  output/features.csv                        — frame-level data (used for fold assignment)

Outputs written to output/:
  ski_classifier_kfold.pt           best fold model (highest val accuracy)
  scaler_mean_kfold.npy             scaler from best fold's training set
  scaler_std_kfold.npy              scaler from best fold's training set
  training_curves_kfold.png         per-fold loss + accuracy curves
  confusion_matrix_kfold.png        aggregated across all folds (every sample predicted once)
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

WINDOW_SIZE = 30
STRIDE      = 15
N_FOLDS     = 5

BATCH_SIZE  = 16
MAX_EPOCHS  = 200
LR          = 1e-3
ES_PATIENCE = 20
LR_PATIENCE = 10

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


# ── Stratified video-level k-fold assignment ────────────────────────────────────
def make_video_folds(video_windows: list[tuple[str, int]], n_folds: int, seed: int) -> dict[str, int]:
    """
    Assigns each unique video to a fold number (0 … n_folds-1).
    Videos are shuffled within each class then distributed round-robin so
    every fold has roughly the same class distribution.

    Returns {video_name: fold_id}.
    """
    video_label: dict[str, int] = {}
    for video, label in video_windows:
        video_label.setdefault(video, label)

    class_to_videos: dict[int, list[str]] = defaultdict(list)
    for video, label in video_label.items():
        class_to_videos[label].append(video)

    rng = random.Random(seed)
    fold_assignment: dict[str, int] = {}

    for label in sorted(class_to_videos):
        videos = list(class_to_videos[label])
        rng.shuffle(videos)
        for i, video in enumerate(videos):
            fold_assignment[video] = i % n_folds

    return fold_assignment


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
        self.lstm  = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc1   = nn.Linear(64, 32)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        x = self.drop1(h)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


# ── Training helpers ─────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
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


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_feat: int,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    """
    Train one fold. Returns (best_state_dict, history, scaler_mean, scaler_std).
    """
    # Standardise
    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std  = (X_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # Class weights on this fold's training set
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    weights = len(y_train) / (len(CLASS_NAMES) * counts.clip(min=1))
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    def make_loader(Xd, yd, shuffle):
        ds = TensorDataset(
            torch.tensor(Xd, dtype=torch.float32),
            torch.tensor(yd, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    model     = SkiClassifier(n_features=n_feat).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val_loss = float("inf")
    best_state    = None
    epochs_no_imp = 0
    history       = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None)
        scheduler.step(va_loss)
        history.append({"train_loss": tr_loss, "val_loss": va_loss,
                         "train_acc": tr_acc,  "val_acc":  va_acc})

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= ES_PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_state, history, mean, std, model, val_loader


# ── Plotting ─────────────────────────────────────────────────────────────────────
def plot_fold_curves(all_histories: list[list[dict]]) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    colours = plt.cm.tab10.colors

    for fold_i, history in enumerate(all_histories):
        epochs    = range(1, len(history) + 1)
        tr_loss   = [h["train_loss"] for h in history]
        va_loss   = [h["val_loss"]   for h in history]
        tr_acc    = [h["train_acc"]  for h in history]
        va_acc    = [h["val_acc"]    for h in history]
        c         = colours[fold_i]
        label     = f"fold {fold_i + 1}"

        ax_loss.plot(epochs, tr_loss, color=c, linestyle="--", alpha=0.6)
        ax_loss.plot(epochs, va_loss, color=c, linestyle="-",  label=label)
        ax_acc.plot(epochs,  tr_acc,  color=c, linestyle="--", alpha=0.6)
        ax_acc.plot(epochs,  va_acc,  color=c, linestyle="-",  label=label)

    for ax, title, ylabel in [
        (ax_loss, "Loss (solid=val, dashed=train)", "Cross-entropy loss"),
        (ax_acc,  "Accuracy (solid=val, dashed=train)", "Accuracy"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_acc.set_ylim(0, 1)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "training_curves_kfold.png"
    fig.savefig(out_path, dpi=150)
    print(f"Training curves saved  →  {out_path}")
    plt.show()


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

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    out_path = OUTPUT_DIR / save_name
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved  →  {out_path}")
    plt.show()


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data …")
    X = np.load(OUTPUT_DIR / "X.npy")
    y = np.load(OUTPUT_DIR / "y.npy")

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

    # ── Assign each video to a fold ──────────────────────────────────────────
    fold_assignment = make_video_folds(video_windows, N_FOLDS, SEED)
    window_folds    = np.array([fold_assignment[v] for v, _ in video_windows])

    print(f"\nFold assignment ({N_FOLDS} folds):")
    for f in range(N_FOLDS):
        mask = window_folds == f
        counts = np.bincount(y[mask], minlength=len(CLASS_NAMES))
        print(f"  fold {f + 1}  {mask.sum():4d} windows  "
              + "  ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES))))

    # ── Cross-validation loop ────────────────────────────────────────────────
    all_histories:    list[list[dict]] = []
    all_val_preds:    list[np.ndarray] = [np.empty(0, int)] * N_FOLDS
    all_val_labels:   list[np.ndarray] = [np.empty(0, int)] * N_FOLDS
    fold_accuracies:  list[float]      = []

    best_fold_acc   = -1.0
    best_fold_state = None
    best_fold_mean  = None
    best_fold_std   = None

    for fold in range(N_FOLDS):
        print(f"\n── Fold {fold + 1}/{N_FOLDS} {'─' * 50}")

        val_mask   = window_folds == fold
        train_mask = ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]

        print(f"  train: {len(X_train)} windows   val: {len(X_val)} windows")

        best_state, history, mean, std, model, val_loader = train_fold(
            X_train.copy(), y_train.copy(),
            X_val.copy(),   y_val.copy(),
            n_feat,
        )
        all_histories.append(history)

        # Collect predictions on the held-out fold
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
                labels.append(yb)
        fold_preds  = torch.cat(preds).numpy()
        fold_labels = torch.cat(labels).numpy()

        all_val_preds[fold]  = fold_preds
        all_val_labels[fold] = fold_labels

        fold_acc = (fold_preds == fold_labels).mean()
        fold_accuracies.append(fold_acc)
        print(f"  fold {fold + 1} val accuracy: {fold_acc:.4f}")

        if fold_acc > best_fold_acc:
            best_fold_acc   = fold_acc
            best_fold_state = best_state
            best_fold_mean  = mean
            best_fold_std   = std

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n── Cross-validation results {'─' * 40}")
    for i, acc in enumerate(fold_accuracies):
        print(f"  fold {i + 1}:  {acc:.4f}")
    mean_acc = np.mean(fold_accuracies)
    std_acc  = np.std(fold_accuracies)
    print(f"  mean:   {mean_acc:.4f} ± {std_acc:.4f}")

    # ── Aggregated confusion matrix (every window predicted exactly once) ──
    y_all_true = np.concatenate(all_val_labels)
    y_all_pred = np.concatenate(all_val_preds)

    print("\n  Per-class accuracy (aggregated):")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_all_true == i
        if mask.sum() > 0:
            acc = (y_all_pred[mask] == i).mean()
            print(f"    {name:12s}  {acc:.3f}  ({int(mask.sum())} windows)")

    # ── Save best fold model ──────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict": best_fold_state,
            "n_features":       n_feat,
            "n_classes":        len(CLASS_NAMES),
            "class_names":      CLASS_NAMES,
            "best_fold":        int(np.argmax(fold_accuracies)) + 1,
            "cv_mean_acc":      float(mean_acc),
            "cv_std_acc":       float(std_acc),
        },
        OUTPUT_DIR / "ski_classifier_kfold.pt",
    )
    np.save(OUTPUT_DIR / "scaler_mean_kfold.npy", best_fold_mean)
    np.save(OUTPUT_DIR / "scaler_std_kfold.npy",  best_fold_std)

    print(f"\n── Saved artefacts ──────────────────────────────────────────────")
    print(f"  output/ski_classifier_kfold.pt       (best fold: {int(np.argmax(fold_accuracies)) + 1})")
    print(f"  output/scaler_mean_kfold.npy / scaler_std_kfold.npy")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_all_true, y_all_pred,
        title=f"Aggregated confusion matrix ({N_FOLDS}-fold CV)\nmean acc={mean_acc:.3f} ± {std_acc:.3f}",
        save_name="confusion_matrix_kfold.png",
    )
    plot_fold_curves(all_histories)


if __name__ == "__main__":
    main()
