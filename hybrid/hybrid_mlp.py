import os
import json
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .config import HybridMLPConfig


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


class DenseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return residual + out


class StrongMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, num_blocks: int, dropout: float):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 1.5, label_smoothing: float = 0.03):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def _evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    return {
        "balanced_accuracy": round(balanced_accuracy_score(all_true, all_preds), 6),
        "macro_f1": round(f1_score(all_true, all_preds, average="macro", zero_division=0), 6),
    }


def _resolve_device(device) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    name = str(device).strip().lower()
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    return torch.device(name)


def run_mlp(vecdir, cfg: HybridMLPConfig | None = None, epochs: int | None = None, device=None):
    """
    Обучение StrongMLP на гибридных векторах.

    cfg     — HybridMLPConfig (если None — берётся дефолт).
    epochs  — если передан, переопределяет cfg.epochs (backward compatibility со старым CLI).
    device  — "cuda" / "mps" / "cpu" / torch.device / None (autodetect).
    """
    if cfg is None:
        cfg = HybridMLPConfig()

    n_epochs = int(epochs) if epochs is not None else int(cfg.epochs)

    _set_seed(cfg.seed)

    X_train = sp.load_npz(os.path.join(vecdir, "X_train_hybrid.npz")).astype(np.float32)
    X_test = sp.load_npz(os.path.join(vecdir, "X_test_hybrid.npz")).astype(np.float32)

    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(le.classes_)),
        y=y_train_enc,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    train_ds = DenseDataset(X_train, y_train_enc)
    test_ds = DenseDataset(X_test, y_test_enc)

    device = _resolve_device(device)
    use_cuda = (device.type == "cuda")
    pin = bool(use_cuda)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    model = StrongMLP(
        input_dim=X_train.shape[1],
        num_classes=len(le.classes_),
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    class_weights = class_weights.to(device)
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-5,
    )

    best_f1 = -1.0
    best_metrics = None
    best_epoch = -1
    no_improve = 0
    epoch = -1

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

        metrics = _evaluate(model, test_loader, device)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch={epoch + 1} "
            f"lr={current_lr:.2e} "
            f"train_loss={mean_loss:.6f} "
            f"metrics={metrics}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    results = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "best_metrics": best_metrics,
        "epochs_ran": epoch + 1,
        "batch_size": cfg.batch_size,
        "lr": cfg.learning_rate,
        "patience": cfg.patience,
        "weight_decay": cfg.weight_decay,
        "hidden_dim": cfg.hidden_dim,
        "num_blocks": cfg.num_blocks,
        "dropout": cfg.dropout,
        "focal_gamma": cfg.focal_gamma,
        "label_smoothing": cfg.label_smoothing,
        "device": str(device),
        "num_classes": int(len(le.classes_)),
        "input_dim": int(X_train.shape[1]),
        "seed": cfg.seed,
    }

    print("\nBEST RESULT")
    print(best_metrics)
    json.dumps(results, ensure_ascii=False, indent=2)
    return results


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)

    # Гиперпараметры MLP — каждый из CLI может переопределить cfg
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)

    return parser


def _cfg_from_args(args) -> HybridMLPConfig:
    from dataclasses import replace
    cfg = HybridMLPConfig()

    overrides = {}
    for name in (
        "seed", "batch_size", "learning_rate", "patience", "weight_decay",
        "hidden_dim", "num_blocks", "dropout", "focal_gamma", "label_smoothing",
    ):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val

    return replace(cfg, **overrides) if overrides else cfg


def main():
    parser = build_argparser()
    args = parser.parse_args()
    print(f"[ARGS] {vars(args)}")

    cfg = _cfg_from_args(args)
    run_mlp(args.vecdir, cfg=cfg, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()