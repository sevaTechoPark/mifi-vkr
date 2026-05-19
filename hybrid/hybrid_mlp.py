import os
import json
import argparse
import warnings

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .config import HybridMLPConfig, hybrid_mlp_config_from_profile, HYBRID_MLP_PROFILES

def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CuDNN детерминизм (включается опционально, может замедлить)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    def __init__(self, input_dim, num_classes, hidden_dim, num_blocks, dropout):
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

    def forward_features(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        return self.head(features)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def _mixup_features(features, targets, alpha, num_classes):
    """Mixup на скрытом представлении (после input_proj). Возвращает (mixed_features, soft_targets)."""
    if alpha <= 0.0:
        # one-hot без смешивания
        return features, F.one_hot(targets, num_classes=num_classes).float()

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(features.size(0), device=features.device)

    mixed_features = lam * features + (1 - lam) * features[idx]
    y_a = F.one_hot(targets, num_classes=num_classes).float()
    y_b = F.one_hot(targets[idx], num_classes=num_classes).float()
    mixed_targets = lam * y_a + (1 - lam) * y_b
    return mixed_features, mixed_targets


def _soft_cross_entropy(logits, soft_targets, class_weight=None):
    """CE для soft-targets (после mixup). class_weight применяется per-class к log-prob."""
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weight is not None:
        log_probs = log_probs * class_weight.unsqueeze(0)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _evaluate(model, loader, device, labels_all=None):
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        f1 = f1_score(
            all_true, all_preds,
            average="macro", zero_division=0,
            labels=labels_all,
        )
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(all_true, all_preds)), 6),
        "macro_f1": round(float(f1), 6),
    }


def _resolve_device(device):
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
    if cfg is None:
        cfg = HybridMLPConfig()
    n_epochs = int(epochs) if epochs is not None else int(cfg.epochs)
    _seed_everything(cfg.seed)

    # Если есть dense-версия (после SVD) — берём её, она быстрее и часто лучше для MLP.
    dense_train = os.path.join(vecdir, "X_train_dense.npy")
    dense_test = os.path.join(vecdir, "X_test_dense.npy")
    if os.path.exists(dense_train) and os.path.exists(dense_test):
        X_train = np.load(dense_train).astype(np.float32)
        X_test = np.load(dense_test).astype(np.float32)
        print(f"[hybrid_mlp] using DENSE (SVD) features: dim={X_train.shape[1]}")
    else:
        X_train = sp.load_npz(os.path.join(vecdir, "X_train_hybrid.npz")).astype(np.float32).toarray()
        X_test = sp.load_npz(os.path.join(vecdir, "X_test_hybrid.npz")).astype(np.float32).toarray()
        print(f"[hybrid_mlp] using SPARSE→DENSE hybrid features: dim={X_train.shape[1]}")

    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)
    labels_all = list(range(num_classes))

    if cfg.use_class_weight:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_train_enc,
        )
        class_weights_t = torch.tensor(cw, dtype=torch.float32)
    else:
        class_weights_t = None

    train_ds = DenseDataset(X_train, y_train_enc)
    test_ds = DenseDataset(X_test, y_test_enc)

    device = _resolve_device(device)
    pin = (device.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = StrongMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    cw_device = class_weights_t.to(device) if class_weights_t is not None else None
    criterion = FocalLoss(alpha=cw_device, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=cfg.min_lr)

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

            if cfg.mixup_alpha > 0.0:
                features = model.forward_features(xb)
                mixed_features, soft_targets = _mixup_features(
                    features, yb, alpha=cfg.mixup_alpha, num_classes=num_classes,
                )
                logits = model.head(mixed_features)
                loss = _soft_cross_entropy(logits, soft_targets, class_weight=cw_device)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        metrics = _evaluate(model, test_loader, device, labels_all=labels_all)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch + 1} lr={current_lr:.2e} train_loss={mean_loss:.6f} metrics={metrics}")

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
        "mixup_alpha": cfg.mixup_alpha,
        "use_class_weight": cfg.use_class_weight,
        "device": str(device),
        "num_classes": int(num_classes),
        "input_dim": int(X_train.shape[1]),
        "seed": cfg.seed,
        "config_full": {
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "hidden_dim": cfg.hidden_dim,
            "num_blocks": cfg.num_blocks,
            "dropout": cfg.dropout,
            "focal_gamma": cfg.focal_gamma,
            "label_smoothing": cfg.label_smoothing,
            "mixup_alpha": cfg.mixup_alpha,
            "use_class_weight": cfg.use_class_weight,
            "batch_size": cfg.batch_size,
        },
    }

    print("\nBEST RESULT")
    print(best_metrics)

    # сохраняем результаты MLP
    out_path = os.path.join(vecdir, "mlp_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {out_path}")

    return results


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
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
    parser.add_argument("--mixup-alpha", type=float, default=None)
    parser.add_argument("--use-class-weight", type=lambda v: str(v).lower() in {"true","1","yes"}, default=None)
    parser.add_argument("--profile", type=str, default=None, choices=sorted(HYBRID_MLP_PROFILES.keys()), help='Профиль гиперпараметров: "noisy" (для baseline) или "clean" (для custom_embedder).')
    return parser


def _cfg_from_args(args):
    from dataclasses import replace

    # 1) База: либо профиль, либо обычный HybridMLPConfig()
    if getattr(args, "profile", None):
        cfg = hybrid_mlp_config_from_profile(args.profile)
        print(f"[hybrid_mlp] using profile: {args.profile!r}")
    else:
        cfg = HybridMLPConfig()

    # 2) CLI-оверрайды поверх профиля
    overrides = {}
    for name in (
        "seed", "batch_size", "learning_rate", "patience", "weight_decay",
        "hidden_dim", "num_blocks", "dropout", "focal_gamma", "label_smoothing",
        "mixup_alpha", "use_class_weight",
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