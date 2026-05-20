import os
import json
import argparse
import warnings
import random
import copy
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    HybridMLPConfig,
    hybrid_mlp_config_from_profile,
    HYBRID_MLP_PROFILES,
    HYBRID_MLP_FEATURE_SOURCES,
    DEFAULT_HYBRID_MLP_FEATURE_SOURCE,
    default_mlp_profile_for_features,
)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    if alpha <= 0.0:
        return features, F.one_hot(targets, num_classes=num_classes).float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(features.size(0), device=features.device)
    mixed_features = lam * features + (1 - lam) * features[idx]
    y_a = F.one_hot(targets, num_classes=num_classes).float()
    y_b = F.one_hot(targets[idx], num_classes=num_classes).float()
    mixed_targets = lam * y_a + (1 - lam) * y_b
    return mixed_features, mixed_targets


def _soft_cross_entropy(logits, soft_targets, class_weight=None):
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weight is not None:
        log_probs = log_probs * class_weight.unsqueeze(0)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _evaluate(model, loader, device, labels_all=None, return_preds=False):
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
        f1 = f1_score(all_true, all_preds, average="macro", zero_division=0, labels=labels_all)
    metrics = {
        "balanced_accuracy": round(float(balanced_accuracy_score(all_true, all_preds)), 6),
        "macro_f1": round(float(f1), 6),
    }
    # v17: дополнительная диагностика — сколько классов модель реально предсказывает
    pred_counter = Counter(all_preds)
    metrics["unique_pred_classes"] = int(len(pred_counter))
    metrics["top1_pred_share"] = round(float(pred_counter.most_common(1)[0][1] / max(1, len(all_preds))), 4)
    if return_preds:
        return metrics, all_true, all_preds
    return metrics


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


# -----------------------------------------------------------------------------
# v17: sanity-checks + лучшее масштабирование фичей
# -----------------------------------------------------------------------------
def _sanitize_features(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        print(f"[hybrid_mlp][WARN] {name}: NaN={n_nan}, Inf={n_inf} — заменяю на 0.0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(X.astype(np.float64) ** 2))) if X.size else 0.0
    print(f"[hybrid_mlp] {name}: shape={X.shape} min={X.min():.4f} "
          f"max={X.max():.4f} mean={X.mean():.4f} rms={rms:.4f}")
    return X


def _rescale_for_mlp(X_train: np.ndarray, X_test: np.ndarray, tag: str):
    """
    v17: после L2-нормы rms ≈ 1/sqrt(dim) ≈ 0.031 для dim=1024.
    Это слишком слабый сигнал для MLP с lr=1e-4: градиенты по входу маленькие,
    bias-член доминирует и модель учит "среднее" — отсюда top1_pred_share≈0.5+.

    Решение: после L2-нормы умножаем на sqrt(dim), чтобы rms ≈ 1.
    Норма сохраняется (все векторы остаются на сфере того же радиуса),
    меняется только масштаб — это эквивалентно скейлингу первого Linear слоя
    в sqrt(dim) раз, что хорошо приживается с дефолтным init.

    Фичи остаются сравнимы между объектами (каждый имеет одну и ту же норму
    sqrt(dim) после умножения), что важно для residual MLP.
    """
    scale = float(np.sqrt(X_train.shape[1]))
    X_train = (X_train * scale).astype(np.float32)
    X_test = (X_test * scale).astype(np.float32)
    rms = float(np.sqrt(np.mean(X_train.astype(np.float64) ** 2)))
    print(f"[hybrid_mlp][v17] {tag}: rescaled by sqrt(dim)={scale:.2f} → rms_train={rms:.3f}")
    return X_train, X_test


def _load_features(vecdir: str, feature_source: str):
    """
    feature_source='bert_only': X_{train,test}_bert.npy (1024-dim dense).
    feature_source='hybrid': X_{train,test}_hybrid.npz (sparse) или X_*_dense.npy (SVD).

    v17: после загрузки/L2-нормы делаем rescale на sqrt(dim).
    """
    if feature_source == "bert_only":
        bert_train = os.path.join(vecdir, "X_train_bert.npy")
        bert_test = os.path.join(vecdir, "X_test_bert.npy")
        if not (os.path.exists(bert_train) and os.path.exists(bert_test)):
            raise FileNotFoundError(
                f"feature_source='bert_only' требует файлов:\n"
                f"  {bert_train}\n"
                f"  {bert_test}\n"
                f"Они создаются на этапе `python -m hybrid.main build`. "
                f"Перестрой векторы или используй --features hybrid."
            )
        X_train = np.load(bert_train).astype(np.float32)
        X_test = np.load(bert_test).astype(np.float32)
        X_train = _sanitize_features(X_train, "X_train_bert(raw)")
        X_test = _sanitize_features(X_test, "X_test_bert(raw)")
        X_train = normalize(X_train, norm="l2", axis=1).astype(np.float32)
        X_test = normalize(X_test, norm="l2", axis=1).astype(np.float32)
        print(f"[hybrid_mlp] features='bert_only' (dense, L2-renorm) dim={X_train.shape[1]}")
        X_train, X_test = _rescale_for_mlp(X_train, X_test, tag="bert_only")
        return X_train, X_test

    if feature_source == "hybrid":
        dense_train = os.path.join(vecdir, "X_train_dense.npy")
        dense_test = os.path.join(vecdir, "X_test_dense.npy")
        if os.path.exists(dense_train) and os.path.exists(dense_test):
            X_train = np.load(dense_train).astype(np.float32)
            X_test = np.load(dense_test).astype(np.float32)
            print(f"[hybrid_mlp] features='hybrid' (DENSE/SVD) dim={X_train.shape[1]}")
        else:
            X_train = sp.load_npz(os.path.join(vecdir, "X_train_hybrid.npz")).astype(np.float32).toarray()
            X_test = sp.load_npz(os.path.join(vecdir, "X_test_hybrid.npz")).astype(np.float32).toarray()
            print(f"[hybrid_mlp] features='hybrid' (SPARSE→DENSE) dim={X_train.shape[1]}")
        X_train = _sanitize_features(X_train, "X_train_hybrid")
        X_test = _sanitize_features(X_test, "X_test_hybrid")
        # v17: hybrid после L2-норм уже имеет rms ≈ 1/sqrt(D_eff). Аналогично rescale.
        rms_tr = float(np.sqrt(np.mean(X_train.astype(np.float64) ** 2)))
        if 0 < rms_tr < 0.5:
            scale = 1.0 / rms_tr
            X_train = (X_train * scale).astype(np.float32)
            X_test = (X_test * scale).astype(np.float32)
            print(f"[hybrid_mlp][v17] hybrid: rms was {rms_tr:.4f} → rescaled by {scale:.2f}")
        return X_train, X_test

    raise ValueError(
        f"Unknown feature_source={feature_source!r}. "
        f"Allowed: {HYBRID_MLP_FEATURE_SOURCES}"
    )


# -----------------------------------------------------------------------------
# v17: smoothed class weights
# -----------------------------------------------------------------------------
def _smoothed_class_weights(y_train_enc: np.ndarray, num_classes: int,
                            power: float = 0.5) -> np.ndarray:
    """
    Standard "balanced" weight: w_c = n_train / (n_classes * count_c).
    Для long-tail (min=1, max=много) это даёт w_c в диапазоне 1..40+ → MLP падает.

    v17: w_c_smoothed = w_c ** power, по умолчанию sqrt. Тогда диапазон сужается,
    но скос в пользу редких классов остаётся.
    """
    raw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train_enc,
    )
    smoothed = np.power(raw, power)
    # Нормируем, чтобы средний вес = 1 (иначе loss scale меняется)
    smoothed = smoothed / smoothed.mean()
    print(f"[hybrid_mlp][v17] class_weights smoothed (power={power}): "
          f"raw[{raw.min():.2f}..{raw.max():.2f}] → "
          f"smoothed[{smoothed.min():.2f}..{smoothed.max():.2f}]")
    return smoothed


def run_mlp(
    vecdir,
    cfg: HybridMLPConfig | None = None,
    epochs: int | None = None,
    device=None,
    feature_source: str = DEFAULT_HYBRID_MLP_FEATURE_SOURCE,
):
    if cfg is None:
        cfg = HybridMLPConfig()
    if feature_source not in HYBRID_MLP_FEATURE_SOURCES:
        raise ValueError(
            f"feature_source must be one of {HYBRID_MLP_FEATURE_SOURCES}, "
            f"got {feature_source!r}"
        )
    n_epochs = int(epochs) if epochs is not None else int(cfg.epochs)
    _seed_everything(cfg.seed)

    X_train, X_test = _load_features(vecdir, feature_source)

    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)
    labels_all = list(range(num_classes))

    if cfg.use_class_weight:
        cw = _smoothed_class_weights(y_train_enc, num_classes, power=0.5)
        class_weights_t = torch.tensor(cw, dtype=torch.float32)
    else:
        class_weights_t = None

    train_ds = DenseDataset(X_train, y_train_enc)
    test_ds = DenseDataset(X_test, y_test_enc)

    device = _resolve_device(device)
    pin = (device.type == "cuda")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin, generator=g, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    model = StrongMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    model_name = f"StrongMLP[{feature_source}]"
    print(f"\n=== {model_name} | dim={X_train.shape[1]} | classes={num_classes} ===")
    print(f"[hybrid_mlp] cfg: lr={cfg.learning_rate} wd={cfg.weight_decay} "
          f"dropout={cfg.dropout} focal_gamma={cfg.focal_gamma} "
          f"ls={cfg.label_smoothing} mixup={cfg.mixup_alpha} "
          f"warmup={cfg.warmup_epochs} epochs={n_epochs} patience={cfg.patience}")

    cw_device = class_weights_t.to(device) if class_weights_t is not None else None
    criterion = FocalLoss(alpha=cw_device, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # v17: исправленный warmup
    warmup_epochs = int(getattr(cfg, "warmup_epochs", 0))
    # Не даём warmup занять больше половины обучения
    max_warmup = max(1, n_epochs // 2)
    if warmup_epochs > max_warmup:
        print(f"[hybrid_mlp][v17] warmup_epochs={warmup_epochs} > n_epochs/2={max_warmup} → "
              f"режу warmup до {max_warmup}")
        warmup_epochs = max_warmup
    base_lr = cfg.learning_rate
    min_lr = cfg.min_lr
    max_grad_norm = float(getattr(cfg, "max_grad_norm", 1.0))

    def _set_lr(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            # v17: warmup стартует с 0.1*base_lr, а не с нуля — нужен хоть какой-то сигнал
            warmup_start = base_lr * 0.1
            warmup_progress = (epoch_idx + 1) / max(1, warmup_epochs)
            warmup_lr = warmup_start + (base_lr - warmup_start) * warmup_progress
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            return
        cosine_epochs = max(1, n_epochs - warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        cos_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cos_lr

    best_f1 = -1.0
    best_metrics = None
    best_epoch = -1
    best_state = None     # v17: запоминаем веса лучшей эпохи
    no_improve = 0
    epoch = -1
    history = []          # v17

    for epoch in range(n_epochs):
        _set_lr(epoch)
        model.train()
        epoch_losses = []
        nan_batches = 0
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

            if not torch.isfinite(loss):
                nan_batches += 1
                continue

            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        metrics = _evaluate(model, test_loader, device, labels_all=labels_all)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        phase = "warmup" if epoch < warmup_epochs else "cosine"
        nan_tag = f" nan_batches={nan_batches}" if nan_batches else ""
        print(f"epoch={epoch + 1} phase={phase} lr={current_lr:.2e} "
              f"train_loss={mean_loss:.6f} "
              f"bacc={metrics['balanced_accuracy']:.4f} f1={metrics['macro_f1']:.4f} "
              f"pred_classes={metrics['unique_pred_classes']}/{num_classes} "
              f"top1_share={metrics['top1_pred_share']}{nan_tag}")

        history.append({
            "epoch": epoch + 1,
            "phase": phase,
            "lr": round(current_lr, 8),
            "train_loss": round(mean_loss, 6),
            **metrics,
        })

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())  # v17
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # v17: восстанавливаем веса лучшей эпохи и пересчитываем метрики
    if best_state is not None:
        model.load_state_dict(best_state)
        final_metrics, y_true, y_pred = _evaluate(
            model, test_loader, device, labels_all=labels_all, return_preds=True,
        )
        print(f"\n[hybrid_mlp][v17] restored best epoch {best_epoch} → {final_metrics}")
        # Per-class breakdown
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels_all, zero_division=0)
        per_class = {
            le.classes_[i]: round(float(per_class_f1[i]), 4)
            for i in range(num_classes)
        }
        n_zero = int(sum(1 for v in per_class.values() if v == 0.0))
        print(f"[hybrid_mlp][v17] per-class f1: zero={n_zero}/{num_classes} классов с f1=0")
    else:
        final_metrics = best_metrics or {}
        per_class = {}

    results = {
        "model": model_name,
        "model_key": "strong_mlp",
        "feature_source": feature_source,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "best_metrics": best_metrics,
        "final_metrics_on_best_weights": final_metrics,  # v17
        "per_class_f1_on_best_weights": per_class,        # v17
        "epochs_ran": epoch + 1,
        "history": history,                               # v17
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
        "warmup_epochs": warmup_epochs,
        "max_grad_norm": max_grad_norm,
        "device": str(device),
        "num_classes": int(num_classes),
        "input_dim": int(X_train.shape[1]),
        "seed": cfg.seed,
        "v17_smoothing_power": 0.5,
        "v17_feature_rescale": "sqrt(dim) for bert_only / 1/rms for hybrid",
    }

    print("\nBEST RESULT")
    print(f"  {model_name}: {best_metrics}")

    vec_base = os.path.basename(os.path.normpath(vecdir)) or "vecdir"
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    out_name = f"mlp-results-{vec_base}-{stamp}.json"
    out_path = os.path.join(vecdir, out_name)
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
    parser.add_argument("--use-class-weight",
                        type=lambda v: str(v).lower() in {"true", "1", "yes"},
                        default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--profile", type=str, default=None,
                        choices=sorted(HYBRID_MLP_PROFILES.keys()),
                        help='Профиль гиперпараметров. По умолчанию выбирается от --features.')
    parser.add_argument(
        "--features", type=str, default=None,
        choices=list(HYBRID_MLP_FEATURE_SOURCES),
        help=f'Источник фич. По умолчанию: "{DEFAULT_HYBRID_MLP_FEATURE_SOURCE}".',
    )
    return parser


def _cfg_from_args(args):
    from dataclasses import replace

    feature_source = getattr(args, "features", None) or DEFAULT_HYBRID_MLP_FEATURE_SOURCE

    if getattr(args, "profile", None):
        profile_name = args.profile
        print(f"[hybrid_mlp] using profile: {profile_name!r} (explicit)")
    else:
        profile_name = default_mlp_profile_for_features(feature_source)
        print(f"[hybrid_mlp] using profile: {profile_name!r} (auto for features={feature_source!r})")

    cfg = hybrid_mlp_config_from_profile(profile_name)

    overrides = {}
    for name in (
        "seed", "batch_size", "learning_rate", "patience", "weight_decay",
        "hidden_dim", "num_blocks", "dropout", "focal_gamma", "label_smoothing",
        "mixup_alpha", "use_class_weight", "warmup_epochs", "max_grad_norm",
    ):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg, feature_source


def main():
    parser = build_argparser()
    args = parser.parse_args()
    print(f"[ARGS] {vars(args)}")
    cfg, feature_source = _cfg_from_args(args)
    run_mlp(args.vecdir, cfg=cfg, epochs=args.epochs, device=args.device,
            feature_source=feature_source)


if __name__ == "__main__":
    main()