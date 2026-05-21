"""
hybrid/hybrid_mlp.py — v20

Возврат к проверенной v3-архитектуре (StrongMLP + ResidualBlock + Mixup в feature space
+ FocalLoss + balanced class_weight + CosineAnnealingLR без warmup), на которой
исторически были метрики:
  [custom_embeder] noisy  → balanced_accuracy=0.5115, macro_f1=0.5039
  [custom_embeder] clean  → balanced_accuracy=0.4207, macro_f1=0.4563
  [default]        noisy  → balanced_accuracy=0.2778, macro_f1=0.2688
  [default]        clean  → balanced_accuracy=0.2813, macro_f1=0.3287

Что выкинуто относительно v18 (после диагностики — это давало коллапс в majority-class):
  - SimpleHeadMLP (270K parameters) — слабая ёмкость для 36 классов, в проде валится в 1-5 классов
  - warmup (5 эпох) перед cosine с пиком 3e-4 — на 22 батчах epoch_length=22 «прыгает» loss с 8→3→2
  - retry-логика (3 попытки с lr×0.333 и lr×0.1) — лечила симптом, не причину
  - smoothed class_weights (cw**0.5) — искажает balanced
  - sqrt(dim) feature rescale для bert_only — не нужно при правильном scheduler

Что сохранено из v15+ (полезные обёртки, не влияющие на качество):
  - feature_source ("hybrid" | "bert_only") — выбор источника фич
  - Имя файла: mlp-results-<basename(vecdir)>-<YYYY-MM-DDTHH-MM>.json
  - Pretty-имя модели в логе: "MLP-StrongMLP[<features>] profile=<name>"
  - --features CLI-аргумент
  - В JSON сохраняется per_class_f1 на лучшей эпохе

Дефолтный feature_source = "hybrid" (как в v3, где и получались 0.5039).
"""

import os
import json
import argparse
import warnings
from datetime import datetime

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

from .config import (
    HybridMLPConfig,
    hybrid_mlp_config_from_profile,
    HYBRID_MLP_PROFILES,
    HYBRID_MLP_FEATURE_SOURCES,
    DEFAULT_HYBRID_MLP_FEATURE_SOURCE,
    default_mlp_profile_for_features,
)


# ----------------------------------------------------------------------------- 
# Seeding
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class DenseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Архитектура — v3 StrongMLP (input_proj → N×ResidualBlock → head)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# FocalLoss + Mixup в feature-space (как в v3)
# -----------------------------------------------------------------------------
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
        return features, F.one_hot(targets, num_classes=num_classes).float()

    lam = float(np.random.beta(alpha, alpha))
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


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def _evaluate(model, loader, device, labels_all=None, return_per_class=False):
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
        per_class = None
        if return_per_class:
            per_class = f1_score(
                all_true, all_preds,
                average=None, zero_division=0,
                labels=labels_all,
            )
    metrics = {
        "balanced_accuracy": round(float(balanced_accuracy_score(all_true, all_preds)), 6),
        "macro_f1": round(float(f1), 6),
    }
    if return_per_class:
        return metrics, [float(round(x, 6)) for x in per_class.tolist()]
    return metrics


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
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
# Загрузка фич (hybrid TF-IDF+BERT либо только BERT)
# -----------------------------------------------------------------------------
def _load_features(vecdir: str, feature_source: str):
    """
    feature_source:
      - "hybrid"    → X_train_dense.npy (если есть, после SVD) или X_train_hybrid.npz (TF-IDF+BERT)
      - "bert_only" → X_train_bert.npy / X_test_bert.npy (только BERT-эмбеддинги, 1024-dim)
    """
    if feature_source == "bert_only":
        ftr = os.path.join(vecdir, "X_train_bert.npy")
        fte = os.path.join(vecdir, "X_test_bert.npy")
        if not (os.path.exists(ftr) and os.path.exists(fte)):
            raise FileNotFoundError(
                f"feature_source='bert_only' требует файлов:\n"
                f"  {ftr}\n  {fte}\n"
                f"Их не нашёл. Используй feature_source='hybrid' или сгенерируй bert-only фичи."
            )
        X_train = np.load(ftr).astype(np.float32)
        X_test = np.load(fte).astype(np.float32)
        print(f"[hybrid_mlp] using BERT-ONLY features: dim={X_train.shape[1]}")
        return X_train, X_test

    if feature_source == "hybrid":
        dense_train = os.path.join(vecdir, "X_train_dense.npy")
        dense_test = os.path.join(vecdir, "X_test_dense.npy")
        if os.path.exists(dense_train) and os.path.exists(dense_test):
            X_train = np.load(dense_train).astype(np.float32)
            X_test = np.load(dense_test).astype(np.float32)
            print(f"[hybrid_mlp] using DENSE (SVD) hybrid features: dim={X_train.shape[1]}")
            return X_train, X_test
        sparse_train = os.path.join(vecdir, "X_train_hybrid.npz")
        sparse_test = os.path.join(vecdir, "X_test_hybrid.npz")
        if not (os.path.exists(sparse_train) and os.path.exists(sparse_test)):
            raise FileNotFoundError(
                f"feature_source='hybrid' требует файлов:\n"
                f"  {sparse_train} или {dense_train}\n"
                f"  {sparse_test} или {dense_test}"
            )
        X_train = sp.load_npz(sparse_train).astype(np.float32).toarray()
        X_test = sp.load_npz(sparse_test).astype(np.float32).toarray()
        print(f"[hybrid_mlp] using SPARSE→DENSE hybrid features: dim={X_train.shape[1]}")
        return X_train, X_test

    raise ValueError(
        f"Unknown feature_source={feature_source!r}. "
        f"Allowed: {list(HYBRID_MLP_FEATURE_SOURCES)}"
    )


# -----------------------------------------------------------------------------
# Главная функция: один запуск (v3-логика, без retry)
# -----------------------------------------------------------------------------
def run_mlp(
    vecdir,
    cfg: HybridMLPConfig | None = None,
    epochs: int | None = None,
    device=None,
    feature_source: str = DEFAULT_HYBRID_MLP_FEATURE_SOURCE,
    simple_head: bool | None = None,  # принимается ради обратной совместимости, игнорируется
):
    """
    Один прогон MLP по v3-схеме.

    feature_source:
      - "hybrid"    (дефолт) — TF-IDF+BERT, hidden_dim≈1024 для custom, ~78к для baseline
      - "bert_only" — только BERT-эмбеддинги (1024-dim)

    simple_head игнорируется (был в v18, выкинут как нерабочий).
    """
    if simple_head is not None:
        # тихое предупреждение, чтобы не ломать старые вызовы
        print(f"[hybrid_mlp] note: simple_head={simple_head} в v19 игнорируется "
              f"(возвращена единая архитектура StrongMLP).")

    if cfg is None:
        cfg = HybridMLPConfig()
    if feature_source not in HYBRID_MLP_FEATURE_SOURCES:
        raise ValueError(
            f"Unknown feature_source={feature_source!r}. "
            f"Allowed: {list(HYBRID_MLP_FEATURE_SOURCES)}"
        )

    n_epochs = int(epochs) if epochs is not None else int(cfg.epochs)
    _set_seed(cfg.seed)

    # --- 1. данные
    X_train, X_test = _load_features(vecdir, feature_source)

    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)
    labels_all = list(range(num_classes))

    # --- 2. class weights (balanced, как в v3 — БЕЗ smoothing)
    if cfg.use_class_weight:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_train_enc,
        )
        class_weights_t = torch.tensor(cw, dtype=torch.float32)
    else:
        class_weights_t = None

    # --- 3. loaders
    train_ds = DenseDataset(X_train, y_train_enc)
    test_ds = DenseDataset(X_test, y_test_enc)

    device = _resolve_device(device)
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    # --- 4. модель / loss / optimizer / scheduler (всё как v3)
    model = StrongMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    pretty = f"MLP-StrongMLP[{feature_source}]"
    print(f"[hybrid_mlp] {pretty} | input_dim={X_train.shape[1]} num_classes={num_classes} "
          f"params={n_params:,}")
    print(f"[hybrid_mlp] hidden_dim={cfg.hidden_dim} blocks={cfg.num_blocks} dropout={cfg.dropout} "
          f"lr={cfg.learning_rate} wd={cfg.weight_decay} bs={cfg.batch_size} "
          f"focal_gamma={cfg.focal_gamma} ls={cfg.label_smoothing} mixup={cfg.mixup_alpha} "
          f"class_weight={cfg.use_class_weight}")

    cw_device = class_weights_t.to(device) if class_weights_t is not None else None
    criterion = FocalLoss(
        alpha=cw_device,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # ВАЖНО: cosine annealing без warmup — как в v3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=cfg.min_lr,
    )

    # --- 5. обучение
    best_f1 = -1.0
    best_metrics = None
    best_per_class = None
    best_epoch = -1
    no_improve = 0
    epoch = -1
    history = []

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # на лучшей эпохе попросим ещё per-class f1
        metrics, per_class = _evaluate(
            model, test_loader, device, labels_all=labels_all, return_per_class=True,
        )
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch + 1:>3} lr={current_lr:.2e} train_loss={mean_loss:.6f} metrics={metrics}")

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": round(mean_loss, 6),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        })

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            best_per_class = per_class
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    per_class_f1_dict = None
    if best_per_class is not None:
        per_class_f1_dict = {
            str(cls): score for cls, score in zip(le.classes_, best_per_class)
        }

    results = {
        "model_name": pretty,
        "feature_source": feature_source,
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
        "max_grad_norm": cfg.max_grad_norm,
        "device": str(device),
        "num_classes": int(num_classes),
        "input_dim": int(X_train.shape[1]),
        "seed": cfg.seed,
        "n_params": int(n_params),
        "v20_notes": "v19 + batch_size 128→64 + patience 8→12 (стабилизация mixup, без обрыва на augmented)",
        "per_class_f1_on_best": per_class_f1_dict,
        "history": history,
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
            "max_grad_norm": cfg.max_grad_norm,
        },
    }

    print("\nBEST RESULT")
    print(f"  {pretty}: {best_metrics}  (epoch {best_epoch})")

    vec_base = os.path.basename(os.path.normpath(vecdir)) or "vecdir"
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    out_name = f"mlp-results-{vec_base}-{stamp}.json"
    out_path = os.path.join(vecdir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {out_path}")
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
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
    parser.add_argument(
        "--use-class-weight",
        type=lambda v: str(v).lower() in {"true", "1", "yes"},
        default=None,
    )
    parser.add_argument("--max-grad-norm", type=float, default=None)
    # warmup-epochs принимаем для обратной совместимости, но игнорируем
    parser.add_argument(
        "--warmup-epochs", type=int, default=None,
        help="DEPRECATED в v19 — игнорируется (CosineAnnealing без warmup как в v3).",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        choices=sorted(HYBRID_MLP_PROFILES.keys()),
        help='Профиль гиперпараметров. По умолчанию выбирается от --features.',
    )
    parser.add_argument(
        "--features", type=str, default=None,
        choices=list(HYBRID_MLP_FEATURE_SOURCES),
        help=f'Источник фич. По умолчанию: "{DEFAULT_HYBRID_MLP_FEATURE_SOURCE}".',
    )
    # принимаем ради совместимости вызова из main.py, в v19 игнорируется
    parser.add_argument(
        "--simple-head",
        type=lambda v: str(v).lower() in {"true", "1", "yes"},
        default=None,
        help="DEPRECATED в v19 — игнорируется (SimpleHeadMLP убран).",
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
        "mixup_alpha", "use_class_weight", "max_grad_norm",
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
    run_mlp(
        args.vecdir,
        cfg=cfg,
        epochs=args.epochs,
        device=args.device,
        feature_source=feature_source,
        simple_head=getattr(args, "simple_head", None),
    )


if __name__ == "__main__":
    main()