"""
hybrid/config.py — v20.1 (откат части a)

Возврат HybridMLPConfig к v3-проверенным дефолтам, на которых были метрики:
  [custom_embeder] noisy → balanced_accuracy=0.5115, macro_f1=0.5039
  [default]        noisy → balanced_accuracy=0.2778, macro_f1=0.2688

Главные изменения относительно v19 (v20.1 — после неудачного v20):
  - patience: 8 → 12 (на augmented v19 останавливался на epoch 5-13 — слишком рано)
  - batch_size: ОСТАВЛЕН 128 (v20 пробовал bs=64 — это разломало обучение,
    f1 упал до 0.02. v3 не зря был на 128: на bs=64 при focal+balanced+mixup
    градиенты слишком шумные на 1404 примерах × 36 классов.)
  - Архитектура и лосс не меняются.

Главные изменения относительно v18:
  - DEFAULT_HYBRID_MLP_FEATURE_SOURCE = "hybrid" (было "bert_only" — это и было корнем регрессии)
  - Удалён профиль "bert_only" (lr=1e-4 без mixup/focal — не работал)
  - Удалены поля warmup_epochs (CosineAnnealing без warmup, как в v3)
  - Дефолтный профиль = "noisy" — он и давал 0.5039 на custom

Профили:
  - "noisy" (default): для hybrid sparse фич, агрессивная регуляризация (mixup+focal+LS+balanced)
  - "clean": лёгкий, без mixup/focal, для дополнительных проверок
  - "custom": промежуточный (опционально)

Для feature_source="bert_only" автоматически берётся профиль "clean"
(умеренная регуляризация на dense 1024-dim фичах).
"""

from dataclasses import dataclass

# Единственный источник правды для параметров эмбеддера/MLM — bert_embeddings.config
from bert_embeddings.config import EmbeddingConfig, MLMConfig, ensure_dir  # noqa: F401

_emb = EmbeddingConfig()


@dataclass
class HybridModelConfig:
    base_model_name: str = _emb.base_model_name
    max_length: int = _emb.max_length
    chunk_size: int = _emb.chunk_size
    chunk_overlap: int = _emb.chunk_overlap
    pooling: str = _emb.pooling
    chunk_aggregation: str = _emb.chunk_aggregation
    batch_size: int = _emb.batch_size

    bert_weight: float = 1.0
    bert_weight_with_model_dir: float = 5.0
    bert_weight_base_model: float = 1.0

    disable_bert_scaler: bool = False

    word_ngram_min: int = 1
    word_ngram_max: int = 2
    word_min_df: int = 2
    word_max_df: float = 0.98

    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_min_df: int = 2
    char_max_df: float = 0.95

    svd_components: int = 0


@dataclass
class HybridDataConfig:
    text_col: str = "text"
    label_col: str = "label"


@dataclass
class HybridPathConfig:
    train_file: str = ""
    test_file: str = ""
    output_dir: str = ""


@dataclass
class HybridMLPConfig:
    """
    Дефолтные значения = v3 noisy.
    """
    # Воспроизводимость
    seed: int = 234

    # Обучение
    batch_size: int = 128         # v20.1: оставлен 128 (как v3); bs=64 в v20 ломал обучение
    learning_rate: float = 3e-4
    patience: int = 12            # v20: 8 → 12, чтобы augmented не обрывался рано
    weight_decay: float = 1e-2
    epochs: int = 40
    min_lr: float = 1e-6

    # Архитектура (StrongMLP)
    hidden_dim: int = 512
    num_blocks: int = 2
    dropout: float = 0.4

    # Лосс
    focal_gamma: float = 1.0
    label_smoothing: float = 0.05
    use_class_weight: bool = True

    # Mixup в feature space
    mixup_alpha: float = 0.2

    # Clip
    max_grad_norm: float = 1.0


# -----------------------------------------------------------------------------
# Профили под качество входных эмбеддингов
# -----------------------------------------------------------------------------
# "noisy"  — для hybrid sparse фич (TF-IDF + BERT, 78k-dim) и для custom_embedder
#            (твой проверенный 0.5039 на custom в v3, как раз эти параметры).
#
# "clean"  — лёгкий профиль для быстрых проверок или для bert_only dense фич
#            (1024-dim, L2-norm): без mixup/focal/LS, чтобы не зашумлять
#            уже хорошо разделимое представление.
#
# "custom" — промежуточный: чуть мягче "noisy", оставлен для совместимости.
# -----------------------------------------------------------------------------
HYBRID_MLP_PROFILES = {
    "noisy": dict(
        batch_size=128,  # v20.1: оставлен 128 (v3-проверенный)
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=12,     # v20: 8 → 12
        hidden_dim=512,
        num_blocks=2,
        dropout=0.4,
        focal_gamma=1.0,
        label_smoothing=0.05,
        mixup_alpha=0.2,
        use_class_weight=True,
        max_grad_norm=1.0,
    ),
    "clean": dict(
        batch_size=128,
        learning_rate=2e-4,
        weight_decay=5e-3,
        epochs=30,
        patience=10,     # v20: 6 → 10 (синхронно с noisy)
        hidden_dim=384,
        num_blocks=1,
        dropout=0.2,
        focal_gamma=0.0,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        use_class_weight=False,
        max_grad_norm=1.0,
    ),
    "custom": dict(
        batch_size=128,  # v20.1: синхронно с noisy (откат)
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=12,     # v20: 10 → 12
        hidden_dim=512,
        num_blocks=2,
        dropout=0.3,
        focal_gamma=0.5,
        label_smoothing=0.03,
        mixup_alpha=0.1,
        use_class_weight=True,
        max_grad_norm=1.0,
    ),
}

DEFAULT_HYBRID_MLP_PROFILE = "noisy"


def hybrid_mlp_config_from_profile(profile: str = DEFAULT_HYBRID_MLP_PROFILE, **overrides) -> HybridMLPConfig:
    """
    Сборка HybridMLPConfig по имени профиля.
    overrides — позволяет переопределить любое поле (например, epochs=50).
    """
    if profile not in HYBRID_MLP_PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}. Available: {list(HYBRID_MLP_PROFILES.keys())}"
        )
    from dataclasses import replace
    base = HybridMLPConfig()
    profile_overrides = HYBRID_MLP_PROFILES[profile]
    cfg = replace(base, **profile_overrides)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def autodetect_mlp_profile(vecdir: str) -> str:
    """
    Оставлено ради обратной совместимости. Сейчас выбор делает
    default_mlp_profile_for_features.
    """
    return DEFAULT_HYBRID_MLP_PROFILE


# -----------------------------------------------------------------------------
# Допустимые источники фич для MLP
# -----------------------------------------------------------------------------
HYBRID_MLP_FEATURE_SOURCES = ("hybrid", "bert_only")

# v19: дефолт возвращён к "hybrid" (как в v3, где получались 0.5039).
# В v18 был "bert_only" — именно это и приводило к коллапсу.
DEFAULT_HYBRID_MLP_FEATURE_SOURCE = "hybrid"


def default_mlp_profile_for_features(feature_source: str) -> str:
    """
    Дефолтный профиль зависит от источника фич:
      - hybrid    → 'noisy' (агрессивная регуляризация под sparse 78k-dim, v3-проверенный)
      - bert_only → 'clean' (мягкий, под dense L2-norm 1024-dim)
    """
    if feature_source == "hybrid":
        return "noisy"
    if feature_source == "bert_only":
        return "clean"
    raise ValueError(f"Unknown feature_source {feature_source!r}")