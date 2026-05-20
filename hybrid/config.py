from dataclasses import dataclass

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
    seed: int = 234
    batch_size: int = 64
    learning_rate: float = 3e-4
    patience: int = 8
    weight_decay: float = 1e-2
    epochs: int = 40
    min_lr: float = 1e-6

    hidden_dim: int = 512
    num_blocks: int = 2
    dropout: float = 0.4

    focal_gamma: float = 1.0
    label_smoothing: float = 0.05
    use_class_weight: bool = True

    mixup_alpha: float = 0.2

    warmup_epochs: int = 5
    max_grad_norm: float = 1.0


# -----------------------------------------------------------------------------
# MLP-профили.
#   - "noisy": для sparse hybrid (78507-dim), агрессивный (focal+mixup+LS).
#             Это твой проверенный 0.511 на v3.
#   - "clean": лёгкий, для быстрых проверок.
#   - "custom": экспериментальный.
#   - "bert_only" (НОВОЕ): для dense bert-only (1024-dim, L2-norm). Мягкий, без focal/mixup/LS.
#             Гиперпараметры подобраны под распределение фич, на которых классика
#             даёт 0.54+ (LogReg/LinearSVC на bert_only C=0.05).
# -----------------------------------------------------------------------------
HYBRID_MLP_PROFILES = {
    "noisy": dict(
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=8,
        hidden_dim=512,
        num_blocks=2,
        dropout=0.4,
        focal_gamma=1.0,
        label_smoothing=0.05,
        mixup_alpha=0.2,
        use_class_weight=True,
        warmup_epochs=5,
        max_grad_norm=1.0,
    ),
    "clean": dict(
        batch_size=64,
        learning_rate=2e-4,
        weight_decay=5e-3,
        epochs=30,
        patience=6,
        hidden_dim=384,
        num_blocks=1,
        dropout=0.2,
        focal_gamma=0.0,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        use_class_weight=False,
        warmup_epochs=3,
        max_grad_norm=1.0,
    ),
    "custom": dict(
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=10,
        hidden_dim=512,
        num_blocks=2,
        dropout=0.3,
        focal_gamma=0.5,
        label_smoothing=0.03,
        mixup_alpha=0.1,
        use_class_weight=True,
        warmup_epochs=5,
        max_grad_norm=1.0,
    ),
    # НОВОЕ в v15: для bert_only фичей (1024-dim dense, L2-norm).
    # Мягкий профиль: без focal/mixup/LS — данных мало, агрессивная регуляризация
    # ломает обучение. Class_weight=balanced. lr=1e-4 (ниже, чем у noisy, потому
    # что dense L2-norm фичи дают более крутые градиенты).
    "bert_only": dict(
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=5e-3,
        epochs=40,
        patience=10,
        hidden_dim=512,
        num_blocks=2,
        dropout=0.3,
        focal_gamma=0.0,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        use_class_weight=True,
        warmup_epochs=3,
        max_grad_norm=1.0,
    ),
}

# Глобальный дефолт (используется только если features не указано).
# Оставлен "noisy" ради backward compat _cfg_from_args; реальный выбор делает
# default_mlp_profile_for_features в main.py.
DEFAULT_HYBRID_MLP_PROFILE = "noisy"


def hybrid_mlp_config_from_profile(profile: str = DEFAULT_HYBRID_MLP_PROFILE, **overrides) -> HybridMLPConfig:
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
    Оставлено для обратной совместимости. Сейчас выбор профиля делает
    default_mlp_profile_for_features.
    """
    return DEFAULT_HYBRID_MLP_PROFILE


# Допустимые источники фич для MLP.
HYBRID_MLP_FEATURE_SOURCES = ("bert_only", "hybrid")
DEFAULT_HYBRID_MLP_FEATURE_SOURCE = "bert_only"


def default_mlp_profile_for_features(feature_source: str) -> str:
    """
    Дефолтный профиль зависит от источника фич:
      - bert_only → 'bert_only' (мягкий, под dense L2-norm)
      - hybrid    → 'noisy'     (агрессивный, под sparse 78507-dim)
    """
    if feature_source == "bert_only":
        return "bert_only"
    if feature_source == "hybrid":
        return "noisy"
    raise ValueError(f"Unknown feature_source {feature_source!r}")