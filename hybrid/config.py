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
    batch_size: int = 64          # было 128; на 1404 примерах 11 батчей мало → 22 батча
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

    # Новое в v13: warmup и max_grad_norm
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0


# -----------------------------------------------------------------------------
# MLP-профили. Дефолт = noisy (проверенный, даёт 0.51 на custom_embedder).
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
    # экспериментальный — пока с дефолта снят
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
}

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
    Главное правило: custom_embedder — приоритет. На custom лучше всех работает 'noisy'
    (это твой проверенный 0.511). На baseline тоже 'noisy' (для него она и задумана).
    Так что автодетект — это всегда 'noisy', оставлено только для совместимости.
    """
    return DEFAULT_HYBRID_MLP_PROFILE