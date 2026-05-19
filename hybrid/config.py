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

    # bert_weight: масштаб BERT-блока ОТНОСИТЕЛЬНО TF-IDF-блока.
    # 1.0 — TF-IDF и BERT после нормировок дают примерно равный вклад.
    # 2.0-3.0 — BERT доминирует, но TF-IDF ещё ощутим (хорошо для small-data).
    # 5.0+ — BERT доминирует почти полностью (старый дефолт, мешал TF-IDF работать).
    bert_weight: float = 1.0

    # TF-IDF: ngram-границы и min_df.
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    word_min_df: int = 2
    word_max_df: float = 0.98

    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_min_df: int = 2
    char_max_df: float = 0.95

    # Truncated SVD для понижения размерности гибридных векторов перед MLP.
    # 0 — без SVD. 256/512 — типичные значения. На small-data часто помогает.
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
    # Воспроизводимость
    seed: int = 234

    # Обучение
    batch_size: int = 128
    learning_rate: float = 3e-4   # было 1e-4; на маленьких слоях помогает
    patience: int = 8
    weight_decay: float = 1e-2
    epochs: int = 40              # больше эпох + early stop
    min_lr: float = 1e-6

    # Архитектура
    hidden_dim: int = 512
    num_blocks: int = 2
    dropout: float = 0.4          # слегка выше — против переобучения на small-data

    # Лосс
    focal_gamma: float = 1.0      # 1.5 → 1.0: меньше резкости, стабильнее на 8+ классах
    label_smoothing: float = 0.05 # 0.03 → 0.05
    use_class_weight: bool = True

    # Mixup в feature space (на скрытом представлении после input_proj)
    # 0 — выключен. 0.1-0.3 — типично. Помогает при сильном дисбалансе.
    mixup_alpha: float = 0.2

# -----------------------------------------------------------------------------
# Профили HybridMLP под качество входных эмбеддингов
# -----------------------------------------------------------------------------

# "noisy"  — для baseline ruRoberta-large-chunkmean (необучаемая модель).
#            Эмбеддинги шумные → нужна агрессивная регуляризация:
#            mixup, label_smoothing, focal_gamma, class_weight, более высокий dropout.
#
# "clean"  — для custom_embedder (после v3 fix: pooling match, MNR без overlap,
#            normalize+max consistency). Эмбеддинги уже хорошо разделяют классы →
#            всякая регуляризация (mixup, label_smoothing) только мешает.
HYBRID_MLP_PROFILES = {
    "noisy": dict(
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
    ),
    "clean": dict(
        learning_rate=2e-4,
        weight_decay=5e-3,
        epochs=30,
        patience=6,
        hidden_dim=384,    # эмбеддинги уже разделены, меньшая модель быстрее и не переобучается
        num_blocks=1,      # 1 residual block достаточно
        dropout=0.2,       # умеренно
        focal_gamma=0.0,   # обычный CE (focal только мешает, когда классы и так разделимы)
        label_smoothing=0.0,
        mixup_alpha=0.0,   # выключен
        use_class_weight=False,  # эмбеддинги нормированы, class_weight перекосит decision boundary
    ),
}


def hybrid_mlp_config_from_profile(profile: str = "noisy", **overrides) -> HybridMLPConfig:
    """
    Сборка HybridMLPConfig по имени профиля.
    overrides — позволяет переопределить любое поле (например, epochs=50).
    """
    if profile not in HYBRID_MLP_PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}. Available: {list(HYBRID_MLP_PROFILES.keys())}"
        )
    base = HybridMLPConfig()
    profile_overrides = HYBRID_MLP_PROFILES[profile]
    from dataclasses import replace
    cfg = replace(base, **profile_overrides)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg