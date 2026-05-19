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

    # Два дефолта: для кастомного triplet-обученного эмбеддера (model_dir задан)
    # имеет смысл сильнее весить BERT-блок, потому что эмбеддинги уже линейно разделимы.
    # Для baseline ruRoberta (model_dir не задан) TF-IDF реально помогает, и bw=1.0 ок.
    # Эти дефолты применяются в hybrid/main.py при build, если пользователь явно
    # не передал --bert-weight в CLI.
    bert_weight: float = 1.0  # legacy-поле, оставляем для backward compat
    bert_weight_with_model_dir: float = 5.0
    bert_weight_base_model: float = 1.0

    # Если True — отключает StandardScaler по BERT-блоку. Включается автоматически
    # для custom embedder, потому что его выходы уже L2-нормированы (triplet/MNR
    # обучение специально подгоняет их на единичную сферу). StandardScaler в этом
    # случае ломает структуру и убивает classical-метрики.
    disable_bert_scaler: bool = False

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
# Профили HybridMLP
# -----------------------------------------------------------------------------
#
# Главное правило: custom_embedder — приоритет. Дефолты настраиваем под него.
#
# "custom" (DEFAULT) — для custom_embedder после triplet/MNR.
#                Эмбеддинги полу-разделимы, нужна МЯГКАЯ регуляризация и достаточно
#                ёмкая голова, чтобы вытянуть 36 классов на 1400-train.
#
# "clean"  — экспериментальный лёгкий профиль (1 блок, без регуляризации).
#            На практике на 36 классах он недо-фитит, оставлен для исследований.
#
# "noisy"  — для baseline ruRoberta (raw mean-pool, без triplet).
#            Эмбеддинги шумные → агрессивная регуляризация (mixup, focal,
#            label_smoothing, class_weight, dropout 0.4).

HYBRID_MLP_PROFILES = {
    "custom": dict(
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=10,
        hidden_dim=512,
        num_blocks=2,
        dropout=0.3,            # умеренный
        focal_gamma=0.5,        # лёгкий focal, помогает на 36-классовом дисбалансе
        label_smoothing=0.03,
        mixup_alpha=0.1,        # лёгкий mixup
        use_class_weight=True,  # классы дисбалансные (min_class=1)
    ),
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
        hidden_dim=384,
        num_blocks=1,
        dropout=0.2,
        focal_gamma=0.0,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        use_class_weight=False,
    ),
}

# Дефолтный профиль для всего модуля
DEFAULT_HYBRID_MLP_PROFILE = "custom"


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
    Подбирает MLP-профиль по meta.json в vecdir.
    Логика:
      - если model_dir задан (custom_embedder)  → "custom"
      - иначе (baseline ruRoberta raw mean-pool) → "noisy"
    """
    import json, os as _os
    meta_path = _os.path.join(vecdir, "meta.json")
    if not _os.path.exists(meta_path):
        return DEFAULT_HYBRID_MLP_PROFILE
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return "custom" if meta.get("model_dir") else "noisy"
    except Exception:
        return DEFAULT_HYBRID_MLP_PROFILE