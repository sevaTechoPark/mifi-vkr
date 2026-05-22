"""
Конфигурации гибридного пайплайна (TF-IDF + BERT).

Модуль содержит датаклассы для трёх стадий пайплайна и набор профилей
гиперпараметров для MLP-классификатора:

  * HybridModelConfig — параметры построения признаков (TF-IDF, BERT,
    взвешивание BERT-блока, опциональный TruncatedSVD).
  * HybridDataConfig  — имена колонок входного CSV.
  * HybridPathConfig  — пути входных и выходных файлов.
  * HybridMLPConfig   — гиперпараметры обучения MLP-классификатора.

Параметры эмбеддера (base_model_name, max_length, chunk_size и т. д.)
импортируются из bert_embeddings.config — это единый источник правды
для всего проекта.
"""

from dataclasses import dataclass

from bert_embeddings.config import EmbeddingConfig, MLMConfig, ensure_dir  # noqa: F401

_emb = EmbeddingConfig()


@dataclass
class HybridModelConfig:
    """Параметры построения гибридного признакового пространства."""

    # Параметры эмбеддера (наследуются из bert_embeddings)
    base_model_name: str = _emb.base_model_name
    max_length: int = _emb.max_length
    chunk_size: int = _emb.chunk_size
    chunk_overlap: int = _emb.chunk_overlap
    pooling: str = _emb.pooling
    chunk_aggregation: str = _emb.chunk_aggregation
    batch_size: int = _emb.batch_size

    # Вес BERT-блока в гибридном векторе.
    # Активный bert_weight подменяется на одно из двух значений ниже в
    # зависимости от того, передан model_dir или используется базовая модель.
    bert_weight: float = 1.0
    bert_weight_with_model_dir: float = 5.0
    bert_weight_base_model: float = 1.0

    # Отключение StandardScaler для уже L2-нормированных эмбеддингов.
    disable_bert_scaler: bool = False

    # TF-IDF (word n-grams)
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    word_min_df: int = 2
    word_max_df: float = 0.98

    # TF-IDF (char_wb n-grams)
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_min_df: int = 2
    char_max_df: float = 0.95

    # Опциональное понижение размерности через TruncatedSVD.
    # 0 — отключено, > 0 — число компонент.
    svd_components: int = 0


@dataclass
class HybridDataConfig:
    """Имена колонок текста и метки во входных CSV."""

    text_col: str = "text"
    label_col: str = "label"


@dataclass
class HybridPathConfig:
    """Пути входных и выходных файлов."""

    train_file: str = ""
    test_file: str = ""
    output_dir: str = ""


@dataclass
class HybridMLPConfig:
    """
    Гиперпараметры MLP-классификатора над гибридными признаками.

    Дефолты соответствуют профилю ``noisy`` — агрессивная регуляризация
    (focal loss + mixup + label smoothing + balanced class weights),
    рассчитанная на разреженный высокоразмерный гибридный вектор.
    """

    # Воспроизводимость
    seed: int = 234

    # Параметры обучения
    batch_size: int = 128
    learning_rate: float = 3e-4
    patience: int = 12
    weight_decay: float = 1e-2
    epochs: int = 40
    min_lr: float = 1e-6

    # Архитектура StrongMLP
    hidden_dim: int = 512
    num_blocks: int = 2
    dropout: float = 0.4

    # Лосс
    focal_gamma: float = 1.0
    label_smoothing: float = 0.05
    use_class_weight: bool = True

    # Mixup в пространстве признаков
    mixup_alpha: float = 0.2

    # Ограничение нормы градиента
    max_grad_norm: float = 1.0


# -----------------------------------------------------------------------------
# Профили гиперпараметров под качество входных эмбеддингов
# -----------------------------------------------------------------------------
# noisy  — для разреженных гибридных признаков (TF-IDF + BERT);
#          включена агрессивная регуляризация.
# clean  — лёгкий профиль для плотных уже L2-нормированных представлений
#          (например, чистые BERT-эмбеддинги): без mixup, focal и
#          label smoothing.
# custom — промежуточный профиль с умеренной регуляризацией.
# -----------------------------------------------------------------------------
HYBRID_MLP_PROFILES = {
    "noisy": dict(
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=12,
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
        patience=10,
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
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-2,
        epochs=40,
        patience=12,
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


def hybrid_mlp_config_from_profile(
    profile: str = DEFAULT_HYBRID_MLP_PROFILE,
    **overrides,
) -> HybridMLPConfig:
    """
    Собрать ``HybridMLPConfig`` по имени профиля.

    Дополнительные именованные аргументы переопределяют отдельные поля
    конфигурации (например, ``epochs=50``).
    """
    if profile not in HYBRID_MLP_PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}. "
            f"Available: {list(HYBRID_MLP_PROFILES.keys())}"
        )
    from dataclasses import replace

    base = HybridMLPConfig()
    profile_overrides = HYBRID_MLP_PROFILES[profile]
    cfg = replace(base, **profile_overrides)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


# -----------------------------------------------------------------------------
# Допустимые источники признаков для MLP
# -----------------------------------------------------------------------------
HYBRID_MLP_FEATURE_SOURCES = ("hybrid", "bert_only")

DEFAULT_HYBRID_MLP_FEATURE_SOURCE = "hybrid"


def default_mlp_profile_for_features(feature_source: str) -> str:
    """
    Подобрать профиль гиперпараметров по типу источника признаков.

    Для гибридных разреженных признаков используется ``noisy``, для
    плотных BERT-эмбеддингов — ``clean``.
    """
    if feature_source == "hybrid":
        return "noisy"
    if feature_source == "bert_only":
        return "clean"
    raise ValueError(f"Unknown feature_source {feature_source!r}")
