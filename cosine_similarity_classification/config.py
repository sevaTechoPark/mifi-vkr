"""
Конфиг модуля cosine_similarity_classification.

Источник правды по гиперпараметрам эмбеддера — bert_embeddings.config.EmbeddingConfig.
"""

from bert_embeddings.config import EmbeddingConfig

_emb = EmbeddingConfig()

BASE_MODEL_NAME: str = _emb.base_model_name

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

# Дефолтный метод. Доступно: "centroid" | "nearest" | "centroid_nn"
METHOD: str = "centroid"

MAX_LENGTH: int = _emb.max_length
CHUNK_SIZE: int = _emb.chunk_size
CHUNK_OVERLAP: int = _emb.chunk_overlap

POOLING: str = _emb.pooling
CHUNK_AGGREGATION: str = _emb.chunk_aggregation

BATCH_SIZE: int = _emb.batch_size

DEVICE: str | None = None
MODEL_DIR: str = ""

# -----------------------------------------------------------------------------
# kNN soft-vote
# -----------------------------------------------------------------------------
KNN_K: int = 5
KNN_TEMPERATURE: float = 0.1
# Sweep по k: "1,3,5,7,9,11,15"; пустая строка → один k из KNN_K
KNN_K_SWEEP: str = "1,3,5,7,9,11,15"
# Sweep по температуре: "0.05,0.1,0.2,0.3"; пустая → один T из KNN_TEMPERATURE
KNN_T_SWEEP: str = "0.05,0.1,0.2"

# -----------------------------------------------------------------------------
# Centroid: trim + soft-trim + refinement
# -----------------------------------------------------------------------------
# CENTROID_TRIM_RATIO=0.15 — доля «дальних» точек, отбрасываемых перед усреднением.
# CENTROID_TRIM_MODE — "hard" (старое поведение, drop трим%) или "soft" (вес = sim^TRIM_POWER).
# CENTROID_REFINE_ITERS — число итераций пересчёта (1 = одна доп. итерация после init).
CENTROID_TRIM_RATIO: float = 0.15
CENTROID_TRIM_MODE: str = "soft"   # "hard" | "soft"
CENTROID_TRIM_POWER: float = 4.0   # для soft-trim: вес = max(sim, 0)^power
CENTROID_REFINE_ITERS: int = 1     # итеративный пересчёт центроидов

# -----------------------------------------------------------------------------
# centroid_nn ensemble
# -----------------------------------------------------------------------------
# Финальный score = alpha * centroid_score + (1 - alpha) * nn_softvote_score
ENSEMBLE_ALPHA: float = 0.5