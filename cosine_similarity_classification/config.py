"""Параметры по умолчанию для cosine_similarity_classification.

Эмбеддерные параметры (BASE_MODEL_NAME, MAX_LENGTH, CHUNK_SIZE …) берутся
из bert_embeddings.config.EmbeddingConfig, чтобы train-time и inference-time
конфигурация энкодера была согласована.
"""

from bert_embeddings.config import EmbeddingConfig

_emb = EmbeddingConfig()

BASE_MODEL_NAME: str = _emb.base_model_name

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

METHOD: str = "all"

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
KNN_K_SWEEP: str = "1,3,5,7,9,11,15,21,31"
KNN_T_SWEEP: str = "0.03,0.05,0.1,0.15,0.2,0.3"

# -----------------------------------------------------------------------------
# Centroid: робастный центроид с trim + soft-trim + iterative refinement
# -----------------------------------------------------------------------------
CENTROID_TRIM_RATIO: float = 0.15
CENTROID_TRIM_MODE: str = "soft"
CENTROID_TRIM_POWER: float = 4.0
CENTROID_REFINE_ITERS: int = 1

# Sweep по гиперпараметрам центроида — перебираются все комбинации mode × ratio × power.
CENTROID_TRIM_POWER_SWEEP: str = "2,4,6,8"
CENTROID_TRIM_MODE_SWEEP: str = "hard,soft"
CENTROID_TRIM_RATIO_SWEEP: str = "0,0.1,0.15,0.2"

# -----------------------------------------------------------------------------
# centroid_nn ensemble (смесь centroid и nearest)
# -----------------------------------------------------------------------------
ENSEMBLE_ALPHA: float = 0.5
ENSEMBLE_ALPHA_SWEEP: str = "0.3,0.4,0.5,0.6,0.7"
