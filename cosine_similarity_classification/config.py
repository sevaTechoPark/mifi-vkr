"""
Конфиг модуля cosine_similarity_classification.

Единственный источник правды по гиперпараметрам эмбеддера — bert_embeddings.config.EmbeddingConfig.
Здесь мы только перечисляем поля для CLI-аргументов и значения по умолчанию вытаскиваем
из EmbeddingConfig().
"""

from bert_embeddings.config import EmbeddingConfig

_emb = EmbeddingConfig()

BASE_MODEL_NAME: str = _emb.base_model_name

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

METHOD: str = "centroid"

MAX_LENGTH: int = _emb.max_length
CHUNK_SIZE: int = _emb.chunk_size
CHUNK_OVERLAP: int = _emb.chunk_overlap

POOLING: str = _emb.pooling
CHUNK_AGGREGATION: str = _emb.chunk_aggregation

BATCH_SIZE: int = _emb.batch_size

DEVICE: str | None = None
MODEL_DIR: str = ""

# kNN-параметры для метода "nearest"
KNN_K: int = 5


# kNN: soft-voting параметры
KNN_TEMPERATURE: float = 0.1
# Список k для sweep (через CLI --knn-k-sweep "1,3,5,7,9,11").
KNN_K_SWEEP: str = ""

# Centroid: доля «дальних» точек класса, которые отбрасываются перед усреднением.
# trim=0.15 выбран по результатам v6-сравнения. На custom embedder даёт +1.4pp
# на train.csv, на baseline +0.7pp. trim=0.2 эквивалентен в среднем, но менее
# устойчив на baseline_train.
CENTROID_TRIM_RATIO: float = 0.15