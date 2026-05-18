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