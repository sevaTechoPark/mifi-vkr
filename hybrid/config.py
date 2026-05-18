from dataclasses import dataclass

# Единственный источник правды для параметров эмбеддера/MLM — bert_embeddings.config
from bert_embeddings.config import EmbeddingConfig, MLMConfig, ensure_dir  # noqa: F401  (реэкспорт)

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
    # После финальной L2-нормировки гибридного вектора меняется только пропорция
    # энергии между блоками. Логируется bert_share_mean в meta.json.
    bert_weight: float = 5.0


@dataclass
class HybridDataConfig:
    text_col: str = "text"
    label_col: str = "label"


@dataclass
class HybridPathConfig:
    train_file: str = ""
    test_file: str = ""
    output_dir: str = ""