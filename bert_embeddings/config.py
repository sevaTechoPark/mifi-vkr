from dataclasses import dataclass
from pathlib import Path


@dataclass
class MLMConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    text_col: str = "text"
    max_length: int = 512
    mlm_probability: float = 0.15
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 15
    warmup_ratio: float = 0.10
    logging_steps: int = 100
    fp16: bool = True
    seed: int = 42
    val_size: float = 0.02
    checkpoint_every_n_epochs: int = 3
    early_stopping_patience: int = 3  # всегда включён, передаётся в нативный EarlyStoppingCallback


@dataclass
class EmbeddingConfig:
    max_length: int = 512
    chunk_size: int = 448
    chunk_overlap: int = 96
    pooling: str = "mean_max"
    chunk_aggregation: str = "mean_max"
    batch_size: int = 8
    normalize_chunks: bool = True
    normalize_document: bool = True
    add_global_chunk: bool = True
    base_model_name: str = "ai-forever/ruRoberta-large"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path