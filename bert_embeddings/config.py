from dataclasses import dataclass
from pathlib import Path


@dataclass
class MLMConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    text_col: str = "text"
    label_col: str = "label"

    # Важно: для ruRoberta-large безопасно держать 512.
    # Для длинных писем используем chunk-aware обучение/инференс.
    max_length: int = 512

    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    fp16: bool = True
    seed: int = 42

    val_size: float = 0.1
    checkpoint_every_n_epochs: int = 1
    early_stopping_patience: int = 3

    max_pairs_per_label: int = 20000
    max_negative_pairs: int = 20000
    max_eval_pairs_per_label: int = 4000

    train_pair_strategy: str = "pair_class"   # pair_class | pair_score
    train_loss: str = "mnr"                   # softmax | cosent | mnr

    train_chunk_size: int = 448
    train_chunk_overlap: int = 128
    add_global_chunk_to_training: bool = True

    sentence_pooling: str = "mean"
    save_sentence_transformer_artifacts: bool = True


@dataclass
class EmbeddingConfig:
    max_length: int = 512
    chunk_size: int = 448
    chunk_overlap: int = 128
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