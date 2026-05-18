from dataclasses import dataclass
from pathlib import Path


@dataclass
class MLMConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    text_col: str = "text"
    label_col: str = "label"

    # Для ruRoberta-large безопасно держать 512.
    # Для длинных писем используем chunk-aware обучение/инференс.
    max_length: int = 512

    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    logging_steps: int = 50

    # На A100 ставим bf16, fp16 отключаем — bf16 стабильнее и нативно поддержан.
    # На non-A100/без CUDA — bf16=False, fp16=False (выставляется в main.py).
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    tf32: bool = True

    seed: int = 42

    val_size: float = 0.1
    early_stopping_patience: int = 3

    # Снижены: 5000 уже даёт качественный сигнал без насыщения MNR.
    max_pairs_per_label: int = 5000
    max_negative_pairs: int = 5000
    max_eval_pairs_per_label: int = 4000

    train_pair_strategy: str = "pair_class"   # pair_class | pair_score
    train_loss: str = "mnr"                   # softmax | cosent | mnr

    train_chunk_size: int = 448
    train_chunk_overlap: int = 128
    add_global_chunk_to_training: bool = True

    # Кросс-документные позитивы для MNR (фикс representation collapse).
    cross_document_positives_only: bool = True

    # POOLING обучения. На инференсе EmbeddingConfig.pooling ДОЛЖЕН совпадать.
    sentence_pooling: str = "mean"

    # Заморозить N нижних слоёв на 1-й эпохе (24 слоя у ruRoberta-large).
    freeze_lower_layers: int = 12


@dataclass
class EmbeddingConfig:
    max_length: int = 512
    chunk_size: int = 448
    chunk_overlap: int = 128
    # MATCH с MLMConfig.sentence_pooling. Не менять без переобучения энкодера.
    pooling: str = "mean"
    chunk_aggregation: str = "mean"
    batch_size: int = 8
    normalize_chunks: bool = True
    normalize_document: bool = True
    add_global_chunk: bool = True
    base_model_name: str = "ai-forever/ruRoberta-large"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path