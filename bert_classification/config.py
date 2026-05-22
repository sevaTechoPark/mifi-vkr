"""Конфигурация пайплайна классификации на основе ruRoberta-large.

Конфиги разделены по смыслу:
  - ModelConfig — параметры модели и токенизации (включая чанкование длинных текстов);
  - TrainConfig — параметры обучения (LR, оптимизация, регуляризация, mixed precision);
  - DataConfig  — имена колонок в CSV;
  - PathConfig  — пути к входным файлам и каталогу вывода.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Параметры модели и обработки входных текстов."""

    # Базовая предобученная модель.
    model_name: str = "ai-forever/ruRoberta-large"

    # Параметры чанкования длинных документов.
    # Документ режется на перекрывающиеся окна длиной max_length токенов с шагом
    # max_length - stride; берётся не более max_chunks окон. Эмбеддинги окон
    # усредняются на уровне модели — см. ChunkMeanPoolRobertaClassifier.
    max_length: int = 512
    stride: int = 128
    max_chunks: int = 6

    # Регуляризация классификационной головы.
    head_dropout: float = 0.2
    label_smoothing: float = 0.02

    # Заморозка нижних слоёв энкодера для борьбы с переобучением на малой выборке.
    # 0  — обучаем весь энкодер;
    # 12 — фризим нижнюю половину (рекомендуемое значение для ruRoberta-large);
    # 18 — оставляем для обучения только верхние 6 слоёв.
    freeze_encoder_layers: int = 12

    # Эмбеддинги (word/position/type) при тонкой настройке обычно меняются мало,
    # фриз даёт стабильность и экономит память.
    freeze_embeddings: bool = True


@dataclass
class TrainConfig:
    """Параметры цикла обучения."""

    # Размер батча и эффективный размер через накопление градиента.
    batch_size: int = 1
    grad_accum_steps: int = 8

    num_epochs: int = 10
    early_stopping_patience: int = 3

    # Раздельные LR для энкодера и для головы — задаются в WeightedChunkTrainer.
    lr_encoder: float = 1e-5
    lr_head: float = 1e-5

    weight_decay: float = 0.01

    # Доля от общего числа шагов на warmup. Используется, если warmup_steps == 0,
    # иначе берётся абсолютное число warmup_steps.
    warmup_ratio: float = 0.06
    warmup_steps: int = 0

    seed: int = 42
    dataloader_num_workers: int = 2
    max_grad_norm: float = 1.0
    metric_for_best_model: str = "f1_macro"

    # Mixed precision. На устройствах с capability >= 8 (Ampere и новее)
    # используем bf16 — он стабильнее, чем fp16. На более старых GPU
    # автоматически откатываемся на fp16, если разрешено флагом ниже.
    bf16: bool = True
    fp16_fallback_on_non_a100: bool = True

    # Экономия памяти ценой ~20–30% скорости.
    gradient_checkpointing: bool = True

    # TF32-ускорение матричных операций (только при наличии CUDA).
    tf32: bool = True


@dataclass
class DataConfig:
    """Имена колонок в CSV с обучающей и тестовой выборками."""

    text_col: str = "text"
    label_col: str = "label"


@dataclass
class PathConfig:
    """Пути к входным CSV и каталогу вывода (метрики, чекпоинт)."""

    train_file: str = ""
    test_file: str = ""
    output_dir: str = ""
