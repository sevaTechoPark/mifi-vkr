from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    max_length: int = 512
    stride: int = 128
    max_chunks: int = 6              # A: было 4 — больше контекста для длинных текстов
    head_dropout: float = 0.2
    label_smoothing: float = 0.02

    # D: Заморозка нижних слоёв ruRoberta-large для борьбы с переобучением.
    # 0  — ничего не замораживаем (старое поведение).
    # 12 — заморозить layers 0..11 (нижняя половина) — рекомендуемое значение.
    # 18 — заморозить layers 0..17 (агрессивно, оставит только 6 верхних).
    freeze_encoder_layers: int = 12
    # Embeddings обычно стабильны и небольшой train сильно их не улучшит.
    # True — рекомендуемое значение для small-data fine-tune.
    freeze_embeddings: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 10            # было 16; early stop всё равно остановит раньше
    lr_encoder: float = 1e-5        # было 8e-6 — чуть выше для активного дообучения
    lr_head: float = 1e-5           # было 2e-5 — выравниваем с энкодером
    weight_decay: float = 0.01      # было 0.02
    warmup_ratio: float = 0.06      # вместо абсолютных warmup_steps
    warmup_steps: int = 0           # 0 значит «использовать warmup_ratio»
    early_stopping_patience: int = 3
    seed: int = 42
    dataloader_num_workers: int = 2
    max_grad_norm: float = 1.0
    metric_for_best_model: str = "f1_macro"
    # bf16 на A100 (стабильнее fp16); на T4/L4 fp16 включается автоматически
    bf16: bool = True
    fp16_fallback_on_non_a100: bool = True
    gradient_checkpointing: bool = True
    tf32: bool = True


@dataclass
class DataConfig:
    text_col: str = "text"
    label_col: str = "label"


@dataclass
class PathConfig:
    train_file: str = ""
    test_file: str = ""
    output_dir: str = ""