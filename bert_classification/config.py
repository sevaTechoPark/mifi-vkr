from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    max_length: int = 512
    stride: int = 256
    max_chunks: int = 6
    head_dropout: float = 0.3
    label_smoothing: float = 0.05


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 20
    lr_encoder: float = 1.2e-5
    lr_head: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    checkpoint_every_n_epochs: int = 3
    seed: int = 42
    dataloader_num_workers: int = 2
    max_grad_norm: float = 1.0
    metric_for_best_model: str = "f1_macro"


@dataclass
class DataConfig:
    text_col: str = "text"
    label_col: str = "label"


@dataclass
class PathConfig:
    train_file: str = ""
    test_file: str = ""
    output_dir: str = ""