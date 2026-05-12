from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    max_length: int = 512
    stride: int = 256
    max_chunks: int = 6
    head_dropout: float = 0.4
    label_smoothing: float = 0.02


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 16
    lr_encoder: float = 8e-6
    lr_head: float = 2e-5
    weight_decay: float = 0.02
    warmup_steps: int = 150
    early_stopping_patience: int = 3
    checkpoint_every_n_epochs: int = 4
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