import argparse
from typing import Optional

from .config import ModelConfig, TrainConfig, DataConfig, PathConfig
from .training import run_training_pipeline


def build_configs_from_args(args):
    model_cfg = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        stride=args.stride,
        max_chunks=args.max_chunks,
        head_dropout=args.head_dropout,
        label_smoothing=args.label_smoothing,
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        max_grad_norm=args.max_grad_norm,
        metric_for_best_model="f1_macro",
    )

    data_cfg = DataConfig(
        text_col=args.text_col,
        label_col=args.label_col,
    )

    path_cfg = PathConfig(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
    )

    return model_cfg, train_cfg, data_cfg, path_cfg


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", default="./saved_models/bert-classification")

    parser.add_argument("--model-name", default="ai-forever/ruRoberta-large")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr-encoder", type=float, default=1.2e-5)
    parser.add_argument("--lr-head", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")

    return parser


def run_from_configs(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    path_cfg: PathConfig,
):
    return run_training_pipeline(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        path_cfg=path_cfg,
    )


def run_from_params(
    train_file: str,
    test_file: str,
    output_dir: str = "./saved_models/bert-classification",
    model_name: str = "ai-forever/ruRoberta-large",
    max_length: int = 512,
    stride: int = 256,
    max_chunks: int = 6,
    head_dropout: float = 0.3,
    label_smoothing: float = 0.05,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    num_epochs: int = 20,
    lr_encoder: float = 1.2e-5,
    lr_head: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 2,
    checkpoint_every_n_epochs: int = 3,
    seed: int = 42,
    dataloader_num_workers: int = 2,
    max_grad_norm: float = 1.0,
    text_col: str = "text",
    label_col: str = "label",
):
    model_cfg = ModelConfig(
        model_name=model_name,
        max_length=max_length,
        stride=stride,
        max_chunks=max_chunks,
        head_dropout=head_dropout,
        label_smoothing=label_smoothing,
    )

    train_cfg = TrainConfig(
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        num_epochs=num_epochs,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        early_stopping_patience=early_stopping_patience,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        max_grad_norm=max_grad_norm,
        metric_for_best_model="f1_macro",
    )

    data_cfg = DataConfig(
        text_col=text_col,
        label_col=label_col,
    )

    path_cfg = PathConfig(
        train_file=train_file,
        test_file=test_file,
        output_dir=output_dir,
    )

    return run_from_configs(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        path_cfg=path_cfg,
    )


def cli_main(argv: Optional[list] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    model_cfg, train_cfg, data_cfg, path_cfg = build_configs_from_args(args)
    return run_from_configs(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        path_cfg=path_cfg,
    )


if __name__ == "__main__":
    cli_main()