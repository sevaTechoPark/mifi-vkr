import argparse
from dataclasses import fields, replace
from typing import Optional

from .config import ModelConfig, TrainConfig, DataConfig, PathConfig
from .training import run_training_pipeline


def _override_dataclass_from_args(instance, args, field_names):
    updates = {}
    for name in field_names:
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    return replace(instance, **updates)


def build_configs_from_args(args):
    model_cfg = _override_dataclass_from_args(
        ModelConfig(),
        args,
        {f.name for f in fields(ModelConfig)},
    )
    train_cfg = _override_dataclass_from_args(
        TrainConfig(),
        args,
        {f.name for f in fields(TrainConfig)},
    )
    data_cfg = _override_dataclass_from_args(
        DataConfig(),
        args,
        {f.name for f in fields(DataConfig)},
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
    parser.add_argument("--output-dir", required=True)

    for f in fields(ModelConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)

    for f in fields(TrainConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)

    for f in fields(DataConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)

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
    output_dir: str,
    **overrides,
):
    model_field_names = {f.name for f in fields(ModelConfig)}
    train_field_names = {f.name for f in fields(TrainConfig)}
    data_field_names = {f.name for f in fields(DataConfig)}

    model_updates = {k: v for k, v in overrides.items() if k in model_field_names}
    train_updates = {k: v for k, v in overrides.items() if k in train_field_names}
    data_updates = {k: v for k, v in overrides.items() if k in data_field_names}

    unknown = set(overrides) - model_field_names - train_field_names - data_field_names
    if unknown:
        raise ValueError(f"Unknown run_from_params overrides: {sorted(unknown)}")

    model_cfg = replace(ModelConfig(), **model_updates)
    train_cfg = replace(TrainConfig(), **train_updates)
    data_cfg = replace(DataConfig(), **data_updates)
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