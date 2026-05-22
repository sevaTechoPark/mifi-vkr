"""Точки входа: CLI и Python API.

Поддерживаются три способа запуска:
  1. CLI: `python -m bert_classification.main --train-file ... --test-file ... --output-dir ...`
  2. run_from_params(...) — вызов из Python с произвольным набором overrides;
  3. run_from_configs(...) — вызов из Python уже собранными dataclass-конфигами.

Все параметры из ModelConfig / TrainConfig / DataConfig автоматически
превращаются в опции командной строки --kebab-case.
"""

import argparse
from dataclasses import fields, replace
from typing import Optional

from .config import ModelConfig, TrainConfig, DataConfig, PathConfig
from .training import run_training_pipeline


def _str2bool(v):
    """Парсит '1/true/yes' и '0/false/no' в bool — argparse не делает этого сам."""
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _field_arg_type(f):
    """Выбирает тип аргумента argparse по типу поля dataclass.

    Особый случай — bool: argparse сам по себе плохо работает с булевыми
    значениями, поэтому подменяем тип на _str2bool.
    """
    if f.type is bool:
        return _str2bool
    if f.type in (int, float, str):
        return f.type

    # Если тип задан строкой (например, при from __future__ import annotations),
    # f.type становится строкой — тогда смотрим на тип default-значения.
    default_type = type(f.default)
    if default_type is bool:
        return _str2bool
    return default_type


def _override_dataclass_from_args(instance, args, field_names):
    """Применяет к dataclass только те аргументы CLI, которые были явно переданы (не None)."""
    updates = {}
    for name in field_names:
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    return replace(instance, **updates)


def build_configs_from_args(args):
    """Собирает 4 конфига из parsed args. Пути обязательны, остальное — overrides."""
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
    """Создаёт ArgumentParser. Поля dataclass'ов добавляются автоматически."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate BERT text classification model"
    )

    parser.add_argument("--train-file", "--train", dest="train_file", required=True)
    parser.add_argument("--test-file", "--test", dest="test_file", required=True)
    parser.add_argument("--output-dir", required=True)

    # Добавляем по одному --kebab-case флагу на каждое поле каждого конфига.
    for cfg_cls in (ModelConfig, TrainConfig, DataConfig):
        for f in fields(cfg_cls):
            parser.add_argument(
                f"--{f.name.replace('_', '-')}",
                type=_field_arg_type(f),
                default=None,
            )

    return parser


def run_from_configs(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    path_cfg: PathConfig,
):
    """Запуск пайплайна с уже собранными конфигами (например, из Jupyter)."""
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
    """Удобная обёртка: пути + произвольные overrides в виде kwargs.

    Каждый override попадает в тот dataclass, в котором есть поле с таким именем.
    Если имя не найдено ни в одном — поднимаем ValueError, чтобы опечатки
    не игнорировались тихо.
    """
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
