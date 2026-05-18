import argparse
from dataclasses import fields, replace

import torch

from .config import HybridModelConfig, HybridDataConfig, HybridPathConfig
from .hybrid_vector_build import run_build
from .hybrid_classical_models import run_classical
from .hybrid_mlp import run_mlp


def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _field_arg_type(f):
    if f.type is bool:
        return _str2bool
    if f.type in (int, float, str):
        return f.type
    default_type = type(f.default)
    if default_type is bool:
        return _str2bool
    return default_type


def _override_dataclass_from_args(cfg, args, allowed_fields):
    updates = {}
    for name in allowed_fields:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                updates[name] = value
    return replace(cfg, **updates)


def _detect_device(device=None):
    if device is not None:
        device = str(device).strip().lower()
        if device:
            return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_configs_from_args(args):
    model_cfg = _override_dataclass_from_args(
        HybridModelConfig(),
        args,
        {f.name for f in fields(HybridModelConfig)},
    )
    data_cfg = _override_dataclass_from_args(
        HybridDataConfig(),
        args,
        {f.name for f in fields(HybridDataConfig)},
    )

    path_cfg = HybridPathConfig(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
    )

    return model_cfg, data_cfg, path_cfg


def _add_dataclass_args(parser, dc_cls):
    for f in fields(dc_cls):
        parser.add_argument(
            f"--{f.name.replace('_', '-')}",
            dest=f.name,
            type=_field_arg_type(f),
            default=None,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified CLI for hybrid text classification experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build hybrid vectors")
    build_parser.add_argument("--train-file", required=True)
    build_parser.add_argument("--test-file", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--model-dir", default=None)
    build_parser.add_argument("--device", default=None)

    _add_dataclass_args(build_parser, HybridModelConfig)
    _add_dataclass_args(build_parser, HybridDataConfig)

    classical_parser = subparsers.add_parser(
        "classical",
        help="Run classical models on hybrid vectors",
    )
    classical_parser.add_argument("--vecdir", required=True)

    mlp_parser = subparsers.add_parser("mlp", help="Run MLP on hybrid vectors")
    mlp_parser.add_argument("--vecdir", required=True)
    mlp_parser.add_argument("--epochs", type=int, default=None)
    mlp_parser.add_argument("--device", default=None)

    # MLP-гиперпараметры (если не задавать — берутся из HybridMLPConfig)
    mlp_parser.add_argument("--seed", type=int, default=None)
    mlp_parser.add_argument("--batch-size", type=int, default=None)
    mlp_parser.add_argument("--learning-rate", type=float, default=None)
    mlp_parser.add_argument("--patience", type=int, default=None)
    mlp_parser.add_argument("--weight-decay", type=float, default=None)
    mlp_parser.add_argument("--hidden-dim", type=int, default=None)
    mlp_parser.add_argument("--num-blocks", type=int, default=None)
    mlp_parser.add_argument("--dropout", type=float, default=None)
    mlp_parser.add_argument("--focal-gamma", type=float, default=None)
    mlp_parser.add_argument("--label-smoothing", type=float, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        model_cfg, data_cfg, path_cfg = build_configs_from_args(args)
        device = _detect_device(args.device)
        print(f"[INFO] Using device: {device}")

        run_build(
            train_file=path_cfg.train_file,
            test_file=path_cfg.test_file,
            outdir=path_cfg.output_dir,
            model_dir=args.model_dir,
            device=device,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
        )
    elif args.command == "classical":
        run_classical(args.vecdir)
    elif args.command == "mlp":
        device = _detect_device(args.device)
        print(f"[INFO] Using device: {device}")

        from .hybrid_mlp import _cfg_from_args
        cfg = _cfg_from_args(args)

        run_mlp(args.vecdir, cfg=cfg, epochs=args.epochs, device=device)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()