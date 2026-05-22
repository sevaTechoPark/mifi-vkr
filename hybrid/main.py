"""
Единый CLI для гибридного пайплайна.

Подкоманды:

  * ``build``     — построение гибридных признаков (TF-IDF + BERT);
  * ``classical`` — обучение линейных классификаторов поверх признаков;
  * ``mlp``       — обучение MLP-классификатора поверх признаков.
"""

import argparse
import sys
from dataclasses import fields, replace

import torch

from .config import (
    DEFAULT_HYBRID_MLP_FEATURE_SOURCE,
    HYBRID_MLP_FEATURE_SOURCES,
    HYBRID_MLP_PROFILES,
    HybridDataConfig,
    HybridModelConfig,
    HybridPathConfig,
)
from .hybrid_classical_models import run_classical
from .hybrid_mlp import run_mlp
from .hybrid_vector_build import run_build


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
    """Подобрать тип CLI-аргумента по типу поля датакласса."""
    if f.type is bool:
        return _str2bool
    if f.type in (int, float, str):
        return f.type
    default_type = type(f.default)
    if default_type is bool:
        return _str2bool
    return default_type


def _override_dataclass_from_args(cfg, args, allowed_fields):
    """Применить непустые CLI-аргументы поверх полей датакласса."""
    updates = {}
    for name in allowed_fields:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                updates[name] = value
    return replace(cfg, **updates)


def _detect_device(device=None):
    """Подобрать вычислительное устройство по пожеланию пользователя."""
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
    """Собрать model_cfg / data_cfg / path_cfg из argparse-Namespace."""
    model_cfg = _override_dataclass_from_args(
        HybridModelConfig(), args,
        {f.name for f in fields(HybridModelConfig)},
    )
    data_cfg = _override_dataclass_from_args(
        HybridDataConfig(), args,
        {f.name for f in fields(HybridDataConfig)},
    )
    path_cfg = HybridPathConfig(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
    )
    return model_cfg, data_cfg, path_cfg


def _add_dataclass_args(parser, dc_cls):
    """Добавить в parser по одному CLI-аргументу для каждого поля датакласса."""
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

    # --- build --------------------------------------------------------------
    build_parser_ = subparsers.add_parser("build", help="Build hybrid vectors")
    build_parser_.add_argument("--train-file", required=True)
    build_parser_.add_argument("--test-file", required=True)
    build_parser_.add_argument("--output-dir", required=True)
    build_parser_.add_argument("--model-dir", default=None)
    build_parser_.add_argument("--device", default=None)
    _add_dataclass_args(build_parser_, HybridModelConfig)
    _add_dataclass_args(build_parser_, HybridDataConfig)

    # --- classical ----------------------------------------------------------
    classical_parser = subparsers.add_parser(
        "classical", help="Run classical models on hybrid vectors",
    )
    classical_parser.add_argument("--vecdir", required=True)
    classical_parser.add_argument(
        "--class-weight", default="balanced",
        choices=["balanced", "none"], nargs="?",
        help="Балансировка классов. По умолчанию 'balanced' "
             "(важно при сильном дисбалансе). 'none' отключает балансировку.",
    )
    classical_parser.add_argument("--no-tfidf-only", action="store_true")
    classical_parser.add_argument("--feature-sources", default=None)
    classical_parser.add_argument("--c-grid", default=None)
    classical_parser.add_argument("--no-stacking", action="store_true")
    classical_parser.add_argument("--no-rbf", action="store_true")

    # --- mlp ---------------------------------------------------------------
    mlp_parser = subparsers.add_parser("mlp", help="Run MLP on hybrid vectors")
    mlp_parser.add_argument("--vecdir", required=True)
    mlp_parser.add_argument("--epochs", type=int, default=None)
    mlp_parser.add_argument("--device", default=None)
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
    mlp_parser.add_argument("--mixup-alpha", type=float, default=None)
    mlp_parser.add_argument("--max-grad-norm", type=float, default=None)
    mlp_parser.add_argument(
        "--profile", type=str, default=None,
        choices=sorted(HYBRID_MLP_PROFILES.keys()),
        help="Профиль гиперпараметров MLP. По умолчанию выбирается от --features.",
    )
    mlp_parser.add_argument(
        "--features", type=str, default=None,
        choices=list(HYBRID_MLP_FEATURE_SOURCES),
        help=f'Источник признаков для MLP. По умолчанию: "{DEFAULT_HYBRID_MLP_FEATURE_SOURCE}".',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        model_cfg, data_cfg, path_cfg = build_configs_from_args(args)
        device = _detect_device(args.device)
        print(f"[INFO] Using device: {device}")

        # Если пользователь не задал --bert-weight явно, подставляем
        # подходящее значение: для дообученной модели — повышенный вес,
        # для базовой — единичный.
        user_passed_bw = any(arg.startswith("--bert-weight") for arg in sys.argv)
        if not user_passed_bw:
            if args.model_dir:
                model_cfg.bert_weight = model_cfg.bert_weight_with_model_dir
                print(
                    f"[hybrid.build] model_dir is set → "
                    f"auto bert_weight={model_cfg.bert_weight}"
                )
            else:
                model_cfg.bert_weight = model_cfg.bert_weight_base_model
                print(
                    f"[hybrid.build] no model_dir → "
                    f"auto bert_weight={model_cfg.bert_weight}"
                )
        else:
            print(
                f"[hybrid.build] using user-supplied "
                f"bert_weight={model_cfg.bert_weight}"
            )

        # Для дообученного эмбеддера StandardScaler по BERT-блоку отключается:
        # его эмбеддинги уже L2-нормированы метрическим обучением.
        if args.model_dir and not model_cfg.disable_bert_scaler:
            model_cfg.disable_bert_scaler = True
            print(
                "[hybrid.build] model_dir is set → disabling BERT StandardScaler"
            )

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
        sources = tuple(
            s.strip()
            for s in (args.feature_sources or "bert_only,hybrid,tfidf_only").split(",")
            if s.strip()
        )
        c_grid = tuple(
            float(c)
            for c in (args.c_grid or "0.05,0.1,0.3,0.5,1.0,2.0,3.0,5.0").split(",")
            if c.strip()
        )
        cw_arg = args.class_weight
        if cw_arg is None or (isinstance(cw_arg, str) and cw_arg.lower() == "none"):
            cw = None
        else:
            cw = cw_arg
        run_classical(
            args.vecdir,
            class_weight=cw,
            include_tfidf_only=(not args.no_tfidf_only) and ("tfidf_only" in sources),
            feature_sources=sources,
            c_grid=c_grid,
        )

    elif args.command == "mlp":
        device = _detect_device(args.device)
        print(f"[INFO] Using device: {device}")

        from .hybrid_mlp import _cfg_from_args
        cfg, feature_source = _cfg_from_args(args)
        run_mlp(
            args.vecdir, cfg=cfg, epochs=args.epochs, device=device,
            feature_source=feature_source,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
