import argparse
from dataclasses import fields, replace

from bert_embeddings.config import MLMConfig
from bert_embeddings.main import run_from_params


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sentence-embedding training for ruRoberta (legacy entrypoint name: mlm_train)"
    )

    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    defaults = {f.name: f.default for f in fields(MLMConfig)}
    for f in fields(MLMConfig):
        default = defaults[f.name]
        if isinstance(default, bool):
            continue
        arg_name = f"--{f.name.replace('_', '-')}"
        arg_type = type(default)
        parser.add_argument(arg_name, type=arg_type, default=None)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = MLMConfig()

    overrides = {}
    for f in fields(MLMConfig):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if val is not None:
            overrides[f.name] = val

    if overrides:
        cfg = replace(cfg, **overrides)

    run_from_params(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()