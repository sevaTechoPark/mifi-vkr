import argparse

from hybrid_vector_build import run_build
from hybrid_classical_models import run_classical
from hybrid_mlp import run_mlp


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified CLI for hybrid text classification experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build hybrid vectors")
    build_parser.add_argument("--train", required=True)
    build_parser.add_argument("--test", required=True)
    build_parser.add_argument("--outdir", required=True)
    build_parser.add_argument("--model_dir", default=None)
    build_parser.add_argument("--device", default="cpu")
    build_parser.add_argument("--bert_weight", type=float, default=5.0)
    build_parser.add_argument("--base_model_name", default="ai-forever/ruRoberta-large")
    build_parser.add_argument("--text_col", default="text")
    build_parser.add_argument("--label_col", default="label")
    build_parser.add_argument("--max_length", type=int, default=512)
    build_parser.add_argument("--chunk_size", type=int, default=448)
    build_parser.add_argument("--chunk_overlap", type=int, default=96)
    build_parser.add_argument("--pooling", choices=["mean", "cls", "max", "mean_max"], default="mean_max")
    build_parser.add_argument("--chunk_aggregation", choices=["mean", "max", "mean_max"], default="mean_max")
    build_parser.add_argument("--batch_size", type=int, default=8)

    classical_parser = subparsers.add_parser("classical", help="Run classical models on hybrid vectors")
    classical_parser.add_argument("--vecdir", required=True)

    mlp_parser = subparsers.add_parser("mlp", help="Run MLP on hybrid vectors")
    mlp_parser.add_argument("--vecdir", required=True)
    mlp_parser.add_argument("--epochs", type=int, default=25)
    mlp_parser.add_argument("--device", default="cpu")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        run_build(
            train_file=args.train,
            test_file=args.test,
            outdir=args.outdir,
            model_dir=args.model_dir,
            device=args.device,
            bert_weight=args.bert_weight,
            base_model_name=args.base_model_name,
            text_col=args.text_col,
            label_col=args.label_col,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            pooling=args.pooling,
            chunk_aggregation=args.chunk_aggregation,
            batch_size=args.batch_size,
        )
    elif args.command == "classical":
        run_classical(args.vecdir)
    elif args.command == "mlp":
        run_mlp(args.vecdir, epochs=args.epochs, device=args.device)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()