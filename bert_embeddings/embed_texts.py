import argparse
from dataclasses import fields, replace

import numpy as np
import pandas as pd

from bert_embeddings.config import EmbeddingConfig
from bert_embeddings.embedding_model import LongTextRobertaEmbedder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed texts with fine-tuned ruRoberta encoder"
    )

    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-npy", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--model-dir", type=str, required=True)

    defaults = {f.name: f.default for f in fields(EmbeddingConfig)}
    for f in fields(EmbeddingConfig):
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

    df = pd.read_csv(args.input_csv)
    df = df[["text"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    cfg = EmbeddingConfig()

    overrides = {}
    for f in fields(EmbeddingConfig):
        if not hasattr(args, f.name):
            continue
        val = getattr(args, f.name)
        if val is not None:
            overrides[f.name] = val

    if overrides:
        cfg = replace(cfg, **overrides)

    embedder = LongTextRobertaEmbedder(
        model_dir=args.model_dir,
        cfg=cfg,
    )

    embs, chunk_counts = embedder.encode(df["text"].tolist(), return_chunk_counts=True)
    np.save(args.output_npy, embs)
    print(f"Saved embeddings to: {args.output_npy}")

    if args.output_csv:
        out = df.copy()
        out["chunk_count"] = chunk_counts
        out["embedding_dim"] = embs.shape[1]
        out["embedding_norm"] = np.linalg.norm(embs, axis=1)
        out.to_csv(args.output_csv, index=False)
        print(f"Saved manifest to: {args.output_csv}")


if __name__ == "__main__":
    main()