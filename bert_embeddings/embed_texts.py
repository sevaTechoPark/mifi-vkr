import argparse
import numpy as np
import pandas as pd

from bert_embeddings.embedding_model import LongTextRobertaEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_npy", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="ai-forever/ruRoberta-large")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=448)
    parser.add_argument("--chunk_overlap", type=int, default=96)
    parser.add_argument("--pooling", type=str, choices=["mean", "cls", "max", "mean_max"], default="mean_max")
    parser.add_argument("--chunk_aggregation", type=str, choices=["mean", "max", "mean_max"], default="mean_max")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--disable_global_chunk", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[[args.text_col]].dropna().copy()
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""].reset_index(drop=True)

    embedder = LongTextRobertaEmbedder(
        model_dir=args.model_dir,
        base_model_name=args.base_model_name,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pooling=args.pooling,
        chunk_aggregation=args.chunk_aggregation,
        batch_size=args.batch_size,
        add_global_chunk=not args.disable_global_chunk,
    )

    embs, chunk_counts = embedder.encode(df[args.text_col].tolist(), return_chunk_counts=True)
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