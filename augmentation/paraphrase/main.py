# paraphrase/main.py
import argparse
import os
import torch
import pandas as pd

from common.embeddings import load_embed_model
from common.augment_loop import run_augmentation_loop
from common.seed import set_seed, get_seed_or_default
from paraphrase.models import load_paraphrase_model
from paraphrase.augment import paraphrase_document


def build_augment_fn(tok, model, embed_model, device):
    def augment_fn(text: str) -> str:
        return paraphrase_document(text, tok, model, embed_model, device)
    return augment_fn


def main():
    parser = argparse.ArgumentParser(description="Paraphrase augmentation")
    parser.add_argument("--train",      required=True,  help="Path to train CSV")
    parser.add_argument("--output-dir", required=True,  help="Directory for output files")
    parser.add_argument("--single",     default=None,   help="Single text to paraphrase")
    parser.add_argument("--seed",       type=int,       default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # фиксируем рандом
    seed = get_seed_or_default(args.seed)
    set_seed(seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_model = load_embed_model(device)
    tok, para_model = load_paraphrase_model(device)

    augment_fn = build_augment_fn(tok, para_model, embed_model, device)

    if args.single:
        result = augment_fn(args.single)
        print("\n=== Paraphrased ===")
        print(result)
        return

    df = pd.read_csv(args.train)
    aug_file = os.path.join(args.output_dir, "train_paraphrase_partial.csv")

    df_aug = run_augmentation_loop(
        df=df,
        embed_model=embed_model,
        augment_fn=augment_fn,
        aug_file_path=aug_file,
        augmentation_type="paraphrase",
    )

    df_full = pd.concat([df, df_aug[["label", "text"]]], ignore_index=True)
    final_path = os.path.join(args.output_dir, "train_paraphrase.csv")
    df_full.to_csv(final_path, index=False)
    print(f"Финальный датасет: {final_path}")


if __name__ == "__main__":
    main()