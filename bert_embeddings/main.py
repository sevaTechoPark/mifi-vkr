from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    models,
)
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
)
from transformers import EarlyStoppingCallback

from bert_embeddings.config import MLMConfig, ensure_dir
from bert_embeddings.data_utils import (
    build_pair_dataframe,
    build_training_dataframe,
    explode_long_texts_for_training,
)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sentence_transformer(
    model_name: str,
    max_length: int,
    pooling: str,
) -> SentenceTransformer:
    transformer = models.Transformer(
        model_name,
        max_seq_length=max_length,
        model_kwargs={"torch_dtype": "float32"},
    )

    pooling = pooling.lower().strip()
    pooling_model = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=(pooling == "mean"),
        pooling_mode_max_tokens=(pooling == "max"),
        pooling_mode_cls_token=(pooling == "cls"),
    )

    normalize = models.Normalize()
    return SentenceTransformer(modules=[transformer, pooling_model, normalize])


def save_encoder_only_from_sentence_transformer(
    model: SentenceTransformer,
    save_dir: str | Path,
    meta=None,
):
    save_path = ensure_dir(save_dir)

    transformer_module = model[0]
    auto_model = transformer_module.auto_model
    tokenizer = transformer_module.tokenizer

    state_dict = auto_model.state_dict()
    exported = {}
    for k, v in state_dict.items():
        if k.startswith("roberta."):
            exported[k] = v.detach().cpu()
        elif k.startswith("embeddings.") or k.startswith("encoder.") or k.startswith("pooler."):
            exported[f"roberta.{k}"] = v.detach().cpu()
        else:
            exported[f"roberta.{k}"] = v.detach().cpu()

    torch.save(exported, save_path / "pytorch_model.bin")
    auto_model.config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    with open(save_path / "mlm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)


def build_train_and_eval_pairs(cfg: MLMConfig, train_file: str, test_file: str):
    raw_df = build_training_dataframe(
        train_file,
        test_file,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
    )

    exploded_df = explode_long_texts_for_training(
        raw_df,
        model_name=cfg.model_name,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        max_length=cfg.max_length,
        chunk_size=cfg.train_chunk_size,
        chunk_overlap=cfg.train_chunk_overlap,
        add_global_chunk=cfg.add_global_chunk_to_training,
    )

    pair_df = build_pair_dataframe(
        exploded_df,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        max_pairs_per_label=cfg.max_pairs_per_label,
        max_negative_pairs=cfg.max_negative_pairs,
        seed=cfg.seed,
    )

    if len(pair_df) == 0:
        raise ValueError("No training pairs were built from text/label data.")

    val_size = min(max(cfg.val_size, 0.01), 0.3)
    ds = Dataset.from_pandas(pair_df, preserve_index=False)
    split_ds = ds.train_test_split(test_size=val_size, seed=cfg.seed)

    return raw_df, exploded_df, pair_df, split_ds["train"], split_ds["test"]


def run_from_params(
    train_file,
    test_file,
    output_dir,
    cfg: MLMConfig | None = None,
    **kwargs,
):
    if cfg is None:
        cfg = MLMConfig()

    checkpoint_every = cfg.checkpoint_every_n_epochs

    if kwargs:
        if "num_epochs" in kwargs:
            kwargs["num_train_epochs"] = kwargs.pop("num_epochs")
        if "batch_size" in kwargs:
            kwargs["train_batch_size"] = kwargs.pop("batch_size")
        if "checkpoint_every_n_epochs" in kwargs:
            checkpoint_every = kwargs.pop("checkpoint_every_n_epochs")

        valid_fields = MLMConfig.__dataclass_fields__
        cfg = replace(cfg, **{k: v for k, v in kwargs.items() if k in valid_fields})

    cfg = replace(cfg, fp16=torch.cuda.is_available())
    cleanup_memory()
    set_seed(cfg.seed)

    output_dir = ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.json"
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    best_model_dir = output_dir / "best_model"
    final_model_dir = output_dir / "final_model"
    trainer_tmp_dir = output_dir / "_trainer_tmp"

    raw_df, exploded_df, pair_df, train_ds, valid_ds = build_train_and_eval_pairs(
        cfg, train_file, test_file
    )

    model = build_sentence_transformer(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        pooling=cfg.sentence_pooling,
    )

    train_loss = cfg.train_loss.lower().strip()

    if train_loss == "cosent":
        loss = losses.CoSENTLoss(model)
        train_ds = train_ds.remove_columns(["label"])
        valid_ds_for_trainer = valid_ds.remove_columns(["label"])

        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=valid_ds["sentence1"],
            sentences2=valid_ds["sentence2"],
            scores=[float(x) for x in valid_ds["score"]],
            name="valid-sim",
        )
        metric_for_best_model = "eval_valid-sim_spearman_cosine"
        greater_is_better = True
    else:
        loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=2,
        )
        train_ds = train_ds.remove_columns(["score"])
        valid_ds_for_trainer = valid_ds.remove_columns(["score"])

        evaluator = BinaryClassificationEvaluator(
            sentences1=valid_ds["sentence1"],
            sentences2=valid_ds["sentence2"],
            labels=[int(x) for x in valid_ds["label"]],
            name="valid-binary",
        )
        metric_for_best_model = "eval_valid-binary_cosine_ap"
        greater_is_better = True

    args = SentenceTransformerTrainingArguments(
        output_dir=str(trainer_tmp_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        seed=cfg.seed,
        report_to="none",
        fp16=cfg.fp16,
        bf16=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds_for_trainer,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    meta = {k: getattr(cfg, k) for k in MLMConfig.__dataclass_fields__}
    meta.update(
        {
            "type": "sentence_transformer_domain_encoder",
            "train_file": str(train_file),
            "test_file": str(test_file),
            "raw_doc_count": int(len(raw_df)),
            "train_view_count": int(len(exploded_df)),
            "pair_count": int(len(pair_df)),
            "train_pair_count": int(len(train_ds)),
            "valid_pair_count": int(len(valid_ds)),
        }
    )

    for target_dir in [best_model_dir, final_model_dir]:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        save_encoder_only_from_sentence_transformer(model, target_dir, meta=meta)
        if cfg.save_sentence_transformer_artifacts:
            model.save(str(target_dir / "sentence_transformer"))

    for epoch_idx in range(checkpoint_every, cfg.num_train_epochs + 1, checkpoint_every):
        ckpt_dir = checkpoints_dir / f"epoch_{epoch_idx:03d}"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        save_encoder_only_from_sentence_transformer(
            model,
            ckpt_dir,
            meta={
                **meta,
                "epoch": epoch_idx,
                "type": "sentence_transformer_domain_encoder_checkpoint",
            },
        )

    metrics = {
        **meta,
        **{
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in eval_metrics.items()
        },
        "checkpoint_every_n_epochs": checkpoint_every,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if trainer_tmp_dir.exists():
        shutil.rmtree(trainer_tmp_dir, ignore_errors=True)

    components = {
        "output_dir": str(output_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "best_model_dir": str(best_model_dir),
        "final_model_dir": str(final_model_dir),
        "metrics_path": str(metrics_path),
        "model_name": cfg.model_name,
    }

    print("\n" + "=" * 56)
    print("Training finished.")
    print(f"  Raw docs      : {len(raw_df)}")
    print(f"  Train views   : {len(exploded_df)}")
    print(f"  Pair count    : {len(pair_df)}")
    print(f"  Best model    : {components['best_model_dir']}")
    print(f"  Final model   : {components['final_model_dir']}")
    print(f"  Metrics path  : {components['metrics_path']}")
    print("=" * 56)

    return components, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sentence embeddings for ruRoberta with text/label input."
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