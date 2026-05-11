import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from bert_embeddings.config import MLMConfig, ensure_dir
from bert_embeddings.data_utils import build_mlm_corpus


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_encoder_only_from_mlm(model, save_dir, meta=None):
    save_path = ensure_dir(save_dir)
    torch.save(model.state_dict(), save_path / "pytorch_model.bin")
    meta = meta or {}
    with open(save_path / "mlm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class CustomMLMCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, save_every_n_epochs=3, tokenizer=None, meta_fn=None):
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = ensure_dir(self.output_dir / "checkpoints")
        self.best_model_dir = self.output_dir / "best_model"
        self.tokenizer = tokenizer
        self.save_every_n_epochs = save_every_n_epochs
        self.best_eval_loss = float("inf")
        self.meta_fn = meta_fn

    def _save_best_model(self, model, eval_loss, epoch):
        if self.best_model_dir.exists():
            shutil.rmtree(self.best_model_dir)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        meta = {}
        if self.meta_fn is not None:
            meta = self.meta_fn()
        meta.update({
            "type": "mlm_domain_encoder_best",
            "best_eval_loss": eval_loss,
            "best_epoch": epoch,
        })

        save_encoder_only_from_mlm(model, self.best_model_dir, meta=meta)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.best_model_dir)

    def _save_epoch_checkpoint(self, model, epoch):
        ckpt_dir = self.checkpoints_dir / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        meta = {}
        if self.meta_fn is not None:
            meta = self.meta_fn()
        meta.update({
            "type": "mlm_domain_encoder_checkpoint",
            "epoch": epoch,
        })

        save_encoder_only_from_mlm(model, ckpt_dir, meta=meta)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if state.epoch is None or metrics is None:
            return control

        epoch_num = int(round(state.epoch))
        eval_loss = metrics.get("eval_loss")

        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self._save_best_model(model, eval_loss, epoch_num)

        if epoch_num % self.save_every_n_epochs == 0:
            self._save_epoch_checkpoint(model, epoch_num)

        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="ai-forever/ruRoberta-large")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--val_size", type=float, default=0.02)
    parser.add_argument("--save_every_n_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cleanup_memory()

    cfg = MLMConfig(
        model_name=args.model_name,
        text_col=args.text_col,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    output_dir = ensure_dir(args.output_dir)
    final_model_dir = ensure_dir(output_dir / "final_model")
    metrics_path = output_dir / "metrics.json"

    raw_df = build_mlm_corpus(args.train_file, args.test_file, text_col=cfg.text_col)
    print("MLM corpus size:", len(raw_df))

    full_ds = Dataset.from_pandas(raw_df, preserve_index=False)
    split_ds = full_ds.train_test_split(test_size=args.val_size, seed=args.seed)
    train_ds = split_ds["train"]
    valid_ds = split_ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize_for_mlm(batch):
        return tokenizer(batch[cfg.text_col], truncation=True, max_length=cfg.max_length)

    train_ds = train_ds.map(tokenize_for_mlm, batched=True, remove_columns=[cfg.text_col])
    valid_ds = valid_ds.map(tokenize_for_mlm, batched=True, remove_columns=[cfg.text_col])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability,
    )

    model_config = RobertaConfig.from_pretrained(cfg.model_name)
    model = RobertaForMaskedLM.from_pretrained(cfg.model_name, config=model_config)

    def build_meta():
        return {
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
            "text_col": cfg.text_col,
            "epochs": cfg.num_train_epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "val_size": args.val_size,
            "train_file": args.train_file,
            "test_file": args.test_file,
            "seed": args.seed,
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer_tmp"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=cfg.fp16,
        seed=cfg.seed,
    )

    checkpoint_callback = CustomMLMCheckpointCallback(
        output_dir=output_dir,
        save_every_n_epochs=args.save_every_n_epochs,
        tokenizer=tokenizer,
        meta_fn=build_meta,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        callbacks=[checkpoint_callback],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    metrics = {
        "train_file": args.train_file,
        "test_file": args.test_file,
        "seed": args.seed,
        "val_size": args.val_size,
        "epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "eval_loss": eval_metrics.get("eval_loss"),
        "perplexity": math.exp(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else None,
        "best_eval_loss": checkpoint_callback.best_eval_loss,
        "save_every_n_epochs": args.save_every_n_epochs,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    save_encoder_only_from_mlm(
        trainer.model,
        final_model_dir,
        meta={
            **build_meta(),
            "type": "mlm_domain_encoder_final",
            "final_eval_loss": metrics["eval_loss"],
            "final_perplexity": metrics["perplexity"],
        },
    )

    print("Training finished.")
    print("Metrics saved to:", metrics_path)
    print("Best model dir:", output_dir / "best_model")
    print("Final model dir:", final_model_dir)
    print("Checkpoints dir:", output_dir / "checkpoints")


if __name__ == "__main__":
    main()