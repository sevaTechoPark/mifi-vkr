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
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))
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


def run_from_params(
    train_file,
    test_file,
    output_dir,
    model_name="ai-forever/ruRoberta-large",
    text_col="text",
    num_epochs=15,
    checkpoint_every_n_epochs=3,
    batch_size=4,
    eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    mlm_probability=0.15,
    max_length=512,
    val_size=0.02,
    logging_steps=100,
    seed=42,
):
    cleanup_memory()

    cfg = MLMConfig(
        model_name=model_name,
        text_col=text_col,
        max_length=max_length,
        mlm_probability=mlm_probability,
        train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    output_dir = ensure_dir(output_dir)
    final_model_dir = ensure_dir(output_dir / "final_model")
    metrics_path = output_dir / "metrics.json"

    raw_df = build_mlm_corpus(train_file, test_file, text_col=cfg.text_col)
    print("MLM corpus size:", len(raw_df))

    full_ds = Dataset.from_pandas(raw_df, preserve_index=False)
    split_ds = full_ds.train_test_split(test_size=val_size, seed=seed)
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
            "val_size": val_size,
            "train_file": str(train_file),
            "test_file": str(test_file),
            "seed": seed,
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer_tmp"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
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
        save_every_n_epochs=checkpoint_every_n_epochs,
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
        "train_file": str(train_file),
        "test_file": str(test_file),
        "seed": seed,
        "val_size": val_size,
        "epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "eval_loss": eval_metrics.get("eval_loss"),
        "perplexity": math.exp(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else None,
        "best_eval_loss": checkpoint_callback.best_eval_loss,
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
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

    components = {
        "output_dir": str(output_dir),
        "checkpoints_dir": str(output_dir / "checkpoints"),
        "best_model_dir": str(output_dir / "best_model"),
        "final_model_dir": str(final_model_dir),
        "metrics_path": str(metrics_path),
        "model_name": cfg.model_name,
        "tokenizer_name": cfg.model_name,
    }

    print("Training finished.")
    print("Best model dir:", components["best_model_dir"])
    print("Final model dir:", components["final_model_dir"])
    print("Metrics path:", components["metrics_path"])

    return components, metrics