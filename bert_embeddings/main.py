import gc
import json
import math
import shutil
from dataclasses import replace
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
    EarlyStoppingCallback,
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
    with open(save_path / "mlm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)


class CustomMLMCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, save_every_n_epochs=3, tokenizer=None, meta_fn=None):
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = ensure_dir(self.output_dir / "checkpoints")
        self.best_model_dir = self.output_dir / "best_model"
        self.tokenizer = tokenizer
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))
        self.best_eval_loss = float("inf")
        self.best_epoch = None
        self.meta_fn = meta_fn

    def _save_best_model(self, model, eval_loss, epoch):
        if self.best_model_dir.exists():
            shutil.rmtree(self.best_model_dir)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        meta = (self.meta_fn() if self.meta_fn else {})
        meta.update({
            "type": "mlm_domain_encoder_best",
            "best_eval_loss": eval_loss,
            "best_epoch": epoch,
        })
        save_encoder_only_from_mlm(model, self.best_model_dir, meta=meta)
        if self.tokenizer:
            self.tokenizer.save_pretrained(self.best_model_dir)

    def _save_epoch_checkpoint(self, model, epoch):
        ckpt_dir = self.checkpoints_dir / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta = (self.meta_fn() if self.meta_fn else {})
        meta.update({"type": "mlm_domain_encoder_checkpoint", "epoch": epoch})
        save_encoder_only_from_mlm(model, ckpt_dir, meta=meta)
        if self.tokenizer:
            self.tokenizer.save_pretrained(ckpt_dir)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if state.epoch is None or metrics is None:
            return control

        epoch_num = int(round(state.epoch))
        eval_loss = metrics.get("eval_loss")

        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_epoch = epoch_num
            self._save_best_model(model, eval_loss, epoch_num)

        if epoch_num % self.save_every_n_epochs == 0:
            self._save_epoch_checkpoint(model, epoch_num)

        return control


def run_from_params(
    train_file,
    test_file,
    output_dir,
    cfg: MLMConfig | None = None,
    **kwargs,
):
    """
    Дефолты живут только в MLMConfig (config.py).

    Переопределение через kwargs (для вызова из Python):
        run_from_params(..., num_epochs=40, learning_rate=2e-5)

    Или через готовый cfg:
        run_from_params(..., cfg=MLMConfig(num_train_epochs=40))

    Алиасы kwargs для обратной совместимости:
        num_epochs                → num_train_epochs
        batch_size                → train_batch_size
        checkpoint_every_n_epochs → отдельный аргумент (не поле MLMConfig)
    """
    if cfg is None:
        cfg = MLMConfig()

    checkpoint_every = cfg.checkpoint_every_n_epochs

    if kwargs:
        # алиасы
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

    output_dir = ensure_dir(output_dir)
    final_model_dir = ensure_dir(output_dir / "final_model")
    metrics_path = output_dir / "metrics.json"

    raw_df = build_mlm_corpus(train_file, test_file, text_col=cfg.text_col)
    print(f"MLM corpus size: {len(raw_df)}")

    full_ds = Dataset.from_pandas(raw_df, preserve_index=False)
    split_ds = full_ds.train_test_split(test_size=cfg.val_size, seed=cfg.seed)
    train_ds, valid_ds = split_ds["train"], split_ds["test"]

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

    model = RobertaForMaskedLM.from_pretrained(
        cfg.model_name,
        config=RobertaConfig.from_pretrained(cfg.model_name),
    )

    def build_meta():
        return {k: getattr(cfg, k) for k in MLMConfig.__dataclass_fields__}

    # early stopping по умолчанию включён, если patience > 0
    use_early_stopping = (
        cfg.early_stopping_patience is not None
        and cfg.early_stopping_patience > 0
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer_tmp"),
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch" if use_early_stopping else "no",
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        report_to="none",
        fp16=cfg.fp16,
        seed=cfg.seed,
    )

    callbacks = [
        CustomMLMCheckpointCallback(
            output_dir=output_dir,
            save_every_n_epochs=checkpoint_every,
            tokenizer=tokenizer,
            meta_fn=build_meta,
        )
    ]
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience
            )
        )

    checkpoint_callback = callbacks[0]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    metrics = {
        **build_meta(),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "eval_loss": eval_metrics.get("eval_loss"),
        "perplexity": math.exp(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else None,
        "best_eval_loss": checkpoint_callback.best_eval_loss,
        "best_epoch": checkpoint_callback.best_epoch,
        "checkpoint_every_n_epochs": checkpoint_every,
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
    }

    best_ppl = (
        math.exp(checkpoint_callback.best_eval_loss)
        if checkpoint_callback.best_eval_loss < float("inf")
        else None
    )
    print("\n" + "=" * 56)
    print("Training finished.")
    print(
        f"  Best model  → epoch {checkpoint_callback.best_epoch}, "
        f"eval_loss={checkpoint_callback.best_eval_loss:.6f}, "
        f"perplexity={best_ppl:.4f}"
    )
    print(
        f"  Final model → eval_loss={metrics['eval_loss']:.6f}, "
        f"perplexity={metrics['perplexity']:.4f}"
    )
    print(f"  Best model dir : {components['best_model_dir']}")
    print(f"  Final model dir: {components['final_model_dir']}")
    print(f"  Metrics path   : {components['metrics_path']}")
    print("=" * 56)

    return components, metrics