"""Дообучение sentence-encoder на базе ruRoberta-large.

Энкодер обучается на парах (sentence1, sentence2) с одним из лосс-функций:
- MNR (MultipleNegativesRankingLoss) — рекомендуемый, использует только positive-пары;
- CoSENT — обучение по непрерывным score;
- Softmax — бинарная классификация пар.

Только positive-пары между чанками РАЗНЫХ документов одного класса
(см. data_utils.build_pair_dataframe).
"""

from __future__ import annotations

import tempfile
import argparse
import gc
import json
import random
import shutil
from dataclasses import fields, replace
from pathlib import Path
import os
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

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
from transformers import EarlyStoppingCallback, TrainerCallback

from bert_embeddings.config import MLMConfig, ensure_dir
from bert_embeddings.data_utils import (
    build_pair_dataframe,
    build_training_dataframe,
    explode_long_texts_for_training,
)


def cleanup_memory():
    """Освобождает кэши GPU/MPS между прогонами."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def set_seed(seed: int):
    """Фиксирует seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def build_sentence_transformer(
    model_name: str,
    max_length: int,
    pooling: str,
) -> SentenceTransformer:
    """Собирает SentenceTransformer из Transformer + Pooling + Normalize."""
    transformer = models.Transformer(
        model_name,
        max_seq_length=max_length,
        model_kwargs={"torch_dtype": "float32"},
    )

    pooling = pooling.lower().strip()
    # Новый API sentence-transformers: pooling_mode как строка.
    # Если установлена старая версия — fallback на старый kwargs-API.
    try:
        pooling_model = models.Pooling(
            transformer.get_embedding_dimension()
            if hasattr(transformer, "get_embedding_dimension")
            else transformer.get_word_embedding_dimension(),
            pooling_mode=pooling,  # "mean" | "max" | "cls"
        )
    except TypeError:
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
    """Сохраняет только энкодер (без pooling/normalize) в формате HuggingFace.

    Веса экспортируются с префиксом 'roberta.' — это позволяет потом загружать
    их прямо в RobertaModel (LongTextRobertaEmbedder).
    """
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


def build_train_and_eval_pairs(cfg: MLMConfig, train_file: str):
    """Строит train/valid пары: разворачивает документы в чанки и сэмплирует пары."""
    raw_df = build_training_dataframe(
        train_file,
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
        cross_document_positives_only=cfg.cross_document_positives_only,
    )
    if len(pair_df) == 0:
        raise ValueError("No training pairs were built from text/label data.")

    val_size = min(max(cfg.val_size, 0.01), 0.3)
    ds = Dataset.from_pandas(pair_df, preserve_index=False)
    split_ds = ds.train_test_split(test_size=val_size, seed=cfg.seed)
    return raw_df, exploded_df, pair_df, split_ds["train"], split_ds["test"]


class FreezeLowerLayersCallback(TrainerCallback):
    """Замораживает нижние N слоёв энкодера + embeddings на первой эпохе.

    Это снижает риск разрушения предобученных представлений на старте
    и ускоряет первую эпоху. После завершения первой эпохи все слои размораживаются.
    """

    def __init__(self, encoder, n_layers_to_freeze: int):
        self.encoder = encoder
        self.n = n_layers_to_freeze
        self._frozen = False

    def _freeze(self):
        for i, layer in enumerate(self.encoder.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = (i >= self.n)
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        self._frozen = True

    def _unfreeze_all(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._frozen = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.n > 0:
            self._freeze()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._frozen and (state.epoch or 0) >= 1.0:
            self._unfreeze_all()
        return control


class RollingResumeCheckpoint(TrainerCallback):
    """Перезаписывает один resume_checkpoint.pt после каждой эпохи.

    Запись идёт in-place в существующий файл: Google Drive воспринимает atomic
    rename как delete+create и отправляет предыдущую версию в корзину, что быстро
    забивает квоту. In-place запись этого избегает.
    """

    def __init__(self, path: Path, get_model, get_trainer):
        self.path = Path(path)
        self.get_model = get_model
        self.get_trainer = get_trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.get_trainer()
        if trainer is None:
            return control
        model = self.get_model()
        payload = {
            "epoch": int(round(state.epoch)) if state.epoch is not None else None,
            "global_step": state.global_step,
            "encoder_state_dict": {
                k: v.detach().cpu()
                for k, v in model[0].auto_model.state_dict().items()
            },
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None,
            "scheduler_state_dict": trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            torch.save(payload, f)
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                # На некоторых ФС (включая Drive-маунты) fsync может бросить —
                # данные уже записаны через flush, это не критично.
                pass
        return control


def run_from_params(
    train_file,
    output_dir,
    cfg: MLMConfig | None = None,
    **kwargs,
):
    """Основная точка входа обучения. Принимает train-файл и каталог сохранения.

    Дополнительные kwargs позволяют переопределять поля MLMConfig из ноутбука
    без явного создания cfg. Алиасы num_epochs/batch_size поддерживаются
    для совместимости с ранее использованным API.
    """
    if cfg is None:
        cfg = MLMConfig()

    if kwargs:
        if "num_epochs" in kwargs:
            kwargs["num_train_epochs"] = kwargs.pop("num_epochs")
        if "batch_size" in kwargs:
            kwargs["train_batch_size"] = kwargs.pop("batch_size")
        kwargs.pop("checkpoint_every_n_epochs", None)
        kwargs.pop("save_sentence_transformer_artifacts", None)

        valid_fields = MLMConfig.__dataclass_fields__
        cfg = replace(cfg, **{k: v for k, v in kwargs.items() if k in valid_fields})

    if not torch.cuda.is_available():
        cfg = replace(cfg, bf16=False, fp16=False)

    cleanup_memory()
    set_seed(cfg.seed)

    output_dir = ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.json"
    best_model_dir = output_dir / "best_model"
    resume_ckpt_path = output_dir / "resume_checkpoint.pt"
    # Trainer кладёт полные чекпоинты (~1.4 GB для ruRoberta-large) в output_dir
    # каждую эпоху. Пишем их в системный tmp, а не на Drive, чтобы не забить квоту.
    trainer_tmp_dir = Path(tempfile.mkdtemp(prefix="st_trainer_tmp_"))
    print(f"[bert_embeddings] Trainer tmp dir: {trainer_tmp_dir}")

    raw_df, exploded_df, pair_df, train_ds, valid_ds = build_train_and_eval_pairs(
        cfg, train_file
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

    elif train_loss == "mnr":
        # MNR-loss использует только positive-пары: ему нужно убрать negatives и score.
        pos_mask_train = [int(x) == 1 for x in train_ds["label"]]
        pos_mask_valid = [int(x) == 1 for x in valid_ds["label"]]
        train_ds = train_ds.select([i for i, m in enumerate(pos_mask_train) if m])
        valid_ds_for_trainer = valid_ds.select(
            [i for i, m in enumerate(pos_mask_valid) if m]
        )
        train_ds = train_ds.remove_columns(["score", "label"])
        valid_ds_for_trainer = valid_ds_for_trainer.remove_columns(["score", "label"])
        loss = losses.MultipleNegativesRankingLoss(model)
        evaluator = BinaryClassificationEvaluator(
            sentences1=valid_ds["sentence1"],
            sentences2=valid_ds["sentence2"],
            labels=[int(x) for x in valid_ds["label"]],
            name="valid-binary",
        )
        metric_for_best_model = "eval_valid-binary_cosine_ap"
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
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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
        bf16=cfg.bf16,
        tf32=cfg.tf32 if torch.cuda.is_available() else False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    resume_cb_holder = {"trainer": None}
    resume_cb = RollingResumeCheckpoint(
        path=resume_ckpt_path,
        get_model=lambda: model,
        get_trainer=lambda: resume_cb_holder["trainer"],
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
        resume_cb,
    ]
    if cfg.freeze_lower_layers > 0:
        callbacks.append(
            FreezeLowerLayersCallback(
                encoder=model[0].auto_model,
                n_layers_to_freeze=cfg.freeze_lower_layers,
            )
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds_for_trainer,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )
    resume_cb_holder["trainer"] = trainer

    trainer.train()
    eval_metrics = trainer.evaluate()

    meta = {k: getattr(cfg, k) for k in MLMConfig.__dataclass_fields__}
    meta.update(
        {
            "type": "sentence_transformer_domain_encoder",
            "train_file": str(train_file),
            "raw_doc_count": int(len(raw_df)),
            "train_view_count": int(len(exploded_df)),
            "pair_count": int(len(pair_df)),
            "train_pair_count": int(len(train_ds)),
            "valid_pair_count": int(len(valid_ds)),
        }
    )

    # Сохраняем только best_model — Trainer уже подтянул лучшие веса в память.
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    save_encoder_only_from_sentence_transformer(model, best_model_dir, meta=meta)

    metrics = {
        **meta,
        **{
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in eval_metrics.items()
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if trainer_tmp_dir.exists():
        shutil.rmtree(trainer_tmp_dir, ignore_errors=True)

    components = {
        "output_dir": str(output_dir),
        "best_model_dir": str(best_model_dir),
        "resume_checkpoint_path": str(resume_ckpt_path),
        "metrics_path": str(metrics_path),
        "model_name": cfg.model_name,
    }

    print("\n" + "=" * 56)
    print("Training finished.")
    print(f"  Raw docs        : {len(raw_df)}")
    print(f"  Train views     : {len(exploded_df)}")
    print(f"  Pair count      : {len(pair_df)}")
    print(f"  Best model      : {components['best_model_dir']}")
    print(f"  Resume ckpt     : {components['resume_checkpoint_path']}")
    print(f"  Metrics path    : {components['metrics_path']}")
    print("=" * 56)

    return components, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI: --train-file, --output-dir + все поля MLMConfig как опциональные флаги."""
    parser = argparse.ArgumentParser(
        description="Train sentence embeddings for ruRoberta with text/label input."
    )
    parser.add_argument("--train-file", type=str, required=True)
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
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
