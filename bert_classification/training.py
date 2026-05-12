import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)

from .config import ModelConfig, TrainConfig, DataConfig, PathConfig
from .data import (
    load_and_prepare_dataframes,
    build_label_mappings,
    attach_label_ids,
    compute_class_weights_tensor,
    build_tokenizer,
    build_dataset_dict,
)
from .metrics import compute_metrics
from .model import build_model, ChunkDataCollator
from .utils import ensure_dir, get_filtered_model_state_dict, set_global_seed, cleanup_memory


def build_checkpoint_payload(
    trainer: Trainer,
    epoch: int,
    model_cfg: ModelConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    class_weights_tensor: Optional[torch.Tensor] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "epoch": epoch,
        "model_state_dict": get_filtered_model_state_dict(trainer.model),
        "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer is not None else None,
        "scheduler_state_dict": trainer.lr_scheduler.state_dict() if trainer.lr_scheduler is not None else None,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "model_config": asdict(model_cfg),
        "metrics": metrics,
    }

    if class_weights_tensor is not None:
        payload["class_weights"] = class_weights_tensor.detach().cpu()

    return payload


def save_checkpoint_payload(payload: Dict[str, Any], checkpoint_path: str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def save_epoch_checkpoint(
    trainer: Trainer,
    output_dir: str,
    epoch: int,
    model_cfg: ModelConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    class_weights_tensor: Optional[torch.Tensor] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = build_checkpoint_payload(
        trainer=trainer,
        epoch=epoch,
        model_cfg=model_cfg,
        label2id=label2id,
        id2label=id2label,
        class_weights_tensor=class_weights_tensor,
        metrics=metrics,
    )
    checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch}.pt"
    saved_path = save_checkpoint_payload(payload, checkpoint_path)
    print(f"Saved checkpoint: {saved_path}")
    return saved_path


def save_best_training_checkpoint(
    trainer: Trainer,
    output_dir: str,
    epoch: int,
    model_cfg: ModelConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    class_weights_tensor: Optional[torch.Tensor] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = build_checkpoint_payload(
        trainer=trainer,
        epoch=epoch,
        model_cfg=model_cfg,
        label2id=label2id,
        id2label=id2label,
        class_weights_tensor=class_weights_tensor,
        metrics=metrics,
    )
    checkpoint_path = Path(output_dir) / "best_checkpoint.pt"
    saved_path = save_checkpoint_payload(payload, checkpoint_path)
    print(f"Updated best checkpoint: {saved_path}")
    return saved_path


def export_model_bundle(
    model_state_dict: Dict[str, torch.Tensor],
    tokenizer,
    output_dir: str,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    export_path = ensure_dir(output_dir)

    torch.save(model_state_dict, export_path / "pytorch_model.bin")
    tokenizer.save_pretrained(export_path)

    training_meta = {
        "model_config": asdict(model_cfg),
        "data_config": asdict(data_cfg),
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }

    if extra_config is not None:
        training_meta.update(extra_config)

    with open(export_path / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(training_meta, f, ensure_ascii=False, indent=2)

    print(f"Model weights saved to: {export_path / 'pytorch_model.bin'}")
    print(f"Tokenizer saved to: {export_path}")
    print(f"Meta saved to: {export_path / 'training_meta.json'}")
    print("Excluded from checkpoint: ['class_weights']")


def save_train_history(log_history, output_dir: str) -> Path:
    path = Path(output_dir) / "train_history.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)
    print(f"Train history saved to: {path}")
    return path


def load_recovery_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[Dict[str, Any], int]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"] + 1

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Resume from epoch: {start_epoch}")
    if ckpt.get("metrics") is not None:
        print(f"Stored metrics: {ckpt['metrics']}")

    return ckpt, start_epoch


class BestMetricTrackerCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: str,
        metric_name: str,
        greater_is_better: bool,
        model_cfg: ModelConfig,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        class_weights_tensor: Optional[torch.Tensor] = None,
    ):
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.model_cfg = model_cfg
        self.label2id = label2id
        self.id2label = id2label
        self.class_weights_tensor = class_weights_tensor

        self.trainer_ref = None
        self.best_metric = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_state_dict = None
        self.best_checkpoint_path = None

    def bind_trainer(self, trainer: Trainer) -> None:
        self.trainer_ref = trainer

    def _extract_latest_eval_metrics(self, state) -> Optional[Dict[str, Any]]:
        if not state.log_history:
            return None

        for item in reversed(state.log_history):
            if isinstance(item, dict) and ("eval_f1_macro" in item or "eval_balanced_accuracy" in item):
                return item
        return None

    def _is_better(self, current_metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return current_metric > self.best_metric
        return current_metric < self.best_metric

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer_ref is None:
            return control

        metrics = self._extract_latest_eval_metrics(state)
        if metrics is None:
            return control

        metric_key = f"eval_{self.metric_name}"
        if metric_key not in metrics:
            return control

        current_metric = float(metrics[metric_key])
        epoch_value = state.epoch
        if epoch_value is None:
            return control

        epoch_num = int(round(epoch_value))

        if self._is_better(current_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch_num
            self.best_metrics = dict(metrics)
            self.best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in get_filtered_model_state_dict(self.trainer_ref.model).items()
            }

            self.best_checkpoint_path = save_best_training_checkpoint(
                trainer=self.trainer_ref,
                output_dir=self.output_dir,
                epoch=epoch_num,
                model_cfg=self.model_cfg,
                label2id=self.label2id,
                id2label=self.id2label,
                class_weights_tensor=self.class_weights_tensor,
                metrics=metrics,
            )

        return control


class EpochIntervalCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        every_n_epochs: int,
        output_dir: str,
        model_cfg: ModelConfig,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        class_weights_tensor: Optional[torch.Tensor] = None,
    ):
        self.every_n_epochs = every_n_epochs
        self.output_dir = output_dir
        self.model_cfg = model_cfg
        self.label2id = label2id
        self.id2label = id2label
        self.class_weights_tensor = class_weights_tensor
        self.trainer_ref = None

    def bind_trainer(self, trainer: Trainer) -> None:
        self.trainer_ref = trainer

    def _extract_latest_eval_metrics(self, state) -> Optional[Dict[str, Any]]:
        if not state.log_history:
            return None

        for item in reversed(state.log_history):
            if isinstance(item, dict) and ("eval_f1_macro" in item or "eval_balanced_accuracy" in item):
                return item
        return None

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_value = state.epoch
        if epoch_value is None:
            return control

        epoch_num = int(round(epoch_value))
        if abs(epoch_value - epoch_num) > 1e-8:
            return control

        if epoch_num % self.every_n_epochs != 0:
            return control

        if self.trainer_ref is None:
            return control

        metrics = self._extract_latest_eval_metrics(state)

        save_epoch_checkpoint(
            trainer=self.trainer_ref,
            output_dir=self.output_dir,
            epoch=epoch_num,
            model_cfg=self.model_cfg,
            label2id=self.label2id,
            id2label=self.id2label,
            class_weights_tensor=self.class_weights_tensor,
            metrics=metrics,
        )
        return control


class WeightedChunkTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)

            encoder_params_decay = []
            encoder_params_no_decay = []
            head_params_decay = []
            head_params_no_decay = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                is_decay = name in decay_parameters
                is_encoder = name.startswith("roberta.")

                if is_encoder and is_decay:
                    encoder_params_decay.append(param)
                elif is_encoder and not is_decay:
                    encoder_params_no_decay.append(param)
                elif not is_encoder and is_decay:
                    head_params_decay.append(param)
                else:
                    head_params_no_decay.append(param)

            optimizer_grouped_parameters = [
                {"params": encoder_params_decay, "weight_decay": self.custom_weight_decay, "lr": self.lr_encoder},
                {"params": encoder_params_no_decay, "weight_decay": 0.0, "lr": self.lr_encoder},
                {"params": head_params_decay, "weight_decay": self.custom_weight_decay, "lr": self.lr_head},
                {"params": head_params_no_decay, "weight_decay": 0.0, "lr": self.lr_head},
            ]

            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        return self.optimizer


def attach_trainer_hparams(
    trainer: WeightedChunkTrainer,
    train_cfg: TrainConfig,
) -> WeightedChunkTrainer:
    trainer.lr_encoder = train_cfg.lr_encoder
    trainer.lr_head = train_cfg.lr_head
    trainer.custom_weight_decay = train_cfg.weight_decay
    return trainer


def build_training_arguments(
    path_cfg: PathConfig,
    train_cfg: TrainConfig,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=path_cfg.output_dir,
        save_strategy="no",
        eval_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        num_train_epochs=train_cfg.num_epochs,
        learning_rate=train_cfg.lr_encoder,
        lr_scheduler_type="cosine",
        weight_decay=train_cfg.weight_decay,
        warmup_steps=train_cfg.warmup_steps,
        fp16=torch.cuda.is_available(),
        seed=train_cfg.seed,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        load_best_model_at_end=False,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=True,
        max_grad_norm=train_cfg.max_grad_norm,
        remove_unused_columns=False,
        disable_tqdm=False,
    )


def build_trainer(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    path_cfg: PathConfig,
    dataset,
    data_collator,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    class_weights_tensor: Optional[torch.Tensor],
):
    def model_init():
        return build_model(
            model_cfg=model_cfg,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            class_weights_tensor=class_weights_tensor,
        )

    training_args = build_training_arguments(path_cfg, train_cfg)

    best_callback = BestMetricTrackerCallback(
        output_dir=path_cfg.output_dir,
        metric_name=train_cfg.metric_for_best_model,
        greater_is_better=True,
        model_cfg=model_cfg,
        label2id=label2id,
        id2label=id2label,
        class_weights_tensor=class_weights_tensor,
    )

    epoch_ckpt_callback = EpochIntervalCheckpointCallback(
        every_n_epochs=train_cfg.checkpoint_every_n_epochs,
        output_dir=path_cfg.output_dir,
        model_cfg=model_cfg,
        label2id=label2id,
        id2label=id2label,
        class_weights_tensor=class_weights_tensor,
    )

    trainer = WeightedChunkTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience),
            best_callback,
            epoch_ckpt_callback,
        ],
    )

    trainer = attach_trainer_hparams(trainer, train_cfg)
    best_callback.bind_trainer(trainer)
    epoch_ckpt_callback.bind_trainer(trainer)

    return trainer, best_callback


def prepare_training_components(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    path_cfg: PathConfig,
):
    set_global_seed(train_cfg.seed)
    cleanup_memory()

    tokenizer = build_tokenizer(model_cfg)

    train_df, test_df = load_and_prepare_dataframes(
        train_file=path_cfg.train_file,
        test_file=path_cfg.test_file,
        text_col=data_cfg.text_col,
        label_col=data_cfg.label_col,
    )

    labels, label2id, id2label = build_label_mappings(
        train_df=train_df,
        test_df=test_df,
        label_col=data_cfg.label_col,
    )

    train_df, test_df = attach_label_ids(
        train_df=train_df,
        test_df=test_df,
        label_col=data_cfg.label_col,
        label2id=label2id,
    )

    class_weights_tensor = compute_class_weights_tensor(
        train_label_ids=train_df["label_id"].values,
        num_labels=len(labels),
    )

    dataset = build_dataset_dict(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
    )

    data_collator = ChunkDataCollator(
        max_chunks=model_cfg.max_chunks,
        max_length=model_cfg.max_length,
    )

    trainer, best_callback = build_trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        path_cfg=path_cfg,
        dataset=dataset,
        data_collator=data_collator,
        label2id=label2id,
        id2label=id2label,
        class_weights_tensor=class_weights_tensor,
    )

    return {
        "tokenizer": tokenizer,
        "train_df": train_df,
        "test_df": test_df,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "class_weights_tensor": class_weights_tensor,
        "dataset": dataset,
        "data_collator": data_collator,
        "trainer": trainer,
        "best_callback": best_callback,
    }


def run_training_pipeline(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    path_cfg: PathConfig,
):
    ensure_dir(path_cfg.output_dir)

    components = prepare_training_components(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        path_cfg=path_cfg,
    )

    trainer = components["trainer"]
    dataset = components["dataset"]
    best_callback = components["best_callback"]

    trainer.train()

    eval_metrics = trainer.evaluate(dataset["validation"])

    print("\nFINAL METRICS")
    print(f"balanced_accuracy: {eval_metrics['eval_balanced_accuracy']:.6f}")
    print(f"f1_macro:          {eval_metrics['eval_f1_macro']:.6f}")

    if best_callback.best_state_dict is None:
        best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in get_filtered_model_state_dict(trainer.model).items()
        }
        best_epoch = None
        best_metrics = eval_metrics
        best_checkpoint_path = None
    else:
        best_state_dict = best_callback.best_state_dict
        best_epoch = best_callback.best_epoch
        best_metrics = best_callback.best_metrics
        best_checkpoint_path = str(best_callback.best_checkpoint_path) if best_callback.best_checkpoint_path else None

    print("\nBEST METRICS")
    print(f"best_epoch:         {best_epoch}")
    print(f"balanced_accuracy:  {best_metrics['eval_balanced_accuracy']:.6f}")
    print(f"f1_macro:           {best_metrics['eval_f1_macro']:.6f}")
    if best_checkpoint_path is not None:
        print(f"best_checkpoint:    {best_checkpoint_path}")

    save_train_history(trainer.state.log_history, path_cfg.output_dir)

    export_model_bundle(
        model_state_dict=best_state_dict,
        tokenizer=components["tokenizer"],
        output_dir=path_cfg.output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        label2id=components["label2id"],
        id2label=components["id2label"],
        extra_config={
            "train_config": asdict(train_cfg),
            "path_config": asdict(path_cfg),
            "final_eval_metrics": eval_metrics,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_checkpoint_path": best_checkpoint_path,
        },
    )

    return components, eval_metrics