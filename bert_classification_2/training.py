"""Цикл обучения и сохранения результатов.

Особенности пайплайна:
  - HF Trainer не пишет чекпоинтов сам (save_strategy="no");
  - лучший state_dict копируется в RAM коллбеком BestMetricInMemoryCallback —
    после train() веса откатываются к лучшей эпохе для финального evaluate;
  - resume-чекпоинт (веса + оптимизатор + scheduler) пишется отдельным
    коллбеком в один и тот же файл resume_checkpoint.pt после каждой эпохи —
    это позволяет дообучить с последней эпохи, не плодя файлы;
  - класс-веса рассчитываются по train и попадают в CrossEntropyLoss модели.
"""

import json
import os
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
from .utils import (
    ensure_dir,
    get_filtered_model_state_dict,
    load_state_dict_into_model,
    set_global_seed,
    cleanup_memory,
)


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _is_a100_or_better() -> bool:
    """True, если CUDA-устройство поддерживает bf16 нативно (Ampere и новее)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


def load_recovery_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[Dict[str, Any], int]:
    """Загружает resume-чекпоинт (веса + оптимизатор + scheduler).

    Возвращает (полный payload чекпоинта, номер эпохи для продолжения обучения).
    Если оптимизатор/scheduler не переданы — их состояние просто игнорируется.
    """
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


# ---------------------------------------------------------------------------
# Коллбеки
# ---------------------------------------------------------------------------

class BestMetricInMemoryCallback(TrainerCallback):
    """Держит лучший state_dict в RAM (на CPU), без записи на диск.

    Используется, чтобы отвязать выбор лучшей модели от файлового чекпоинтинга
    и избежать дублирования весов на диске. После trainer.train() веса
    откатываются на лучшую эпоху явным вызовом load_state_dict_into_model.
    """

    def __init__(self, metric_name: str, greater_is_better: bool):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better

        self.trainer_ref: Optional[Trainer] = None
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def bind_trainer(self, trainer: Trainer) -> None:
        self.trainer_ref = trainer

    def _extract_latest_eval_metrics(self, state) -> Optional[Dict[str, Any]]:
        """Достаёт последнюю запись с метриками из истории логов Trainer'а."""
        if not state.log_history:
            return None
        for item in reversed(state.log_history):
            if isinstance(item, dict) and (
                "eval_f1_macro" in item or "eval_balanced_accuracy" in item
            ):
                return item
        return None

    def _is_better(self, current_metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return current_metric > self.best_metric
        return current_metric < self.best_metric

    def on_evaluate(self, args, state, control, **kwargs):
        # Срабатывает после on_epoch_end: к этому моменту метрики эпохи уже
        # лежат в state.log_history. Если использовать on_epoch_end — метрик
        # там ещё нет, и сравнение становится невозможным.
        if self.trainer_ref is None:
            return control
        metrics = self._extract_latest_eval_metrics(state)
        if metrics is None:
            return control
        metric_key = f"eval_{self.metric_name}"
        if metric_key not in metrics:
            return control
        current_metric = float(metrics[metric_key])
        if not self._is_better(current_metric):
            return control

        epoch_value = state.epoch
        epoch_num = int(round(epoch_value)) if epoch_value is not None else None
        self.best_metric = current_metric
        self.best_epoch = epoch_num
        self.best_metrics = dict(metrics)
        # Копируем веса на CPU, чтобы не держать дубль на GPU.
        self.best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in get_filtered_model_state_dict(self.trainer_ref.model).items()
        }
        return control


class RollingResumeCheckpointCallback(TrainerCallback):
    """Перезаписывает один и тот же resume_checkpoint.pt после каждой эпохи.

    Запись делается in-place в существующий файл (truncate + write поверх того
    же inode), а не через atomic rename. Это специально: Google Drive
    интерпретирует rename как delete+create и отправляет предыдущую версию
    в корзину, что быстро забивает место. При in-place записи такой проблемы нет.

    Trade-off: если процесс упадёт во время записи, чекпоинт окажется
    повреждённым — но он переписывается каждую эпоху, так что в худшем случае
    мы потеряем одну эпоху обучения.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.trainer_ref: Optional[Trainer] = None

    def bind_trainer(self, trainer: Trainer) -> None:
        self.trainer_ref = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer_ref is None or state.epoch is None:
            return control
        epoch_num = int(round(state.epoch))
        payload = {
            "epoch": epoch_num,
            "global_step": state.global_step,
            "model_state_dict": get_filtered_model_state_dict(self.trainer_ref.model),
            "optimizer_state_dict": (
                self.trainer_ref.optimizer.state_dict()
                if self.trainer_ref.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.trainer_ref.lr_scheduler.state_dict()
                if self.trainer_ref.lr_scheduler is not None else None
            ),
        }
        target = Path(self.output_dir) / "resume_checkpoint.pt"
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "wb") as f:
            torch.save(payload, f)
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                # На некоторых FS (включая часть Drive-маунтов) fsync не работает —
                # это не критично, данные уже отправлены на запись через flush().
                pass
        return control


# ---------------------------------------------------------------------------
# Trainer с раздельным LR для энкодера и головы
# ---------------------------------------------------------------------------

class WeightedChunkTrainer(Trainer):
    """Trainer с двумя независимыми learning rate'ами: для энкодера и для головы.

    Голова обычно обучается стабильнее на более высоком LR, а энкодер требует
    более бережного дообучения. Раздельные LR задаются через attach_trainer_hparams.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            # Разбираем параметры по 4 группам: (encoder|head) x (decay|no_decay).
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


def attach_trainer_hparams(trainer: WeightedChunkTrainer, train_cfg: TrainConfig) -> WeightedChunkTrainer:
    """Прокидывает гиперпараметры в Trainer как атрибуты, чтобы их видела create_optimizer."""
    trainer.lr_encoder = train_cfg.lr_encoder
    trainer.lr_head = train_cfg.lr_head
    trainer.custom_weight_decay = train_cfg.weight_decay
    return trainer


# ---------------------------------------------------------------------------
# TrainingArguments
# ---------------------------------------------------------------------------

def build_training_arguments(path_cfg: PathConfig, train_cfg: TrainConfig) -> TrainingArguments:
    """Собирает TrainingArguments с автодетектом bf16/fp16 и поддержкой warmup_ratio."""
    use_bf16 = train_cfg.bf16 and _is_a100_or_better()
    use_fp16 = (not use_bf16) and train_cfg.fp16_fallback_on_non_a100 and torch.cuda.is_available()

    # Если задано абсолютное число warmup_steps — используем его, иначе долю.
    warmup_kwargs = {}
    if train_cfg.warmup_steps and train_cfg.warmup_steps > 0:
        warmup_kwargs["warmup_steps"] = train_cfg.warmup_steps
    else:
        warmup_kwargs["warmup_ratio"] = train_cfg.warmup_ratio

    return TrainingArguments(
        output_dir=path_cfg.output_dir,
        # Trainer ничего не сохраняет сам — сохранение делает RollingResumeCheckpointCallback.
        save_strategy="no",
        eval_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        num_train_epochs=train_cfg.num_epochs,
        # learning_rate здесь — формальный (реально LR задаются в WeightedChunkTrainer),
        # но HF Trainer требует это поле для построения lr_scheduler'а.
        learning_rate=train_cfg.lr_encoder,
        lr_scheduler_type="cosine",
        weight_decay=train_cfg.weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=train_cfg.tf32 if torch.cuda.is_available() else False,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        seed=train_cfg.seed,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        # Лучшую модель восстанавливаем сами из BestMetricInMemoryCallback.
        load_best_model_at_end=False,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=True,
        max_grad_norm=train_cfg.max_grad_norm,
        remove_unused_columns=False,
        disable_tqdm=False,
        **warmup_kwargs,
    )


# ---------------------------------------------------------------------------
# Сборка Trainer
# ---------------------------------------------------------------------------

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
    """Собирает Trainer вместе с коллбеками; возвращает (trainer, best_callback)."""

    def model_init():
        # model_init используется HF Trainer для воссоздания модели при пересиде
        # обучения (например, при hyperparameter search). Здесь же мы используем
        # его как обычный конструктор — он будет вызван один раз.
        model = build_model(
            model_cfg=model_cfg,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            class_weights_tensor=class_weights_tensor,
        )
        freeze_stats = model.freeze_lower_layers(
            freeze_encoder_layers=model_cfg.freeze_encoder_layers,
            freeze_embeddings=model_cfg.freeze_embeddings,
        )
        print(
            f"[freeze] frozen_encoder_layers={freeze_stats['frozen_encoder_layers']}, "
            f"frozen_embeddings={freeze_stats['frozen_embeddings']}, "
            f"trainable={freeze_stats['trainable_params']/1e6:.1f}M / "
            f"{freeze_stats['total_params']/1e6:.1f}M "
            f"({freeze_stats['trainable_ratio']*100:.1f}%)"
        )
        return model

    training_args = build_training_arguments(path_cfg, train_cfg)

    best_callback = BestMetricInMemoryCallback(
        metric_name=train_cfg.metric_for_best_model,
        greater_is_better=True,
    )
    resume_ckpt_callback = RollingResumeCheckpointCallback(output_dir=path_cfg.output_dir)

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
            resume_ckpt_callback,
        ],
    )
    trainer = attach_trainer_hparams(trainer, train_cfg)
    best_callback.bind_trainer(trainer)
    resume_ckpt_callback.bind_trainer(trainer)
    return trainer, best_callback


# ---------------------------------------------------------------------------
# Подготовка всего пайплайна и его запуск
# ---------------------------------------------------------------------------

def prepare_training_components(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    path_cfg: PathConfig,
):
    """Готовит все объекты для обучения и возвращает словарь компонентов.

    Удобно вызывать отдельно, если нужно посмотреть на промежуточные результаты
    (label2id, токенизированный датасет, веса классов) без запуска train().
    """
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
        train_df=train_df, test_df=test_df, label_col=data_cfg.label_col,
    )
    train_df, test_df = attach_label_ids(
        train_df=train_df, test_df=test_df,
        label_col=data_cfg.label_col, label2id=label2id,
    )
    class_weights_tensor = compute_class_weights_tensor(
        train_label_ids=train_df["label_id"].values,
        num_labels=len(labels),
    )
    dataset = build_dataset_dict(
        train_df=train_df, test_df=test_df,
        tokenizer=tokenizer, model_cfg=model_cfg, data_cfg=data_cfg,
    )
    data_collator = ChunkDataCollator(
        max_chunks=model_cfg.max_chunks,
        max_length=model_cfg.max_length,
    )
    trainer, best_callback = build_trainer(
        model_cfg=model_cfg, train_cfg=train_cfg, path_cfg=path_cfg,
        dataset=dataset, data_collator=data_collator,
        label2id=label2id, id2label=id2label,
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
    """Полный цикл: подготовка → train → откат на лучшую эпоху → finальный evaluate → сохранение метрик."""
    ensure_dir(path_cfg.output_dir)

    components = prepare_training_components(
        model_cfg=model_cfg, train_cfg=train_cfg,
        data_cfg=data_cfg, path_cfg=path_cfg,
    )

    trainer = components["trainer"]
    dataset = components["dataset"]
    best_callback = components["best_callback"]

    trainer.train()

    # Восстанавливаем лучший state_dict в память перед финальным evaluate.
    # strict=False обязательно: в модели есть buffer `class_weights`,
    # а в сохранённом state_dict его нет (он отфильтрован сознательно).
    if best_callback.best_state_dict is not None:
        load_state_dict_into_model(trainer.model, best_callback.best_state_dict)

    eval_metrics = trainer.evaluate(dataset["validation"])

    print("\nFINAL METRICS (best epoch in-memory weights)")
    print(f"balanced_accuracy: {eval_metrics['eval_balanced_accuracy']:.6f}")
    print(f"f1_macro:          {eval_metrics['eval_f1_macro']:.6f}")
    print(f"best_epoch:        {best_callback.best_epoch}")

    # На диск пишем только метрики и конфиги — никаких весов и истории.
    # Веса доступны через resume_checkpoint.pt (его пишет отдельный коллбек).
    metrics_payload = {
        "best_epoch": best_callback.best_epoch,
        "best_metrics": best_callback.best_metrics,
        "final_eval_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in eval_metrics.items()
        },
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }

    metrics_path = Path(path_cfg.output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nmetrics.json: {metrics_path}")
    resume_path = Path(path_cfg.output_dir) / "resume_checkpoint.pt"
    if resume_path.exists():
        print(f"resume_ckpt:  {resume_path}")

    return components, eval_metrics
