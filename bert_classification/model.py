"""Модель и data collator для классификации длинных документов.

Архитектура — RoBERTa с чанкованием на уровне документа:
  - документ разбит на N чанков по max_length токенов;
  - каждый чанк проходит через энкодер независимо;
  - токенные представления усредняются с учётом attention mask (token mean-pool);
  - полученные эмбеддинги чанков усредняются с учётом валидной длины документа
    (chunk mean-pool по num_chunks);
  - итоговый эмбеддинг документа подаётся в голову LayerNorm → Dropout → Linear.
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

from .config import ModelConfig


class ChunkDataCollator:
    """Собирает батч из уже токенизированных документов.

    На вход приходят семплы фиксированной формы (max_chunks, max_length) —
    задача коллатора лишь сложить их в тензоры. Ассерты по форме служат страховкой
    на случай рассинхронизации с data.py.
    """

    def __init__(self, max_chunks: int, max_length: int):
        self.max_chunks = max_chunks
        self.max_length = max_length

    def __call__(self, features):
        for i, f in enumerate(features):
            assert len(f["input_ids"]) == self.max_chunks, (
                f"sample {i}: number of chunks != max_chunks"
            )
            assert all(len(x) == self.max_length for x in f["input_ids"]), (
                f"sample {i}: input_ids chunk length != max_length"
            )
            assert all(len(x) == self.max_length for x in f["attention_mask"]), (
                f"sample {i}: attention_mask chunk length != max_length"
            )

        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        num_chunks = torch.tensor([f["num_chunks"] for f in features], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "num_chunks": num_chunks,
        }


class ChunkMeanPoolRobertaClassifier(nn.Module):
    """Классификатор с двойным mean-pool: внутри чанка по токенам, между чанками по документу.

    Голова намеренно компактна (LayerNorm → Dropout → Linear) — на малых выборках
    более глубокие головы переобучаются. Класс-веса передаются как `weight` в
    CrossEntropyLoss, что эквивалентно балансировочной перевзвешивающей функции потерь.
    Gradient checkpointing включается через TrainingArguments (а не в __init__) —
    это совместимо с актуальным HF Trainer.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        num_labels: int,
        id2label=None,
        label2id=None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model_cfg = model_cfg

        self.config = RobertaConfig.from_pretrained(
            model_cfg.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            output_hidden_states=False,
        )
        self.roberta = RobertaModel.from_pretrained(model_cfg.model_name, config=self.config)

        # При gradient_checkpointing кэш ключей/значений несовместим с обратным
        # проходом по чек-поинтам; отключаем явно.
        self.roberta.config.use_cache = False

        hidden_size = self.config.hidden_size
        self.head_norm = nn.LayerNorm(hidden_size)
        self.head_dropout = nn.Dropout(model_cfg.head_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # class_weights регистрируем как buffer: они перемещаются вместе с моделью
        # на device, но не учитываются в state_dict при сохранении весов
        # (отфильтровываются в utils.get_filtered_model_state_dict).
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def freeze_lower_layers(self, freeze_encoder_layers: int = 0, freeze_embeddings: bool = False) -> dict:
        """Замораживает нижние слои энкодера и (опционально) слой эмбеддингов.

        Возвращает статистику по числу параметров для логирования.
        """
        total_layers = len(self.roberta.encoder.layer)
        n_freeze = max(0, min(int(freeze_encoder_layers), total_layers))

        if freeze_embeddings:
            for p in self.roberta.embeddings.parameters():
                p.requires_grad = False

        for i in range(n_freeze):
            for p in self.roberta.encoder.layer[i].parameters():
                p.requires_grad = False

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "frozen_encoder_layers": n_freeze,
            "frozen_embeddings": bool(freeze_embeddings),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_ratio": round(trainable_params / max(1, total_params), 4),
        }

    @staticmethod
    def token_mean_pool(last_hidden_state, attention_mask):
        """Усреднение токенных представлений с учётом маски паддинга."""
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def chunk_mean_pool(chunk_embeddings, chunk_mask):
        """Усреднение эмбеддингов чанков документа с учётом числа валидных чанков."""
        mask = chunk_mask.unsqueeze(-1).type_as(chunk_embeddings)
        masked = chunk_embeddings * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None, num_chunks=None, **kwargs):
        # input_ids имеет форму (batch, n_chunks, seq_len) — разворачиваем в (batch*n_chunks, seq_len)
        # чтобы прогнать все чанки через энкодер одним вызовом.
        batch_size, n_chunks, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * n_chunks, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * n_chunks, seq_len)

        outputs = self.roberta(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            return_dict=True,
        )

        # Эмбеддинг каждого чанка как mean-pool токенов.
        token_pooled = self.token_mean_pool(outputs.last_hidden_state, flat_attention_mask)
        chunk_embeddings = token_pooled.view(batch_size, n_chunks, -1)

        # Маска валидных чанков: либо явно по num_chunks, либо по тому, есть ли
        # ненулевые токены в attention_mask.
        if num_chunks is None:
            chunk_mask = (attention_mask.sum(dim=-1) > 0).long()
        else:
            arange = torch.arange(n_chunks, device=input_ids.device).unsqueeze(0)
            chunk_mask = (arange < num_chunks.unsqueeze(1)).long()

        doc_embedding = self.chunk_mean_pool(chunk_embeddings, chunk_mask)

        # Классификационная голова.
        x = self.head_norm(doc_embedding)
        x = self.head_dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.model_cfg.label_smoothing,
            )
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    # Прокси-методы для HF Trainer — он вызывает их через флаг
    # gradient_checkpointing в TrainingArguments.
    def gradient_checkpointing_enable(self, **kwargs):
        self.roberta.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.roberta.gradient_checkpointing_disable()


def build_model(
    model_cfg: ModelConfig,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    class_weights_tensor: Optional[torch.Tensor] = None,
) -> ChunkMeanPoolRobertaClassifier:
    """Удобный конструктор: собирает классификатор по конфигурации."""
    return ChunkMeanPoolRobertaClassifier(
        model_cfg=model_cfg,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        class_weights=class_weights_tensor,
    )
