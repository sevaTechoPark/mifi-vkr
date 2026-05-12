from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

from .config import ModelConfig


class ChunkDataCollator:
    def __init__(self, max_chunks: int, max_length: int):
        self.max_chunks = max_chunks
        self.max_length = max_length

    def __call__(self, features):
        for i, f in enumerate(features):
            assert len(f["input_ids"]) == self.max_chunks, f"Bad num_chunks in sample {i}"
            assert all(len(x) == self.max_length for x in f["input_ids"]), f"Bad input_ids len in sample {i}"
            assert all(len(x) == self.max_length for x in f["attention_mask"]), f"Bad attention_mask len in sample {i}"

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
        self.roberta.gradient_checkpointing_enable()
        self.roberta.config.use_cache = False

        hidden_size = self.config.hidden_size

        self.dropout1 = nn.Dropout(model_cfg.head_dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(model_cfg.head_dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def token_mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def chunk_mean_pool(self, chunk_embeddings, chunk_mask):
        mask = chunk_mask.unsqueeze(-1).type_as(chunk_embeddings)
        masked = chunk_embeddings * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None, num_chunks=None, **kwargs):
        batch_size, n_chunks, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * n_chunks, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * n_chunks, seq_len)

        outputs = self.roberta(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            return_dict=True,
        )

        token_pooled = self.token_mean_pool(outputs.last_hidden_state, flat_attention_mask)
        chunk_embeddings = token_pooled.view(batch_size, n_chunks, -1)

        if num_chunks is None:
            chunk_mask = (attention_mask.sum(dim=-1) > 0).long()
        else:
            arange = torch.arange(n_chunks, device=input_ids.device).unsqueeze(0)
            chunk_mask = (arange < num_chunks.unsqueeze(1)).long()

        doc_embedding = self.chunk_mean_pool(chunk_embeddings, chunk_mask)

        x = self.dropout1(doc_embedding)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout2(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.model_cfg.label_smoothing,
            )
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def build_model(
    model_cfg: ModelConfig,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    class_weights_tensor: Optional[torch.Tensor] = None,
) -> ChunkMeanPoolRobertaClassifier:
    return ChunkMeanPoolRobertaClassifier(
        model_cfg=model_cfg,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        class_weights=class_weights_tensor,
    )