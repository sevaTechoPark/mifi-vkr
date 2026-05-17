from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

from bert_embeddings.config import EmbeddingConfig


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def masked_max_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).bool()
    masked = last_hidden_state.masked_fill(~mask, float("-inf"))
    pooled = masked.max(dim=1).values
    pooled[torch.isinf(pooled)] = 0.0
    return pooled


class LongTextRobertaEmbedder:
    def __init__(
        self,
        model_dir,
        cfg: EmbeddingConfig | None = None,
        base_model_name: str | None = None,
        max_length: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        pooling: str | None = None,
        chunk_aggregation: str | None = None,
        batch_size: int | None = None,
        normalize_chunks: bool | None = None,
        normalize_document: bool | None = None,
        add_global_chunk: bool | None = None,
        device: str | None = None,
    ):
        if cfg is None:
            cfg = EmbeddingConfig()

        self.base_model_name = base_model_name or cfg.base_model_name
        self.max_length = max_length or cfg.max_length
        self.pooling = pooling or cfg.pooling
        self.chunk_aggregation = chunk_aggregation or cfg.chunk_aggregation
        self.batch_size = batch_size or cfg.batch_size
        self.normalize_chunks = (
            normalize_chunks if normalize_chunks is not None else cfg.normalize_chunks
        )
        self.normalize_document = (
            normalize_document if normalize_document is not None else cfg.normalize_document
        )
        self.add_global_chunk = (
            add_global_chunk if add_global_chunk is not None else cfg.add_global_chunk
        )

        raw_chunk_size = chunk_size or cfg.chunk_size
        raw_chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.chunk_size = min(raw_chunk_size, self.max_length - 2)
        self.chunk_overlap = min(raw_chunk_overlap, max(0, self.chunk_size // 2))

        model_dir_str = str(model_dir).strip() if model_dir is not None else ""
        self.model_dir = Path(model_dir_str).expanduser() if model_dir_str else None

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        tokenizer_source = str(self.model_dir) if self.model_dir and self.model_dir.exists() else self.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        if self.model_dir is not None and (self.model_dir / "config.json").exists():
            roberta_cfg = RobertaConfig.from_pretrained(str(self.model_dir))
        else:
            roberta_cfg = RobertaConfig.from_pretrained(self.base_model_name)

        self.model = RobertaModel(roberta_cfg)

        if self.model_dir is not None:
            weights_path = self.model_dir / "pytorch_model.bin"
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")

            state_dict = torch.load(weights_path, map_location="cpu")
            filtered = {
                k.replace("roberta.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("roberta.")
            }
            missing, unexpected = self.model.load_state_dict(filtered, strict=False)
            print(f"Loaded encoder weights from: {weights_path}")
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
        else:
            base_model = RobertaModel.from_pretrained(self.base_model_name, config=roberta_cfg)
            self.model.load_state_dict(base_model.state_dict(), strict=True)
            print(f"Using base model without local fine-tuned weights: {self.base_model_name}")

        self.model.to(self.device)
        self.model.eval()

    def _tokenize_full(self, text):
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]

    def _chunk_token_ids(self, text):
        token_ids = self._tokenize_full(text)
        if not token_ids:
            return []

        stride = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for start in range(0, len(token_ids), stride):
            end = start + self.chunk_size
            piece = token_ids[start:end]
            if not piece:
                continue
            piece = (
                [self.tokenizer.cls_token_id]
                + piece[: self.max_length - 2]
                + [self.tokenizer.sep_token_id]
            )
            chunks.append(piece)
            if end >= len(token_ids):
                break

        if self.add_global_chunk and len(token_ids) > self.chunk_size:
            head = token_ids[: self.chunk_size // 2]
            tail = token_ids[-(self.chunk_size - len(head)) :]
            global_piece = (
                [self.tokenizer.cls_token_id]
                + (head + tail)[: self.max_length - 2]
                + [self.tokenizer.sep_token_id]
            )
            chunks.append(global_piece)

        return chunks

    @torch.no_grad()
    def _encode_chunk_batch(self, batch_token_ids):
        max_len = max(len(x) for x in batch_token_ids)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask = [], []

        for ids in batch_token_ids:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "max":
            emb = masked_max_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == "mean_max":
            mean_emb = mean_pooling(outputs.last_hidden_state, attention_mask)
            max_emb = masked_max_pooling(outputs.last_hidden_state, attention_mask)
            emb = 0.5 * (mean_emb + max_emb)
        else:
            emb = mean_pooling(outputs.last_hidden_state, attention_mask)

        if self.normalize_chunks:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def _aggregate_chunks(self, chunk_embs):
        if self.chunk_aggregation == "max":
            doc_emb = np.max(chunk_embs, axis=0)
        elif self.chunk_aggregation == "mean_max":
            doc_emb = 0.5 * (np.mean(chunk_embs, axis=0) + np.max(chunk_embs, axis=0))
        else:
            doc_emb = np.mean(chunk_embs, axis=0)

        if self.normalize_document:
            doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-12)
        return doc_emb.astype(np.float32)

    def encode(self, texts, return_chunk_counts=False):
        doc_embeddings = []
        chunk_counts = []
        hidden = self.model.config.hidden_size

        for text in texts:
            chunk_ids = self._chunk_token_ids(text)
            chunk_counts.append(len(chunk_ids))
            if not chunk_ids:
                doc_embeddings.append(np.zeros(hidden, dtype=np.float32))
                continue

            chunk_embs = []
            for i in range(0, len(chunk_ids), self.batch_size):
                batch = chunk_ids[i:i + self.batch_size]
                chunk_embs.append(self._encode_chunk_batch(batch))
            chunk_embs = np.vstack(chunk_embs)
            doc_embeddings.append(self._aggregate_chunks(chunk_embs))

        embs = np.vstack(doc_embeddings)
        if return_chunk_counts:
            return embs, chunk_counts
        return embs