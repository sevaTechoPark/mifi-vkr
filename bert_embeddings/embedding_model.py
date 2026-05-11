import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, RobertaConfig, RobertaModel


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
        base_model_name="ai-forever/ruRoberta-large",
        max_length=512,
        chunk_size=448,
        chunk_overlap=96,
        pooling="mean_max",
        chunk_aggregation="mean_max",
        batch_size=8,
        normalize_chunks=True,
        normalize_document=True,
        add_global_chunk=True,
        device=None,
    ):
        self.model_dir = Path(model_dir)
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.chunk_size = min(chunk_size, max_length - 2)
        self.chunk_overlap = min(chunk_overlap, max(0, self.chunk_size // 2))
        self.pooling = pooling
        self.chunk_aggregation = chunk_aggregation
        self.batch_size = batch_size
        self.normalize_chunks = normalize_chunks
        self.normalize_document = normalize_document
        self.add_global_chunk = add_global_chunk

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        cfg = RobertaConfig.from_pretrained(base_model_name)
        self.model = RobertaModel.from_pretrained(base_model_name, config=cfg)

        weights_path = self.model_dir / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            filtered = {k.replace("roberta.", "", 1): v for k, v in state_dict.items() if k.startswith("roberta.")}
            missing, unexpected = self.model.load_state_dict(filtered, strict=False)
            print("Loaded encoder weights.")
            print("Missing keys:", len(missing))
            print("Unexpected keys:", len(unexpected))
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        self.model.to(self.device)
        self.model.eval()

    def _tokenize_full(self, text):
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
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
            piece = [self.tokenizer.cls_token_id] + piece[: self.max_length - 2] + [self.tokenizer.sep_token_id]
            chunks.append(piece)
            if end >= len(token_ids):
                break

        if self.add_global_chunk and len(token_ids) > self.chunk_size:
            head = token_ids[: self.chunk_size // 2]
            tail = token_ids[-(self.chunk_size - len(head)):]
            global_piece = [self.tokenizer.cls_token_id] + (head + tail)[: self.max_length - 2] + [self.tokenizer.sep_token_id]
            chunks.append(global_piece)

        return chunks

    @torch.no_grad()
    def _encode_chunk_batch(self, batch_token_ids):
        max_len = max(len(x) for x in batch_token_ids)
        input_ids = []
        attention_mask = []
        pad_id = self.tokenizer.pad_token_id

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
                vec = np.zeros(hidden, dtype=np.float32)
                doc_embeddings.append(vec)
                continue

            chunk_embs = []
            for i in range(0, len(chunk_ids), self.batch_size):
                batch = chunk_ids[i:i + self.batch_size]
                chunk_embs.append(self._encode_chunk_batch(batch))
            chunk_embs = np.vstack(chunk_embs)
            doc_emb = self._aggregate_chunks(chunk_embs)
            doc_embeddings.append(doc_emb)

        embs = np.vstack(doc_embeddings)
        if return_chunk_counts:
            return embs, chunk_counts
        return embs