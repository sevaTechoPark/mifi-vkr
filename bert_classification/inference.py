import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from .config import ModelConfig, DataConfig
from .model import build_model
from .utils import get_device_map_location


def load_export_bundle(output_dir: str):
    export_path = Path(output_dir)

    with open(export_path / "training_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_cfg = ModelConfig(**meta["model_config"])
    data_cfg = DataConfig(**meta["data_config"])

    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(export_path)

    model = build_model(
        model_cfg=model_cfg,
        num_labels=meta["num_labels"],
        label2id=label2id,
        id2label=id2label,
        class_weights_tensor=None,
    )

    state_dict = torch.load(
        export_path / "pytorch_model.bin",
        map_location=get_device_map_location(),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, tokenizer, model_cfg, data_cfg, label2id, id2label


def build_inference_batch(texts: List[str], tokenizer, model_cfg: ModelConfig):
    input_ids_batch = []
    attention_mask_batch = []
    num_chunks_batch = []

    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=model_cfg.max_length,
            stride=model_cfg.stride,
            return_overflowing_tokens=True,
        )

        input_ids_chunks = encoded["input_ids"][:model_cfg.max_chunks]
        attention_mask_chunks = encoded["attention_mask"][:model_cfg.max_chunks]
        n_chunks = len(input_ids_chunks)

        if n_chunks < model_cfg.max_chunks:
            pad_len = model_cfg.max_chunks - n_chunks
            pad_ids = [tokenizer.pad_token_id] * model_cfg.max_length
            pad_mask = [0] * model_cfg.max_length
            input_ids_chunks += [pad_ids] * pad_len
            attention_mask_chunks += [pad_mask] * pad_len

        input_ids_batch.append(input_ids_chunks)
        attention_mask_batch.append(attention_mask_chunks)
        num_chunks_batch.append(n_chunks)

    batch = {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "num_chunks": torch.tensor(num_chunks_batch, dtype=torch.long),
    }
    return batch


@torch.no_grad()
def predict_texts(output_dir: str, texts: List[str], top_k: int = 3):
    model, tokenizer, model_cfg, data_cfg, label2id, id2label = load_export_bundle(output_dir)
    device = torch.device(get_device_map_location())
    model.to(device)

    batch = build_inference_batch(texts, tokenizer, model_cfg)
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)
    logits = outputs["logits"]
    probs = torch.softmax(logits, dim=-1).cpu()
    pred_ids = probs.argmax(dim=-1).tolist()

    results = []
    for i, pred_id in enumerate(pred_ids):
        prob_vec = probs[i]
        top_probs, top_ids = torch.topk(prob_vec, k=min(top_k, prob_vec.shape[0]))

        results.append(
            {
                "text": texts[i],
                "pred_label": id2label[int(pred_id)],
                "pred_id": int(pred_id),
                "probs": prob_vec.tolist(),
                "top_k": [
                    {
                        "label": id2label[int(cls_id)],
                        "id": int(cls_id),
                        "prob": float(cls_prob),
                    }
                    for cls_prob, cls_id in zip(top_probs.tolist(), top_ids.tolist())
                ],
            }
        )

    return results