import re
import torch
from razdel import sentenize
from tqdm.auto import tqdm

from backtranslate.models import get_model_by_mode
from common.masks import mask_placeholders, unmask_placeholders


def mt_tokens_len(text: str, mode: str, device) -> int:
    tokenizer, _ = get_model_by_mode(mode, device)
    inputs = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    return inputs.input_ids.shape[1]


def preprocess_before_translate(text: str) -> str:
    text = re.sub(r"[-=_*]{5,}", "—", text)
    text = re.sub(r"[.\s]{5,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def generate_translate(text: str, mode: str, device, **gen_kwargs) -> str:
    tokenizer, model = get_model_by_mode(mode, device)
    inputs = tokenizer(text, return_tensors="pt", truncation=False, padding=True).to(device)

    input_len = inputs.input_ids.shape[1]
    if input_len > 300:
        raise ValueError(f"Input too long: {input_len} tokens.")

    defaults = dict(num_beams=1, do_sample=True, top_k=50, top_p=0.92, temperature=1.1)
    defaults.update(gen_kwargs)

    with torch.no_grad():
        generated = model.generate(**inputs, **defaults)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def split_long_sentence(sent: str, max_tokens: int, mode: str, device) -> list[str]:
    if mt_tokens_len(sent, mode, device) <= max_tokens:
        return [sent]

    parts = re.split(r"(?<=[,;:—\-])\s+", sent)

    chunks = []

    if len(parts) == 1:
        words = sent.split()
        current = ""
        for w in words:
            candidate = f"{current} {w}".strip() if current else w
            if mt_tokens_len(candidate, mode, device) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)
        return chunks or [sent]

    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if mt_tokens_len(candidate, mode, device) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.extend(split_long_sentence(current, max_tokens, mode, device))
            if mt_tokens_len(part, mode, device) > max_tokens:
                chunks.extend(split_long_sentence(part, max_tokens, mode, device))
                current = ""
            else:
                current = part
    if current:
        chunks.extend(split_long_sentence(current, max_tokens, mode, device))

    return chunks or [sent]


def safe_translate(text: str, mode: str, max_tokens: int, device, **gen_kwargs) -> str:
    chunks = split_long_sentence(text, max_tokens=max_tokens, mode=mode, device=device)
    out_chunks = []
    for ch in chunks:
        if mt_tokens_len(ch, mode, device) > 300:
            words = ch.split()
            for i in range(0, len(words), 20):
                sub = " ".join(words[i : i + 20])
                if sub.strip():
                    out_chunks.append(generate_translate(sub, mode, device, **gen_kwargs))
        else:
            out_chunks.append(generate_translate(ch, mode, device, **gen_kwargs))
    return " ".join(out_chunks)


def clean_bt_result(text: str) -> str:
    text = re.sub(r"([,.!?])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    return text.strip()