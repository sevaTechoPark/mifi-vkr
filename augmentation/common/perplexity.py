import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import RUGPT_MODEL_NAME

def load_rugpt(device: torch.device):
    tok = AutoTokenizer.from_pretrained(RUGPT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(RUGPT_MODEL_NAME).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def rugpt_perplexity_list(
    texts: list[str],
    tok,
    model,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 8,
) -> list[float]:
    if not texts:
        return []

    ppl_values = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        enc = tok(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        for j in range(len(batch_texts)):
            tokens = input_ids[j]
            mask = attention_mask[j]
            seq_len = mask.sum().item()

            if seq_len == 0:
                ppl_values.append(float("inf"))
                continue

            shift_logits = logits[j, :-1, :]
            shift_labels = tokens[1:]
            shift_mask = mask[1:]

            loss_per_token = loss_fct(shift_logits, shift_labels)
            masked_loss = loss_per_token * shift_mask
            avg_loss = masked_loss.sum() / shift_mask.sum()
            ppl_values.append(torch.exp(avg_loss).item())

    return ppl_values