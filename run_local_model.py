import torch
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "./models/rubert-base-cased"
INPUT_PATH = "data.txt"

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # local_files_only=True гарантирует, что в сеть не полезет
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    model.eval()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)
        emb = mean_pooling(out.last_hidden_state, batch["attention_mask"])

    print("Texts:", len(texts))
    print("Embeddings:", tuple(emb.shape))  # (N, hidden_size)

if __name__ == "__main__":
    main()