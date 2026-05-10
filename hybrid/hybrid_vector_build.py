import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoTokenizer, RobertaModel, RobertaConfig

MODEL_NAME = "ai-forever/ruRoberta-large"
TEXT_COL = "text"
LABEL_COL = "label"

MAX_LENGTH = 512
STRIDE = 256
MAX_CHUNKS = 6

BERT_BATCH_SIZE = 8
BERT_WEIGHT = 5.0


class ChunkMeanPoolRobertaClassifier(nn.Module):
    def __init__(self, model_name, num_labels, id2label=None, label2id=None, head_dropout=0.3):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            output_hidden_states=False,
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout1 = nn.Dropout(head_dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

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

    def encode_documents(self, input_ids=None, attention_mask=None, num_chunks=None):
        # input_ids: [batch, chunks, seq]
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
        return doc_embedding

    def forward(self, input_ids=None, attention_mask=None, labels=None, num_chunks=None, **kwargs):
        doc_embedding = self.encode_documents(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_chunks=num_chunks,
        )

        x = self.dropout1(doc_embedding)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout2(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits, "doc_embedding": doc_embedding}


def build_tfidf(X_train, X_test):
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        sublinear_tf=True,
    )

    X_train_word = word_tfidf.fit_transform(X_train)
    X_test_word = word_tfidf.transform(X_test)

    X_train_char = char_tfidf.fit_transform(X_train)
    X_test_char = char_tfidf.transform(X_test)

    X_train_tfidf = sp.hstack([X_train_word, X_train_char]).tocsr()
    X_test_tfidf = sp.hstack([X_test_word, X_test_char]).tocsr()

    X_train_tfidf = normalize(X_train_tfidf, norm="l2")
    X_test_tfidf = normalize(X_test_tfidf, norm="l2")

    return X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf


def tokenize_document_for_chunks(text, tokenizer, max_length, stride, max_chunks):
    encoded = tokenizer(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
    )

    input_ids_chunks = encoded["input_ids"][:max_chunks]
    attention_mask_chunks = encoded["attention_mask"][:max_chunks]

    n_chunks = len(input_ids_chunks)

    if n_chunks < max_chunks:
        pad_len = max_chunks - n_chunks
        pad_ids = [tokenizer.pad_token_id] * max_length
        pad_mask = [0] * max_length
        input_ids_chunks += [pad_ids] * pad_len
        attention_mask_chunks += [pad_mask] * pad_len

    return input_ids_chunks, attention_mask_chunks, n_chunks


@torch.no_grad()
def document_bert_embeddings_base(texts, tokenizer, model, device, batch_size=BERT_BATCH_SIZE):
    all_vecs = []
    texts = [str(t) for t in texts]
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded, return_dict=True)

        mask = encoded["attention_mask"].unsqueeze(-1).type_as(outputs.last_hidden_state)
        masked = outputs.last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_vecs)


@torch.no_grad()
def document_bert_embeddings_finetuned(
    texts,
    tokenizer,
    model,
    device,
    batch_size=BERT_BATCH_SIZE,
    max_length=MAX_LENGTH,
    stride=STRIDE,
    max_chunks=MAX_CHUNKS,
):
    all_vecs = []
    texts = [str(t) for t in texts]
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        batch_input_ids = []
        batch_attention_mask = []
        batch_num_chunks = []

        for text in batch_texts:
            input_ids_chunks, attention_mask_chunks, n_chunks = tokenize_document_for_chunks(
                text=text,
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
                max_chunks=max_chunks,
            )
            batch_input_ids.append(input_ids_chunks)
            batch_attention_mask.append(attention_mask_chunks)
            batch_num_chunks.append(n_chunks)

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)
        num_chunks = torch.tensor(batch_num_chunks, dtype=torch.long, device=device)

        doc_embedding = model.encode_documents(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_chunks=num_chunks,
        )
        doc_embedding = torch.nn.functional.normalize(doc_embedding, p=2, dim=1)
        all_vecs.append(doc_embedding.cpu().numpy().astype(np.float32))

    return np.vstack(all_vecs)


def load_finetuned_model(finetuned_dir, device):
    meta_path = os.path.join(finetuned_dir, "training_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    num_labels = meta["num_labels"]
    head_dropout = meta.get("head_dropout", 0.3)

    model = ChunkMeanPoolRobertaClassifier(
        model_name=MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        head_dropout=head_dropout,
    )

    state_dict = torch.load(os.path.join(finetuned_dir, "pytorch_model.bin"), map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device)

    print("Loaded fine-tuned weights from:", os.path.join(finetuned_dir, "pytorch_model.bin"))
    print("MISSING KEYS:", missing)
    print("UNEXPECTED KEYS:", unexpected)

    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bert_weight", type=float, default=BERT_WEIGHT)
    parser.add_argument(
        "--finetuned_dir",
        required=False,
        default=None,
        help="Путь к директории с fine-tuned ChunkMeanPoolRobertaClassifier "
             "(pytorch_model.bin + training_meta.json)."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_df = train_df[[TEXT_COL, LABEL_COL]].copy().dropna()
    test_df = test_df[[TEXT_COL, LABEL_COL]].copy().dropna()

    train_df[TEXT_COL] = train_df[TEXT_COL].astype(str).str.strip()
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(str).str.strip()
    test_df[TEXT_COL] = test_df[TEXT_COL].astype(str).str.strip()
    test_df[LABEL_COL] = test_df[LABEL_COL].astype(str).str.strip()

    train_df = train_df[(train_df[TEXT_COL] != "") & (train_df[LABEL_COL] != "")].reset_index(drop=True)
    test_df = test_df[(test_df[TEXT_COL] != "") & (test_df[LABEL_COL] != "")].reset_index(drop=True)

    X_train = train_df[TEXT_COL]
    y_train = train_df[LABEL_COL]
    X_test = test_df[TEXT_COL]
    y_test = test_df[LABEL_COL]

    X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf = build_tfidf(X_train, X_test)

    device = torch.device(args.device)

    if args.finetuned_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.finetuned_dir)
        model, ft_meta = load_finetuned_model(args.finetuned_dir, device)

        max_length = ft_meta.get("max_length", MAX_LENGTH)
        stride = ft_meta.get("stride", STRIDE)
        max_chunks = ft_meta.get("max_chunks", MAX_CHUNKS)

        X_train_bert = document_bert_embeddings_finetuned(
            X_train.tolist(),
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=BERT_BATCH_SIZE,
            max_length=max_length,
            stride=stride,
            max_chunks=max_chunks,
        )
        X_test_bert = document_bert_embeddings_finetuned(
            X_test.tolist(),
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=BERT_BATCH_SIZE,
            max_length=max_length,
            stride=stride,
            max_chunks=max_chunks,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = RobertaModel.from_pretrained(MODEL_NAME).to(device)

        X_train_bert = document_bert_embeddings_base(X_train.tolist(), tokenizer, model, device)
        X_test_bert = document_bert_embeddings_base(X_test.tolist(), tokenizer, model, device)

    scaler_bert = StandardScaler()
    X_train_bert = scaler_bert.fit_transform(X_train_bert).astype(np.float32)
    X_test_bert = scaler_bert.transform(X_test_bert).astype(np.float32)

    X_train_hybrid = sp.hstack([
        X_train_tfidf,
        sp.csr_matrix(X_train_bert * args.bert_weight)
    ]).tocsr()

    X_test_hybrid = sp.hstack([
        X_test_tfidf,
        sp.csr_matrix(X_test_bert * args.bert_weight)
    ]).tocsr()

    sp.save_npz(os.path.join(args.outdir, "X_train_hybrid.npz"), X_train_hybrid)
    sp.save_npz(os.path.join(args.outdir, "X_test_hybrid.npz"), X_test_hybrid)
    pd.Series(y_train).to_csv(os.path.join(args.outdir, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(args.outdir, "y_test.csv"), index=False)

    joblib.dump(word_tfidf, os.path.join(args.outdir, "word_tfidf.joblib"))
    joblib.dump(char_tfidf, os.path.join(args.outdir, "char_tfidf.joblib"))
    joblib.dump(scaler_bert, os.path.join(args.outdir, "scaler_bert.joblib"))
    tokenizer.save_pretrained(os.path.join(args.outdir, "bert_tokenizer"))

    meta = {
        "model_name": MODEL_NAME,
        "finetuned_dir": args.finetuned_dir,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "max_chunks": MAX_CHUNKS,
        "bert_batch_size": BERT_BATCH_SIZE,
        "bert_weight": args.bert_weight,
        "train_shape": X_train_hybrid.shape,
        "test_shape": X_test_hybrid.shape,
    }
    joblib.dump(meta, os.path.join(args.outdir, "meta.joblib"))
    print(meta)


if __name__ == "__main__":
    main()