import re
import random

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str):
    text = str(text)
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents

def first_n_sentences(text: str, n: int) -> str:
    s = split_sentences(text)
    return " ".join(s[:n]) if len(s) >= 1 else str(text)

def random_window_sentences(text: str, n: int, rng: random.Random) -> str:
    s = split_sentences(text)
    if len(s) <= n:
        return " ".join(s)
    start = rng.randint(0, len(s) - n)
    return " ".join(s[start:start+n])

def top_n_sentences_by_tfidf(text: str, n: int, vectorizer) -> str:
    # vectorizer: обученный TfidfVectorizer на предложениях
    sents = split_sentences(text)
    if len(sents) <= n:
        return " ".join(sents)
    X = vectorizer.transform(sents)
    scores = X.sum(axis=1).A1  # сумма TF-IDF весов по словам
    top_idx = scores.argsort()[::-1][:n]
    top_idx = sorted(top_idx)  # сохраняем порядок в тексте
    return " ".join([sents[i] for i in top_idx])

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def random_delete_words(text: str, drop_frac: float, rng: random.Random) -> str:
    tokens = _WORD_RE.findall(str(text))
    if len(tokens) < 5:
        return str(text)

    protected = {"не", "ни"}
    # плейсхолдеры вида [NAME], [DOC_NUMBER]
    def is_placeholder(tok: str) -> bool:
        return bool(re.fullmatch(r"$$[A-Z_]+$$", tok))

    keep = []
    for tok in tokens:
        if tok.lower() in protected or is_placeholder(tok):
            keep.append(tok)
        else:
            if rng.random() > drop_frac:
                keep.append(tok)

    if not keep:
        keep = tokens[:]

    return " ".join(keep)
