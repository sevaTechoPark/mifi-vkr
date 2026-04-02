import re
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

MIN_LIST_LINES = 3
RE_LIST_ITEM = re.compile(
    r'^\s*(?:'
    r'(?:\d{1,4}\s*[.)])'
    r'|(?:\d{1,4}\s*[-–—])'
    r'|(?:[-•*]\s+)'
    r'|(?:no\.?\s*\d{1,4}\b)'
    r'|(?:№\s*\d{1,4}\b)'
    r')',
    flags=re.IGNORECASE
)

MAX_PASSES = 6
TOKEN_RUN_MIN = 3
NGRAM_MIN_N = 2
NGRAM_MAX_N = 5
NGRAM_RUN_MIN = 2
MIN_CHAIN_TOKENS = 8

RE_POINTS_HEADER = re.compile(r'(?is)\bа\s+именно\s+следующ(?:их|ие)\s+точ(?:ек|ки)\s*:')
RE_NUM_ITEM_INLINE = re.compile(r'(?s)(?:^|\s)(?:\d{1,3}\s*[.)])\s+')
RE_POINTS_END = re.compile(
    r'(?is)(?:'
    r'\bначиная\s+с\s+момента\s+времени\b'
    r'|'
    r'\bкроме\s+того\b'
    r'|'
    r'\bтакже\s+сообща(?:ю|ем)\b'
    r'|'
    r'\bприложени[ея]\s*:'
    r')'
)
RE_UPD_BLOCK = re.compile(
    r'(?is)\bупд\s*(?:no|№)\s*$$DOCUMENT_NUMBER$$(?:/\d+)?\s*от\s*$$DATE(?:_TIME)?$$\b'
)
RE_FIN_DATE = re.compile(
    r'(?is)$$(?:FINANCIAL_DATA|FINANCИAL_DATA)$$\s*$$DATE(?:_TIME)?$$'
)

def _norm_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


# ============================================================
# 0) Сжать "точки поставки" в маркер [POINTS_LIST]
# ============================================================
def compress_points_list_block(text: str) -> str:
    if text is None:
        return ""

    s = str(text)
    m = RE_POINTS_HEADER.search(s)
    if not m:
        return s

    start = m.end()
    tail = s[start:]

    if not RE_NUM_ITEM_INLINE.search(tail):
        return s

    m_end = RE_POINTS_END.search(tail)
    after = tail[m_end.start():] if m_end else ""

    prefix = s[:start]
    return prefix + " [POINTS_LIST] " + after


# ============================================================
# 1) Удаление списков (блоками)
# ============================================================
def remove_lists(text: str) -> str:
    if text is None:
        return ""

    lines = str(text).splitlines()
    out = []
    i = 0

    while i < len(lines):
        if RE_LIST_ITEM.match(lines[i] or ""):
            block = []
            while i < len(lines) and RE_LIST_ITEM.match(lines[i] or ""):
                block.append(lines[i])
                i += 1

            if len(block) < MIN_LIST_LINES:
                out.extend(block)
        else:
            out.append(lines[i])
            i += 1

    return "\n".join(out)


# ============================================================
# 2) Дублирующиеся строки (в пределах письма)
# ============================================================
def remove_duplicate_lines(text: str) -> str:
    if text is None:
        return ""

    lines = str(text).splitlines()
    seen = set()
    out_lines = []

    for line in lines:
        key = _norm_space(line)
        if key == "":
            out_lines.append(line)
            continue
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(line)

    return "\n".join(out_lines)


# ============================================================
# 3) Схлопывание подряд одинаковых токенов: X X X -> X
# ============================================================
def collapse_token_runs(text: str) -> str:
    if text is None:
        return ""

    tokens = str(text).split()
    if len(tokens) < TOKEN_RUN_MIN:
        return str(text)

    out = []
    i = 0
    L = len(tokens)

    while i < L:
        j = i + 1
        while j < L and tokens[j] == tokens[i]:
            j += 1
        run_len = j - i

        if run_len >= TOKEN_RUN_MIN:
            out.append(tokens[i])
        else:
            out.extend(tokens[i:j])

        i = j

    return " ".join(out)


# ============================================================
# 4) Схлопывание подряд одинаковых n-грамм (фраз)
# ============================================================
def collapse_ngram_runs(text: str) -> str:
    if text is None:
        return ""

    tokens = str(text).split()
    if len(tokens) < 2 * NGRAM_MIN_N:
        return str(text)

    def one_pass(tok):
        out = []
        i = 0
        changed = False
        L = len(tok)

        while i < L:
            best_n = 0
            best_r = 0

            for n in range(NGRAM_MAX_N, NGRAM_MIN_N - 1, -1):
                if i + 2 * n > L:
                    continue
                gram = tok[i:i+n]

                r = 1
                while i + (r + 1) * n <= L and tok[i + r*n : i + (r+1)*n] == gram:
                    r += 1

                if r >= NGRAM_RUN_MIN:
                    best_n, best_r = n, r
                    break

            if best_n > 0:
                out.extend(tok[i:i+best_n])
                i += best_r * best_n
                changed = True
            else:
                out.append(tok[i])
                i += 1

        return out, changed

    for _ in range(MAX_PASSES):
        tokens, changed = one_pass(tokens)
        if not changed:
            break

    return " ".join(tokens)


# ============================================================
# 5) Схлопывание "ABCDABCD" для длинных цепочек
# ============================================================
def remove_repeated_token_chains(text: str) -> str:
    if text is None:
        return ""

    tokens = str(text).split()
    if len(tokens) < 2 * MIN_CHAIN_TOKENS:
        return str(text)

    def one_pass(tok):
        out = []
        i = 0
        changed = False
        L = len(tok)

        while i < L:
            max_k = (L - i) // 2
            found_k = 0

            k = max_k
            while k >= MIN_CHAIN_TOKENS:
                if tok[i:i+k] == tok[i+k:i+2*k]:
                    found_k = k
                    break
                k -= 1

            if found_k:
                out.extend(tok[i:i+found_k])
                i += 2 * found_k
                changed = True
            else:
                out.append(tok[i])
                i += 1

        return out, changed

    for _ in range(MAX_PASSES):
        tokens, changed = one_pass(tokens)
        if not changed:
            break

    return " ".join(tokens)

def remove_requisites_phrases(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = RE_UPD_BLOCK.sub(" ", s)
    s = RE_FIN_DATE.sub(" ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_text(text: str) -> str:
    t = compress_points_list_block(text)
    t = remove_requisites_phrases(t)
    t = remove_lists(t)
    t = remove_duplicate_lines(t)
    t = collapse_token_runs(t)
    t = collapse_ngram_runs(t)
    t = remove_repeated_token_chains(t)
    return t


def make_df_clean(dt: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df_clean = dt[[label_col, text_col]].copy()
    df_clean = df_clean.rename(columns={text_col: "prev_text"})
    df_clean["next_text"] = df_clean["prev_text"].apply(clean_text)
    df_clean.to_csv(data_dir / "procesed_data.csv", index=False)
    return df_clean
