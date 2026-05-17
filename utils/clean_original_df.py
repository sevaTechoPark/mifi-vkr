import logging
import re
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# ── constants ────────────────────────────────────────────────────────────────

MIN_LIST_LINES = 3
RE_LIST_ITEM = re.compile(
    r'^\s*(?:'
    r'(?:\d{1,4}\s*[.)])'
    r'|(?:\d{1,4}\s*[-–—])'
    r'|(?:[-•*]\s+)'
    r'|(?:no\.?\s*\d{1,4}\b)'
    r'|(?:№\s*\d{1,4}\b)'
    r')',
    flags=re.IGNORECASE,
)

MAX_PASSES      = 6
TOKEN_RUN_MIN   = 3
NGRAM_MIN_N     = 2
NGRAM_MAX_N     = 5
NGRAM_RUN_MIN   = 2
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
    r'(?is)\bупд\s*(?:no|№)\s*\[DOCUMENT_NUMBER\](?:/\d+)?\s*от\s*\[DATE(?:_TIME)?\]\b'
)
RE_FIN_DATE = re.compile(
    r'(?is)\[(?:FINANCIAL_DATA|FINANCИAL_DATA)\]\s*\[DATE(?:_TIME)?\]'
)
RE_ANY_ANON_TOKEN = re.compile(r'\[[A-ZА-ЯЁ_]+\]')

RE_ANON_MARKER = re.compile(
    r'\[(?:FINANCIAL_DATA|FINANCИAL_DATA|ORGANIZATION|PERSON|LOCATION|CONTACT|ID|OBJECT|DOCUMENT_NUMBER|DATE|DATE_TIME|NUMBER|DESCRIPTION|POSITION|TYPE|WORK_TYPE|DOMAIN|WEBSITE|ZIP_CODE|COUNTRY|YEAR|SNAB)\]'
)
RE_TRAIL_START = re.compile(
    r'(?is)\b(приложени[ея]|ведомость\s+мтр|спецификаци[яи]|счет\s+на\s+оплату|расчет\s+стоимости)\b'
)



def _norm_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()



def _marker_ratio(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    markers = RE_ANY_ANON_TOKEN.findall(line)
    return len(markers) / len(tokens)



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
# 0.1) Сжать хвосты приложений в маркер [ATTACHMENTS]
# ============================================================
def truncate_trailing_attachments(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    m = RE_TRAIL_START.search(s)
    if not m:
        return s
    prefix = s[:m.start()].rstrip()
    if not prefix:
        return "[ATTACHMENTS]"
    return prefix + " [ATTACHMENTS]"



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
# 1.1) Схлопывание строк-таблиц с преобладанием маркеров
# ============================================================
def remove_marker_tables(
    text: str,
    min_marker_ratio: float = 0.4,
    min_run_lines: int = 2,
) -> str:
    if text is None:
        return ""


    lines = str(text).splitlines()
    out = []
    i = 0
    n = len(lines)


    while i < n:
        ratio = _marker_ratio(lines[i])
        if ratio >= min_marker_ratio:
            j = i + 1
            while j < n and _marker_ratio(lines[j]) >= min_marker_ratio:
                j += 1
            if j - i >= min_run_lines:
                out.append("[MARKER_TABLE]")
            else:
                out.extend(lines[i:j])
            i = j
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
# 3.1) Схлопывание подряд одинаковых анонимизационных маркеров
# ============================================================
def collapse_marker_runs(text: str) -> str:
    if text is None:
        return ""


    tokens = str(text).split()
    out = []
    i = 0
    L = len(tokens)


    while i < L:
        tok = tokens[i]
        if RE_ANON_MARKER.fullmatch(tok):
            j = i + 1
            while j < L and tokens[j] == tok:
                j += 1
            out.append(tok)
            i = j
        else:
            out.append(tok)
            i += 1


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
                gram = tok[i : i + n]


                r = 1
                while (
                    i + (r + 1) * n <= L
                    and tok[i + r * n : i + (r + 1) * n] == gram
                ):
                    r += 1


                if r >= NGRAM_RUN_MIN:
                    best_n, best_r = n, r
                    break


            if best_n > 0:
                out.extend(tok[i : i + best_n])
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
                if tok[i : i + k] == tok[i + k : i + 2 * k]:
                    found_k = k
                    break
                k -= 1


            if found_k:
                out.extend(tok[i : i + found_k])
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
    t = truncate_trailing_attachments(t)
    t = remove_lists(t)
    t = remove_marker_tables(t)
    t = remove_duplicate_lines(t)
    t = collapse_token_runs(t)
    t = collapse_marker_runs(t)
    t = collapse_ngram_runs(t)
    t = remove_repeated_token_chains(t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _content_token_ratio(text: str) -> tuple[float, int]:
    tokens = text.split()
    if not tokens:
        return 0.0, 0
    content = [t for t in tokens if not RE_ANY_ANON_TOKEN.fullmatch(t)]
    return len(content) / len(tokens), len(content)


def make_df_clean(
    df: pd.DataFrame,
    output_dir: str | Path,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    required_cols = {label_col, text_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_df.csv"


    shape_before = df.shape
    label_stats_before = df[label_col].value_counts(dropna=False).sort_index()
    total_chars_before = df[text_col].fillna("").astype(str).str.len().sum()


    logger.info("Shape before cleaning: %s", shape_before)
    logger.info("Total chars before cleaning: %d", int(total_chars_before))


    df_clean = df[[label_col, text_col]].copy()
    original_text = df_clean[text_col].fillna("").astype(str).str.strip()
    cleaned_text = original_text.apply(clean_text)


    edited_mask = cleaned_text.ne(original_text)
    empty_mask = cleaned_text.str.strip().eq("")
    ratios = cleaned_text.apply(_content_token_ratio)
    low_content_mask = ratios.apply(
        lambda rv: rv[0] < 0.15 or rv[1] < 5
    )
    removed_mask = empty_mask | low_content_mask
    keep_mask = ~removed_mask


    removed_by_label = (
        df_clean.loc[removed_mask, label_col]
        .value_counts(dropna=False)
        .sort_index()
    )
    edited_by_label = (
        df_clean.loc[edited_mask & keep_mask, label_col]
        .value_counts(dropna=False)
        .sort_index()
    )


    df_clean[text_col] = cleaned_text
    df_clean = df_clean.loc[keep_mask].reset_index(drop=True)


    shape_after = df_clean.shape
    label_stats_after = df_clean[label_col].value_counts(dropna=False).sort_index()
    total_chars_after = df_clean[text_col].fillna("").astype(str).str.len().sum()


    all_labels = sorted(
        set(label_stats_before.index)
        .union(set(label_stats_after.index))
        .union(set(removed_by_label.index))
        .union(set(edited_by_label.index)),
        key=lambda x: str(x),
    )


    logger.info("Shape after cleaning: %s", shape_after)
    logger.info("Total chars after cleaning: %d", int(total_chars_after))
    logger.info("Label stats summary:")


    for label in all_labels:
        before_cnt = int(label_stats_before.get(label, 0))
        removed_cnt = int(removed_by_label.get(label, 0))
        edited_cnt = int(edited_by_label.get(label, 0))
        after_cnt = int(label_stats_after.get(label, 0))
        logger.info(
            "label=%s | before=%d | removed=%d | edited=%d | remaining=%d",
            label,
            before_cnt,
            removed_cnt,
            edited_cnt,
            after_cnt,
        )


    df_clean.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataframe to: %s", output_path)
    return df_clean