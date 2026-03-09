import re

_ws_re = re.compile(r"\s+")
_quotes_map = {
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"',
    "’": "'", "‘": "'",
}
_dashes_re = re.compile(r"[‐‑‒–—―]")  # разные тире/дефисы

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    # ё -> е
    s = s.replace("ё", "е").replace("Ё", "Е")

    # унификация кавычек
    for a, b in _quotes_map.items():
        s = s.replace(a, b)

    # унификация тире/дефисов
    s = _dashes_re.sub("-", s)

    # пробелы
    s = _ws_re.sub(" ", s).strip()

    return s