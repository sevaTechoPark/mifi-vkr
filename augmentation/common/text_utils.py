import re
import html


def preprocess_text(text: str) -> str:
    text = re.sub(r"[-=_*]{5,}", "—", text)
    text = re.sub(r"[.\s]{5,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


_URL_PATTERN = re.compile(r"https?://\S+")
_MULTIPUNCT_PATTERN = re.compile(r"([,.;:!?])\1{1,}")
_SPACES_PATTERN = re.compile(r"\s+")

# расширенный паттерн PH-мусора: <PS_0>, <FH_10>, <PH_3>, HP_13, CRR итп
_PH_GARBAGE_PATTERN = re.compile(
    r"""
    (
      <\s*[A-Za-z]{1,4}[_\-\s]*\d+\s*>   # <PS_0>, <FH_10>, < PH-3 >
    |
      \b[A-Z]{1,3}[HPh][_\-]?\d+\b        # HP_13, FH10, PH3
    |
      \b[Pp][Hh][\w.\-_]*\b               # PH, Ph_7, PH.24, pH-33
    |
      \b[A-Z]{2,4}_\d+\b                  # CRR, PSG, RH_17
    )
    """,
    re.VERBOSE,
)

# HTML-entities
_HTML_ENTITY_PATTERN = re.compile(r"&(?:lt|gt|amp|quot|nbsp|apos);")

# реквизитные заголовки без значимого содержания
# ловит: "Телефон:", "e-mail:", "mail:", "адрес:", "и:", "или:"
_LABEL_NOISE_PATTERN = re.compile(
    r"\b(?:тел(?:ефон)?|e[\-\s]?mail|mail|факс|fax|адрес|эл\.?\s*почта|бухгалтерия\s+тел|птo\s+тел)\s*:[\s,;.]*",
    re.IGNORECASE,
)

# паразитные строки вида "должность: - должность:" или ": и." или ":,"
_COLON_NOISE_PATTERN = re.compile(r"(?:должность|контактный телефон|контакт)\s*:\s*[-—]*\s*", re.IGNORECASE)

# одиночные незначимые символы или их цепочки: "Щ Щ Щ", "*", "[]", "♪", "{>", "#>"
_SYMBOL_NOISE_PATTERN = re.compile(
    r"""
    (
      (?:[ЩщЪъЫы]\s+){2,}   # цепочки паразитных букв
    |
      [♪♫♬♩✓✗→←↑↓]          # музыкальные и стрелочные символы
    |
      \{\s*[><!]             # {>, {!, {<
    |
      [#<>]\s*[><!]?         # #>, <., >.
    |
      \[\s*\]                # пустые скобки []
    |
      \*\s*\.?\s*            # * или *.
    )
    """,
    re.VERBOSE,
)


def clean_generated_text(text: str) -> str:
    # HTML entities → нормальные символы
    text = html.unescape(text)

    # убираем URL
    text = _URL_PATTERN.sub("", text)

    # убираем реквизитные заголовки-одиночки
    text = _LABEL_NOISE_PATTERN.sub("", text)

    # убираем "должность: -" и подобные хвосты
    text = _COLON_NOISE_PATTERN.sub("", text)

    # убираем паразитные символы и цепочки
    text = _SYMBOL_NOISE_PATTERN.sub(" ", text)

    # схлопываем повторяющуюся пунктуацию
    text = _MULTIPUNCT_PATTERN.sub(r"\1", text)

    # вычищаем PH-артефакты
    text = _PH_GARBAGE_PATTERN.sub("", text)

    # нормализуем пробелы
    text = _SPACES_PATTERN.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # убираем «висячие» начальные пунктуационные цепочки вида ";,." в начале текста
    text = re.sub(r"^[;:,.\s]+", "", text)

    return text.strip()


def clean_aug_result(text: str) -> str:
    text = re.sub(r"([,.!?])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    return text.strip()


def is_highly_formal(text: str) -> bool:
    digits = sum(ch.isdigit() for ch in text)
    uppers = sum(ch.isupper() for ch in text)
    letters = sum(ch.isalpha() for ch in text)

    if letters < 80:
        return False

    digit_ratio = digits / max(1, letters)
    upper_ratio = uppers / max(1, letters)

    recviz_words = ["ОГРН", "ОКПО", "ОКТМО", "ОКВЭД", "ИНН", "КПП", "БИК", "р/с", "к/с"]
    has_rekviz = any(w in text for w in recviz_words)

    if has_rekviz and (digit_ratio > 0.35 or upper_ratio > 0.55):
        return True
    return False
