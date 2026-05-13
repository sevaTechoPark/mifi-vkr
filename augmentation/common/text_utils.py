import re
import html


def preprocess_text(text: str) -> str:
    """
    Предварительная нормализация исходного текста перед генерацией.
    """
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
      \b[A-Z]{1,3}[HPh][_\-]?\d+\b       # HP_13, FH10, PH3
    |
      \b[Pp][Hh][\w.\-_]*\b              # PH, Ph_7, PH.24, pH-33
    |
      \b[A-Z]{2,4}_\d+\b                 # CRR_12, PSG_4
    )
    """,
    re.VERBOSE,
)


# HTML-entities
_HTML_ENTITY_PATTERN = re.compile(r"&(?:lt|gt|amp|quot|nbsp|apos);")


# реквизитные заголовки без значимого содержания
# ловит: "Телефон:", "e-mail:", "mail:", "адрес:", "бухгалтерия тел.:", "пто тел.:"
_LABEL_NOISE_PATTERN = re.compile(
    r"\b(?:тел(?:ефон)?|тел\.?|e[\-\s]?mail|mail|факс|fax|адрес|эл\.?\s*почта|бухгалтерия\s+тел|пто\s+тел)\s*:[\s,;.]*",
    re.IGNORECASE,
)


# паразитные строки вида "должность: - должность:" или "контактный телефон: -"
_COLON_NOISE_PATTERN = re.compile(
    r"(?:должность|контактный телефон|контакт)\s*:\s*[-—]*\s*",
    re.IGNORECASE,
)


# Кириллические PH-артефакты: ПН-5, ПС_12, ПХ-3, ФХ_7 и т.п.
_CYR_PH_GARBAGE_PATTERN = re.compile(
    r"\b[ПФХКРС][НСХ][\-_]?\d+\b",
    re.IGNORECASE,
)


# Хвостовые артефакты «Включен и подписан на марку ...» и «Дата начала/окончания поиска»
_TRAILING_ARTIFACT_PATTERN = re.compile(
    r"""
    (?:
      включен\s+и\s+подписан\s+на\s+марку
    | дата\s+(?:начала|окончания)\s+поиска
    | [иИ]мпр[её]ктор\b             # "импректор"
    | \bл\.с\b                      # "л.с" как хвост
    | \b(?:на|и)\s*$                # висячие "на ", "и " в самом конце
    )
    [\s\S]{0,40}$                    # + хвост до 40 символов
    """,
    re.IGNORECASE | re.VERBOSE,
)


# HTML-подобные атрибуты: href="...", src="...", class="...", id="...", style="...", data-...="...", ref="..."
_HTML_ATTR_GARBAGE_PATTERN = re.compile(
    r"""\b(?:href|src|class|id|style|data-[\w\-]+|ref)\s*=\s*"(?:[^"]{0,80})" """,
    re.IGNORECASE | re.VERBOSE,
)


# последовательности из трёх и более одинаковых одиночных небуквенно-цифровых символов,
# разделённых пробелами: "- - - - - -", ". . . .", "/ / /", ": : :" и т.п.
_PUNCT_SEQ_WITH_SPACES_PATTERN = re.compile(
    r"(?:\s*([^\w\s])\s*){3,}"
)


# одиночные незначимые символы или их цепочки: "Щ Щ Щ", "*", "[]", "♪", "{>", "#>"
_SYMBOL_NOISE_PATTERN = re.compile(
    r"""
    (
      (?:[ЩщЪъЫы]\s+){2,}        # цепочки паразитных букв
    |
      [♪♫♬♩✓✗→←↑↓]               # музыкальные и стрелочные символы
    |
      \{\s*[><!]                 # {>, {!, {<
    |
      [#<>]\s*[><!]?             # #>, <., >.
    |
      \[\s*\]                    # пустые скобки []
    |
      \*\s*\.?\s*                # * или *.
    )
    """,
    re.VERBOSE,
)


# коннекторы без содержимого: "и:", "или:" в хвостах
_CONNECTIVE_LABEL_NOISE_PATTERN = re.compile(
    r"\b(?:и|или)\s*:\s*(?=[,;.!\s]|$)",
    re.IGNORECASE,
)


# куски, состоящие почти полностью из пунктуации/цифр/слэшей — явный мусор
_PUNCT_ONLY_CHUNK_PATTERN = re.compile(
    r"(?:\s*[.\-–—/\\]{2,}\s*|\s*[.\-–—/\\\d]{3,}\s*)"
)


# пустые или почти пустые скобки: "()", "( )"
_EMPTY_PARENS_PATTERN = re.compile(r"\(\s*\)")


# повторы коротких токенов с точкой: "и. и. состоится", "г.г.г.г."
_SHORT_TOKEN_REPEAT_PATTERN = re.compile(
    r"\b([А-ЯЁа-яёA-Za-z]{1,3}\.)\s+\1\b"
)


def _drop_pure_latin_short_lines(text: str) -> str:
    """
    Короткие "англоязычные" строки / хвосты без кириллицы — часто чистый мусор.
    """
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # если нет кириллицы и строка короткая — считаем её артефактом
        if not re.search(r"[А-Яа-я]", line) and len(line) <= 40:
            continue
        lines.append(line)
    return " ".join(lines)


def _dedup_repeated_tokens(text: str) -> str:
    """
    Схлопываем повторы типа 'Исполнитель: Исполнитель: Исполнитель'
    или 'Исполнитель: Исполнитель:' в одно.
    """
    # повторяющиеся пары с двоеточием: "Исполнитель: Исполнитель:" → "Исполнитель:"
    text = re.sub(r"\b(\w+:\s*)(\1){1,}", r"\1", text)
    # повторяющиеся слова: "Покупатель Покупатель" → "Покупатель"
    text = re.sub(r"\b(\w+)(?:\s+\1\b){1,}", r"\1", text)
    return text


def clean_generated_text(text: str) -> str:
    """
    Пост-обработка сгенерированного текста (BT и paraphrase):
    - раскодируем HTML-entities,
    - вырезаем короткие англоязычные артефактные строки,
    - убираем URL, хвосты "Телефон:, mail:, адрес: и:",
    - чистим PH/PSG/CRR-артефакты, HTML-атрибуты, символический и пунктуационный мусор,
    - аккуратно нормализуем пробелы и "висячие" хвосты.
    """
    # HTML entities → нормальные символы
    text = html.unescape(text)

    # выбрасываем короткие строки без кириллицы — типичные артефакты
    text = _drop_pure_latin_short_lines(text)

    # убираем URL
    text = _URL_PATTERN.sub("", text)

    # убираем HTML-подобные атрибуты href="...", src="...", class="..." и т.п.
    text = _HTML_ATTR_GARBAGE_PATTERN.sub(" ", text)

    # убираем реквизитные заголовки-одиночки без значимого содержимого
    text = _LABEL_NOISE_PATTERN.sub("", text)

    # убираем "и:" / "или:" как пустой хвост
    text = _CONNECTIVE_LABEL_NOISE_PATTERN.sub("", text)

    # убираем "должность: -" и подобные хвосты
    text = _COLON_NOISE_PATTERN.sub("", text)

    # убираем паразитные символы и цепочки
    text = _SYMBOL_NOISE_PATTERN.sub(" ", text)

    # убираем чисто пунктуационные/цифровые фрагменты
    text = _PUNCT_ONLY_CHUNK_PATTERN.sub(" ", text)

    # убираем последовательности из повторяющихся знаков с пробелами: "- - - -", ". . ."
    text = _PUNCT_SEQ_WITH_SPACES_PATTERN.sub(" ", text)

    # убираем пустые скобки
    text = _EMPTY_PARENS_PATTERN.sub(" ", text)

    # схлопываем повторы коротких токенов с точкой ("и. и." → "и.")
    text = _SHORT_TOKEN_REPEAT_PATTERN.sub(r"\1", text)

    # схлопываем повторяющуюся пунктуацию
    text = _MULTIPUNCT_PATTERN.sub(r"\1", text)

    # вычищаем PH-артефакты
    text = _PH_GARBAGE_PATTERN.sub("", text)
    text = _CYR_PH_GARBAGE_PATTERN.sub("", text)

    # схлопываем повторяющиеся токены/фразы
    text = _dedup_repeated_tokens(text)

    # нормализуем пробелы
    text = _SPACES_PATTERN.sub(" ", text)

    # чистим хвостовые артефакты типа "Дата окончания поиска ..."
    text = _TRAILING_ARTIFACT_PATTERN.sub("", text)

    # убираем пробелы перед пунктуацией
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # убираем «висячие» начальные пунктуационные цепочки вида ";,." в начале текста
    text = re.sub(r"^[;:,.\s]+", "", text)

    return text.strip()


def clean_aug_result(text: str) -> str:
    """
    Лёгкая нормализация результатов генерации/перевода.
    Используется там, где не нужна тяжёлая чистка.
    """
    text = re.sub(r"([,.!?])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    return text.strip()


def is_highly_formal(text: str) -> bool:
    """
    Эвристика для отсева сверхформальных, реквизитных текстов
    (применяется только в перефразе, BT идёт всегда).
    """
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