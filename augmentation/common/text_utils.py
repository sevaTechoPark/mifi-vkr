import re

def preprocess_text(text: str) -> str:
    """
    Убираем шумовые конструкции до генерации:
    длинные цепочки символов, множественные пробелы.
    """
    text = re.sub(r"[-=_*]{5,}", "—", text)
    text = re.sub(r"[.\s]{5,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# паттерн для явного мусора в сгенерированных текстах
_URL_PATTERN = re.compile(r"https?://\S+")
_MULTIPUNCT_PATTERN = re.compile(r"([,.;:!?])\1{1,}")  # ,,  ..  !!!  → одна
_SPACES_PATTERN = re.compile(r"\s+")

# артефакты старых масок и T5-галлюцинаций (PH_7, PH.24, PCH., PS_33 и т.п.)
_PH_GARBAGE_PATTERN = re.compile(
    r"""
    \b[Pp][Hh][\w\.\-\_]*\b     # PH, Ph_7, PH.24, pH-33, PCH., PS_33
    """,
    re.VERBOSE,
)


def clean_generated_text(text: str) -> str:
    """
    Лёгкая пост-обработка сгенерированного текста:
    - убираем URL'ы, повторяющуюся пунктуацию,
    - вычищаем артефакты PH/PCH/PH.24,
    - схлопываем пробелы.
    """
    # убираем случайные URL'ы (T5 любит их галлюцинировать)
    text = _URL_PATTERN.sub("", text)

    # схлопываем повторяющуюся пунктуацию
    text = _MULTIPUNCT_PATTERN.sub(r"\1", text)

    # вычищаем PH-артефакты, не затрагивая [PLACEHOLDER]
    text = _PH_GARBAGE_PATTERN.sub("", text)

    # аккуратная нормализация пробелов
    text = _SPACES_PATTERN.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

def is_highly_formal(text: str) -> bool:
    """
    Грубая эвристика: считаем текст 'формальным' (деловые письма с реквизитами),
    если:
      - текст достаточно длинный,
      - явно есть реквизитные маркеры,
      - очень большая доля цифр или заглавных.
    """
    digits = sum(ch.isdigit() for ch in text)
    uppers = sum(ch.isupper() for ch in text)
    letters = sum(ch.isalpha() for ch in text)

    if letters < 80:
        # короткие тексты почти никогда не считаем "слишком формальными"
        return False

    # отношение цифр/букв
    digit_ratio = digits / max(1, letters)
    upper_ratio = uppers / max(1, letters)

    recviz_words = ["ОГРН", "ОКПО", "ОКТМО", "ОКВЭД", "ИНН", "КПП", "БИК", "р/с", "к/с"]
    has_rekviz = any(w in text for w in recviz_words)

    if has_rekviz and (digit_ratio > 0.35 or upper_ratio > 0.55):
        return True
    return False

def clean_aug_result(text: str) -> str:
    """
    Лёгкая пост-обработка результатов генерации/перевода:
    - схлопываем повторяющиеся знаки препинания,
    - чистим пробелы вокруг запятых,
    - убираем лишние пробелы.
    """
    text = re.sub(r"([,.!?])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*", ", ", text)
    return text.strip()