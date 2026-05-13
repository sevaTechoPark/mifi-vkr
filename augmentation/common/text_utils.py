# augmentation/common/text_utils.py
import re


def preprocess_text(text: str) -> str:
    """
    Убираем шумовые конструкции: длинные цепочки символов,
    множественные пробелы.
    """
    text = re.sub(r"[-=_*]{5,}", "—", text)
    text = re.sub(r"[.\s]{5,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()