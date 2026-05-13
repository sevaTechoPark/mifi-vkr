# augmentation/common/masks.py
import re
from typing import Tuple, Dict


# [ORGANIZATION], [DATE_TIME] и т.п.
_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+\]")

# внутренний формат маски, которая идёт в модель
_MASK_TOKEN_TEMPLATE = "<PH_{idx}>"

# паттерн для любых «хвостов» масок, которые нужно удалить после демаскировки
# ловит варианты:
#   <PH_0>, < PH-1 >, PH_2, ph-3, http://PH_4, и т.п.
_MASK_GARBAGE_PATTERN = re.compile(
    r"""
    (https?://[Pp][Hh][_\-\s]*\d+   # ссылки вида http://PH_12
    |
     <\s*[Pp][Hh][_\-\s]*\d+\s*>    # теги <PH_12>, < PH-12 >
    |
     \b[Pp][Hh][_\-\s]*\d+\b        # просто PH_12, ph-12
    )
    """,
    re.VERBOSE,
)


def mask_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Заменяем [ORGANIZATION], [DATE_TIME] и т.п. на <PH_0>, <PH_1>, ...
    """
    mapping: Dict[str, str] = {}
    counter = 0

    def replacer(m: re.Match) -> str:
        nonlocal counter
        original = m.group(0)
        key = _MASK_TOKEN_TEMPLATE.format(idx=counter)
        mapping[key] = original
        counter += 1
        return key

    masked = _PLACEHOLDER_PATTERN.sub(replacer, text)
    return masked, mapping


def unmask_placeholders(text: str, mapping: Dict[str, str]) -> str:
    """
    1) Пытаемся восстановить все маски по точному ключу (<PH_0> -> [ORGANIZATION]).
    2) Всё, что осталось от PH-масок (PH_12, <PH_ 3>, http://PH_4) — вычищаем как мусор.
    """
    # 1. точное восстановление известных масок
    for key in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(key, mapping[key])

    # 2. удалить любые остатки PH_* из текста
    text = _MASK_GARBAGE_PATTERN.sub("", text)

    # 3. чуть подчистим пробелы после удаления мусора
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()
    