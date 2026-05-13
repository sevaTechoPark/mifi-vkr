import re
from typing import Tuple, Dict


_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+\]")          # [ORGANIZATION], [DATE_TIME] и т.п.
_MASK_TOKEN_TEMPLATE = "<PH_{idx}>"                       # новый формат маски
_MASK_LEFTOVER_PATTERN = re.compile(r"<PH_\d+>")          # для safety-check’а


def mask_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Заменяем [ORGANIZATION], [DATE_TIME] и т.п. на <PH_0>, <PH_1>, ...
    Токены вида <PH_N> модели обычно оставляют как есть.
    """
    mapping: Dict[str, str] = {}
    counter = 0

    def replacer(m: re.Match) -> str:
        nonlocal counter
        original = m.group(0)                   # пример: [ORGANIZATION]
        key = _MASK_TOKEN_TEMPLATE.format(idx=counter)  # <PH_0>
        mapping[key] = original
        counter += 1
        return key

    masked = _PLACEHOLDER_PATTERN.sub(replacer, text)
    return masked, mapping


def unmask_placeholders(text: str, mapping: Dict[str, str]) -> str:
    """
    Восстанавливаем плейсхолдеры по ключам <PH_0>, <PH_1>, ...
    Чтобы избежать конфликтов вида <PH_1> внутри <PH_10>,
    заменяем в порядке убывания длины ключа.
    """
    for key in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(key, mapping[key])
    return text


def has_leftover_mask_tokens(text: str) -> bool:
    """
    Проверка: остались ли в тексте маски вида <PH_123>.
    Если да — значит модель всё-таки потрогала токены, и такой текст лучше не использовать.
    """
    return _MASK_LEFTOVER_PATTERN.search(text) is not None
    