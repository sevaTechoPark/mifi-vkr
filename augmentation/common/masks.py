import re
from typing import Tuple, Dict, List


_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+\]")      # [ORGANIZATION], [DATE_TIME], ...
_MASK_TOKEN_TEMPLATE = "<PH_{idx}>"


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
    Восстанавливаем плейсхолдеры по ключам <PH_0>, <PH_1>, ...
    Чтобы избежать конфликтов вида <PH_1> внутри <PH_10>,
    заменяем в порядке убывания длины ключа.
    """
    for key in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(key, mapping[key])
    return text


def _extract_placeholders(text: str) -> List[str]:
    """
    Вытащить все плейсхолдеры вида [SOMETHING] из текста.
    """
    return _PLACEHOLDER_PATTERN.findall(text)


def placeholders_intact(original: str, augmented: str) -> bool:
    """
    Проверка: все плейсхолдеры из оригинального текста присутствуют
    в аугментированном, в том же количестве.

    Если модель \"сломала\" или удалила хотя бы один тег — вернём False.
    """
    orig_tags = _extract_placeholders(original)
    aug_tags  = _extract_placeholders(augmented)
    return sorted(orig_tags) == sorted(aug_tags)
