import re


def mask_placeholders(text: str) -> tuple[str, dict]:
    pattern = r'\[[^\]]+\]'
    mapping = {}
    counter = [0]

    def replacer(m):
        token = m.group(0)
        key = f"MaskTok{counter[0]}X"
        mapping[key] = token
        counter[0] += 1
        return key

    masked = re.sub(pattern, replacer, text)
    return masked, mapping


def unmask_placeholders(text: str, mapping: dict) -> str:
    for key in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(key, mapping[key])
    return text