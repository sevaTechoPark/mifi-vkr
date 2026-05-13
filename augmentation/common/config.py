TARGET_PER_CLASS = 30
EMBED_MODEL_NAME = "deepvk/USER2-base"
RUGPT_MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"

SIM_LABEL_MIN = 0.8
SIM_LABEL_MAX = 0.98
# ── Back-translation ───────────────────────────────────────────────
BT_SIM_MIN = 0.80
BT_SIM_MAX = 0.95

BT_MIN_LEN_RATIO = 0.50
BT_MAX_LEN_RATIO = 1.50

# ── Paraphrase ─────────────────────────────────────────────────────
PARA_SIM_MIN = 0.75
PARA_SIM_MAX = 0.95

# сколько символов считаем "коротким" исходником
SHORT_TEXT_THRESHOLD = 700

# минимальное отношение длины paraphrase/source для коротких текстов
PARAPHRASE_MIN_LEN_RATIO_SHORT = 0.40
PARA_MIN_LEN_RATIO = 0.5
PARA_MAX_LEN_RATIO = 1.30

