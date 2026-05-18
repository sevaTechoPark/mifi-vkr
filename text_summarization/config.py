SUMMARIZATION_MODEL: str = "IlyaGusev/rut5_base_sum_gazeta"
# Альтернативы:
#   "IlyaGusev/mbart_ru_sum_gazeta"
#   "d0rj/rut5-base-summ"

# это потолок снизу. Реальный лимит = min(этого значения, tokenizer.model_max_length).
# Для IlyaGusev/rut5_base_sum_gazeta родной максимум 512–1024 токенов; 600 — компромисс.
# При смене модели на mbart_ru_sum_gazeta ставить 1024.
MAX_INPUT_LENGTH: int = 600

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

MAX_INPUT_LENGTH: int = 600
MAX_SUMMARY_LENGTH: int = 128
MIN_SUMMARY_LENGTH: int = 30
BATCH_SIZE: int = 8

SEPARATOR: str = " [SEP] "