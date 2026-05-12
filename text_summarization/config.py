SUMMARIZATION_MODEL: str = "IlyaGusev/rut5_base_sum_gazeta"
# Альтернативы:
#   "IlyaGusev/mbart_ru_sum_gazeta"
#   "d0rj/rut5-base-summ"

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

MAX_INPUT_LENGTH: int = 600
MAX_SUMMARY_LENGTH: int = 128
MIN_SUMMARY_LENGTH: int = 30
BATCH_SIZE: int = 8

SEPARATOR: str = " [SEP] "