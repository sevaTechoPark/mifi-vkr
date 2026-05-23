"""
Конфигурация пайплайна суммаризации.

Содержит идентификатор модели, имена колонок, ограничения по длине входа
и выхода, размер батча и разделитель для конкатенированных текстов.
"""

# Модель для суммаризации (HuggingFace Hub).
# Альтернативные варианты для русских текстов:
#   "IlyaGusev/mbart_ru_sum_gazeta"   — при использовании поднять MAX_INPUT_LENGTH до 1024
#   "d0rj/rut5-base-summ"
SUMMARIZATION_MODEL: str = "IlyaGusev/rut5_base_sum_gazeta"

# Имена колонок во входном датасете.
TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

# Ограничения по длине.
# MAX_INPUT_LENGTH — целевая длина входа в токенах. Фактический лимит
# вычисляется как min(MAX_INPUT_LENGTH, tokenizer.model_max_length).
# Для rut5_base_sum_gazeta родной максимум 512–1024 токенов; 600 — компромисс.
MAX_INPUT_LENGTH: int = 600
MAX_SUMMARY_LENGTH: int = 128
MIN_SUMMARY_LENGTH: int = 30

BATCH_SIZE: int = 8

# Разделитель между оригинальным текстом и суммаризацией в комбинированном
# датасете.
SEPARATOR: str = " [SEP] "
