# text_summarization

Модуль для создания суммаризированных датасетов на основе русскоязычных seq2seq-моделей с HuggingFace. Предназначен для аугментации данных при задачах классификации текстов.

По одному входному датасету генерирует **два** выходных файла:
- `<name>_summarized.csv` — текст заменён суммаризацией
- `<name>_original_plus_summary.csv` — оригинал `[SEP]` суммаризация

***

## Структура модуля

```
text_summarization/
├── __init__.py       # экспортирует run()
├── config.py         # все константы и гиперпараметры
├── model.py          # загрузка модели суммаризации с HuggingFace
├── summarize.py      # батчевая суммаризация текстов
├── data.py           # загрузка, построение и сохранение датасетов
├── main.py           # точка входа: функция run()
└── README.md
```

***

## API — `run()`

```python
run(
    input_path: str,          # путь к датасету (.csv / .json / .jsonl)
    output_dir: str,          # директория для сохранения результатов
    model_name: str,          # модель суммаризации (default: d0rj/rut5-base-summ)
    text_column: str,         # колонка с текстами (default: "text")
    label_column: str,        # колонка с метками (default: "label")
    batch_size: int,          # размер батча (default: 8, уменьшить при OOM)
    separator: str,           # разделитель orig+summ (default: " [SEP] ")
    max_input_length: int,    # макс. токенов на вход (default: 600)
    max_summary_length: int,  # макс. токенов суммаризации (default: 128)
    min_summary_length: int,  # мин. токенов суммаризации (default: 30)
) -> tuple[pd.DataFrame, pd.DataFrame]   # (df_summarized, df_combined)
```

***

## Запуск в Google Colab

**Ячейка 1 — установка и монтирование диска**
```python
!pip install transformers torch pandas tqdm accelerate -q

from google.colab import drive
drive.mount("/content/drive")
```

**Ячейка 2 — клонирование репы**
```python
!git clone https://github.com/<you>/<repo>.git /content/<repo>

import sys
sys.path.insert(0, "/content/<repo>")
```

**Ячейка 3 — запуск**
```python
import os
from text_summarization.main import run

drive_root = "/content/drive/MyDrive"

INPUT_PATH = os.path.join(drive_root, "train_augmented.csv")
OUTPUT_DIR = os.path.join(drive_root, "summary")

df_summarized, df_combined = run(
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
    # model_name="IlyaGusev/rut5_base_sum_gazeta",
    # batch_size=4,   # уменьшить если OOM
)
```

***

## Гарантия совпадения индексов

Строка `i` во всех трёх файлах — один и тот же пример. В каждый файл добавляется колонка `sample_id`:

```python
df_orig = pd.read_csv("train_augmented.csv")
df_summ = pd.read_csv("train_augmented_summarized.csv")

assert (df_orig["sample_id"] == df_summ["sample_id"]).all()
```

***

## Зависимости

```
torch
transformers
pandas
tqdm
accelerate
```