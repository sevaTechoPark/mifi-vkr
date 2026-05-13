# augmentation

Набор модулей для аугментации текстовых данных:

- **backtranslate** — обратный перевод через MarianMT.
- **paraphrase** — перефразирование на `ruT5-large-paraphraser`.
- **common** — общие утилиты (эмбеддинги, плейсхолдеры, конфиги, цикл аугментации).

## Структура

```text
augmentation/
├── common
│   ├── __init__.py
│   ├── config.py        # константы: TARGET_PER_CLASS, SIM_LABEL_*, BT_*, PARA_*
│   ├── masks.py         # mask_placeholders / unmask_placeholders
│   ├── embeddings.py    # load_embed_model, cos_sim
│   ├── perplexity.py    # load_rugpt, rugpt_perplexity_list
│   ├── seed.py          # set_seed, get_seed_or_default
│   ├── text_utils.py    # preprocess_text, clean_generated_text, is_highly_formal, clean_aug_result
│   └── augment_loop.py  # run_augmentation_loop — общий цикл по "малым" классам
├── backtranslate
│   ├── __init__.py
│   ├── models.py        # загрузка моделей MarianMT с кешированием
│   ├── translate.py     # mt_tokens_len, split_long_sentence, generate_translate, safe_translate
│   ├── augment.py       # generate_bt_candidates, choose_best_bt, back_translate_document
│   ├── main.py          # run_from_params + CLI (--train / --output-dir / --single)
│   └── README.md
└── paraphrase
    ├── __init__.py
    ├── models.py        # загрузка ruT5-large-paraphraser с кешированием
    ├── phrase.py        # split_long_sentence, generate_phrase, safe_paraphrase, postprocess_paraphrase_text
    ├── augment.py       # generate_para_candidates, choose_best_paraphrase, paraphrase_document
    ├── main.py          # run_from_params + CLI (--train / --output-dir / --single)
    └── README.md
```

## Общий цикл аугментации (`common/augment_loop.py`)

`run_augmentation_loop` реализует единый цикл для любых методов аугментации:

- добивает малые классы до `TARGET_PER_CLASS`,
- следит за:
  - **похожестью с исходником** — окно `[sim_min, sim_max]`,
  - **похожестью с классом** — окно `[SIM_LABEL_MIN, SIM_LABEL_MAX]`,
  - **соотношением длин** — `[min_len_ratio, max_len_ratio]`,
- ведёт частичный лог в `*_partial.csv`, чтобы можно было продолжить после прерывания.

Сигнатура:

```python
run_augmentation_loop(
    df: pd.DataFrame,
    embed_model: SentenceTransformer,
    augment_fn,           # callable(text: str) -> str
    aug_file_path: str,
    augmentation_type: str,
    sim_min: float,
    sim_max: float,
    min_len_ratio: float,
    max_len_ratio: float,
) -> pd.DataFrame
```

### Конфигурация (`common/config.py`)

```python
TARGET_PER_CLASS   = 5

EMBED_MODEL_NAME   = "deepvk/USER2-base"
RUGPT_MODEL_NAME   = "sberbank-ai/rugpt3small_based_on_gpt2"

SIM_LABEL_MIN      = 0.8
SIM_LABEL_MAX      = 0.98

# Back-translation
BT_SIM_MIN         = 0.80
BT_SIM_MAX         = 0.95
BT_MIN_LEN_RATIO   = 0.50
BT_MAX_LEN_RATIO   = 1.50

# Paraphrase
PARA_SIM_MIN       = 0.85
PARA_SIM_MAX       = 0.97
PARA_MIN_LEN_RATIO = 0.70
PARA_MAX_LEN_RATIO = 1.30
```

- BT использует более широкое окно по длине, поскольку обратный перевод может чуть расширять/сжимать текст.
- Paraphrase — более строгие пороги по cosine и длине, чтобы отсечь «конспекты» длинных писем и слишком близкие копии.[web:50]