# backtranslate

Модуль аугментации текстовых данных методом **обратного перевода** (back-translation).

Идея: перевести исходный русский текст на промежуточный язык (английский, французский или испанский), а затем обратно на русский. Разные языки-посредники и режимы генерации дают разные перефразировки, сохраняя при этом смысл исходного текста.

---

## Структура

```text
backtranslate/
├── __init__.py
├── models.py      # Загрузка и кэширование моделей MarianMT (ru-en, en-ru, ru-fr, ...).
├── translate.py   # Низкоуровневые функции перевода и сегментации (mt_tokens_len, split_long_sentence, generate_translate, safe_translate).
├── augment.py     # Генерация и отбор BT-кандидатов, основная функция back_translate_document.
├── main.py        # run_from_params + CLI: единичный пример и цикл по датасету.
└── README.md
```

---

## Как это работает

### 1. Защита плейсхолдеров (`common/masks.py`)

Перед переводом все теги вида `[ORGANIZATION]`, `[DATE_TIME]`, `[PERSON]` и т.п. заменяются на специальные токены `<PH_0>`, `<PH_1>`, ...:

```text
[ORGANIZATION] не согласовывает проезд → <PH_0> не согласовывает проезд
                          ↓ перевод ↓
<PH_0> does not approve the passage → <PH_0> не согласовывает проезд
                          ↓ unmask ↓
[ORGANIZATION] не согласовывает проезд
```

После генерации:

1. Все известные `<PH_i>` восстанавливаются в исходные плейсхолдеры.
2. Любые артефакты вида `PH_12`, `http://PH_4`, `< PH-3 >` удаляются как мусор.

### 2. Предобработка (`common/text_utils.py → preprocess_text`)

Сначала текст нормализуется:

- длинные цепочки из `-=_*` (5+ символов) → `—`,
- цепочки точек/пробелов (5+) → пробел,
- множественные пробелы → один пробел.

Это уменьшает количество мусора для переводчика и сегментатора.

### 3. Сегментация длинных предложений (`translate.py → split_long_sentence`)

MarianMT ограничен по длине входа (мы считаем безопасным лимитом 300 токенов). Для перевода используется:

- `max_tokens=120` при первом разбиении (чтобы оставить запас),
- рекурсивное деление:

  1. Если предложение ≤ `max_tokens` — возвращаем как есть.
  2. Пытаемся разбить по знакам препинания (`,`, `;`, `:`, `—`, `-`).
  3. Если разделителей нет — режем по словам, накапливая чанки.
  4. Если часть всё ещё > `max_tokens` — рекурсивно делим её дальше.

### 4. Низкоуровневый перевод (`translate.py → generate_translate`, `safe_translate`)

- `generate_translate(text, mode, device, **gen_kwargs)`:
  - берёт уже подготовленный кусок текста,
  - проверяет, что длина ≤ 300 токенов (иначе кидает ошибку),
  - запускает `MarianMTModel.generate` с дефолтными параметрами (`num_beams=1`, `do_sample=True`, `top_k=50`, `top_p=0.92`, `temperature=1.1`).

- `safe_translate(text, mode, max_tokens, device, **gen_kwargs)`:
  - режет текст через `split_long_sentence`,
  - для каждого чанка:
    - если длина ≤ 300 токенов — переводит `generate_translate`,
    - если > 300 — максимально грубо режет по словам фиксированным окном (по 20 слов) и переводит отдельно,
  - склеивает переводы обратно в строку.

### 5. Генерация кандидатов (`augment.py → generate_bt_candidates`)

Для каждого предложения исходного текста:

1. Для каждой языковой пары:

   - `("ru-en", "en-ru")`,
   - `("ru-fr", "fr-ru")`,
   - `("ru-es", "es-ru")`,

2. Для каждой конфигурации генерации:

   | Режим             | `do_sample` | `num_beams` | `top_p` | `temperature` |
   |-------------------|------------|-------------|---------|---------------|
   | Beam search       | False      | 5           | —       | —             |
   | Sampling мягкий   | True       | 1           | 0.90    | 1.0           |
   | Sampling агрессивный | True    | 1           | 0.95    | 1.2           |

3. Вызов:

   ```python
   mid = safe_translate(text, mode=src_lang, max_tokens=120, device=device, **cfg)
   bt  = safe_translate(mid,  mode=tgt_lang, max_tokens=300, device=device, **cfg)
   ```

4. Результат прогоняется через `clean_aug_result` (лёгкая нормализация пунктуации и пробелов).

Итого на предложение до 9 сырых BT-кандидатов, после дедупликации — меньше.

### 6. Отбор лучшего кандидата (`augment.py → choose_best_bt`)

```python
def choose_best_bt(
    source_chunk: str,
    candidates: list[str],
    embed_model: SentenceTransformer,
    rugpt_tok,
    rugpt_model,
    rugpt_device: torch.device,
) -> str:
    ...
```

Алгоритм:

1. Если кандидатов нет — возвращаем исходный `source_chunk`.
2. Считаем эмбеддинги для `[source_chunk] + candidates` через `deepvk/USER2-base`.
3. Считаем cosine similarity и фильтруем по окну `[BT_SIM_MIN, BT_SIM_MAX]` = `[0.80, 0.95]`.
4. Для прошедших фильтр считаем перплексию `rugpt_perplexity_list` на `rugpt3small`.
5. Выбираем кандидат с минимальной перплексией; при равенстве — с максимальным cosine.
6. Если никто не попал в окно по cosine — берём кандидат с максимальным cosine.

Функция возвращает **строку** — лучший BT‑вариант.

### 7. Сборка документа (`augment.py → back_translate_document`)

```python
def back_translate_document(
    text_orig: str,
    embed_model: SentenceTransformer,
    rugpt_tok,
    rugpt_model,
    device: torch.device,
) -> str:
    masked_text, mapping = mask_placeholders(text_orig)
    masked_text = preprocess_text(masked_text)
    sentences = [s.text.strip() for s in sentenize(masked_text) if s.text.strip()]

    bt_sentences = []
    for s in tqdm(sentences, desc="Sentences"):
        candidates = generate_bt_candidates(s, device)
        best_text = choose_best_bt(s, candidates, embed_model, rugpt_tok, rugpt_model, device)
        bt_sentences.append(best_text)

    result_masked = " ".join(bt_sentences)
    bt_text = unmask_placeholders(result_masked, mapping)
    bt_text = clean_generated_text(bt_text)
    return bt_text
```

---

## Параметры качества (`common/config.py`)

```python
BT_SIM_MIN       = 0.80
BT_SIM_MAX       = 0.95
BT_MIN_LEN_RATIO = 0.50
BT_MAX_LEN_RATIO = 1.50
```

Кроме того, общий цикл проверяет:

- `SIM_LABEL_MIN`, `SIM_LABEL_MAX` — сходство аугментов с базой класса,
- `TARGET_PER_CLASS` — целевой размер каждого «малого» класса.

---

## Запуск

### Единичный пример из CLI

```bash
python -m augmentation.backtranslate.main \
    --train data/train.csv \
    --output-dir out/ \
    --single "Уважаемый [PERSON]! Просим согласовать проезд техники через [OBJECT]."
```

### Цикл по датасету

```bash
python -m augmentation.backtranslate.main \
    --train data/train.csv \
    --output-dir out/
```

Результаты:

- `out/train_backtranslate_partial.csv` — промежуточные результаты (лог аугментов),
- `out/train_backtranslate.csv` — итоговый датасет (оригинал + BT‑примеры).