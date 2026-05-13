# paraphrase

Модуль аугментации текстовых данных методом **перефразирования** (paraphrasing) на основе seq2seq‑модели `fyaronskiy/ruT5-large-paraphraser`.[web:50]

Идея: для каждого предложения генерируется несколько вариантов перефраза (через безопасный вызов модели c разными режимами генерации), после чего отбирается лучший кандидат по косинусному сходству с оригиналом и соотношению длин.

---

## Структура

```text
paraphrase/
├── __init__.py
├── models.py   # Загрузка и кэширование ruT5-large-paraphraser
├── phrase.py   # split_long_sentence, generate_phrase, safe_paraphrase, postprocess_paraphrase_text
├── augment.py  # generate_para_candidates, choose_best_paraphrase, paraphrase_document
├── main.py     # run_from_params + CLI: единичный пример и цикл по датасету
└── README.md
```

---

## Как это работает

### 1. Защита плейсхолдеров (`common/masks.py`)

Перед перефразированием все теги `[ORGANIZATION]`, `[PERSON]`, `[DATE_TIME]` и т.п. заменяются на `<PH_0>`, `<PH_1>`, ... Это предотвращает «творческую интерпретацию» плейсхолдеров моделью (например, `[ORGANIZATION]` → `"Органик и АТИОН"`).

```text
[PERSON] просит согласовать → <PH_0> просит согласовать
              ↓ per Т5 ↓
<PH_0> запрашивает согласование → [PERSON] запрашивает согласование
```

После генерации:

1. `<PH_i>` восстанавливаются обратно в исходные плейсхолдеры.
2. Все артефакты вида `PH_7`, `PH.24`, `PCH.`, `PS_33` удаляются.

### 2. Предобработка и фильтр формальных текстов (`common/text_utils.py`)

Перед вызовом модели:

1. `preprocess_text`:

   - нормализует цепочки `-=_*`,
   - убирает «лапшу» из точек и пробелов,
   - схлопывает пробелы.

2. `is_highly_formal`:

   - оценивает долю цифр и заглавных букв,
   - ищет маркеры реквизитов (`ОГРН`, `ИНН`, `КПП`, `БИК`, `р/с`, ...),
   - если текст слишком «реквизитный», перефраз **пропускается полностью**, возвращается исходный текст без изменений.

Это позволяет не мучить ruT5 на длинных юридических письмах с плотной «шапкой» реквизитов (где модель часто ведёт себя как неконтролируемый суммаризатор).

### 3. Безопасное разбиение по длине (`phrase.py → split_long_sentence`, `safe_paraphrase`)

`ruT5-large-paraphraser` имеет ограничение по длине входа (в коде — 300 токенов). Для надёжной работы:

- `split_long_sentence(sent, tok, device, max_tokens=120)`:
  - если предложение ≤ 120 токенов — возвращает как есть;
  - иначе:
    - пробует разбить по знакам препинания (`,`, `;`, `:`, `—`, `-`),
    - если не помогает — режет по словам, аккумулируя чанки,
    - если часть всё ещё > 120 — рекурсивно делит её аналогично.

- `generate_phrase(text, tok, model, device, **gen_kwargs)`:
  - принимает уже подготовленный кусок (после `split_long_sentence`),
  - проверяет, что длина ≤ 300 токенов (иначе кидает ошибку),
  - запускает `model.generate` с консервативными параметрами (beam-search, без sampling).

- `safe_paraphrase(text, tok, model, max_tokens, device, **gen_kwargs)`:
  - вызывает `split_long_sentence`,
  - для каждого чанка:
    - если длина ≤ 300 токенов — использует `generate_phrase`,
    - если > 300 — режет по словам фиксированным окном (по 20 слов) и применяет `generate_phrase` к каждому,
  - склеивает куски обратно.

### 4. Генерация кандидатов (`augment.py → generate_para_candidates`)

```python
GEN_MODES = [
    {"do_sample": False, "num_beams": 5},  # базовый консервативный beam-search
    {"do_sample": False, "num_beams": 8},  # чуть более "жирный" beam-search
    {"do_sample": True,  "num_beams": 1, "top_p": 0.90, "temperature": 1.0},  # мягкий sampling
]
```

Функция:

```python
def generate_para_candidates(text: str, tok, model, device: torch.device) -> list[str]:
    candidates = []
    for cfg in GEN_MODES:
        para = safe_paraphrase(
            text=text,
            tok=tok,
            model=model,
            max_tokens=120,
            device=device,
            **cfg,
        )
        para = para.strip()
        if para:
            candidates.append(para)
    return list(dict.fromkeys(candidates))  # дедуп
```

Таким образом на одно предложение генерируется несколько кандидатов:

- 2 через beam-search без сэмплинга (стабильные),
- 1 через мягкий sampling (для небольшого разнообразия).[web:50]

### 5. Отбор лучшего кандидата (`augment.py → choose_best_paraphrase`)

```python
def choose_best_paraphrase(
    source_chunk: str,
    candidates: list[str],
    embed_model: SentenceTransformer,
) -> str:
    ...
```

Алгоритм:

1. Если `candidates` пуст — возвращается исходный `source_chunk`.
2. Фильтр по длине:

   - считаем `len_ratio = len(candidate) / len(source_chunk)`,
   - оставляем только те, для которых `[PARA_MIN_LEN_RATIO, PARA_MAX_LEN_RATIO]` = `[0.70, 1.30]`,
   - если после фильтра список пуст — используем всех кандидатов.

3. Считаем эмбеддинги `[source_chunk] + filtered_candidates` через `deepvk/USER2-base`.
4. Считаем cosine similarity.
5. Ищем кандидата в окне `[PARA_SIM_MIN, PARA_SIM_MAX]` = `[0.85, 0.97]`:

   - если окно не пустое — берём кандидата с максимальным cosine внутри окна,
   - если пустое — fallback на кандидата с максимальным cosine по всем.

6. Выбранный текст пропускается через `postprocess_paraphrase_text`:

   - внутри — `clean_aug_result` (нормализация пунктуации и пробелов),
   - затем `clean_generated_text` (удаление URL, PH‑мусора и т.п.).

Функция возвращает **строку** — лучший перефраз для данного предложения.

### 6. Сборка документа (`augment.py → paraphrase_document`)

```python
def paraphrase_document(
    source_text: str,
    tok,
    model,
    embed_model: SentenceTransformer,
    device: torch.device,
) -> str:
    if is_highly_formal(source_text):
        print("текст слишком формальный пропускаем перефраз")
        return source_text

    masked_text, mapping = mask_placeholders(source_text)
    masked_text = preprocess_text(masked_text)
    sentences = [s.text for s in sentenize(masked_text)]
    paraphrased_sentences = []

    for s in tqdm(sentences, desc="Sentences"):
        s = s.strip()
        if not s:
            continue

        candidates = generate_para_candidates(s, tok, model, device)
        best_text = choose_best_paraphrase(
            source_chunk=s,
            candidates=candidates,
            embed_model=embed_model,
        )
        paraphrased_sentences.append(best_text)

    result_masked = " ".join(paraphrased_sentences)
    para_text = unmask_placeholders(result_masked, mapping)
    para_text = postprocess_paraphrase_text(para_text)
    return para_text
```

---

## Параметры качества (`common/config.py`)

```python
PARA_SIM_MIN       = 0.85
PARA_SIM_MAX       = 0.97
PARA_MIN_LEN_RATIO = 0.70
PARA_MAX_LEN_RATIO = 1.30
```

Дополнительно:

- глобальные `SIM_LABEL_MIN` / `SIM_LABEL_MAX` контролируют сходство аугментов с классом,
- `TARGET_PER_CLASS = 5` — целевой размер каждого «малого» класса.

---

## Запуск

### Единичный пример из CLI

```bash
python -m augmentation.paraphrase.main \
    --train data/train.csv \
    --output-dir out/ \
    --single "В соответствии с договором просим согласовать въезд автотранспорта на территорию объекта."
```

### Цикл по датасету

```bash
python -m augmentation.paraphrase.main \
    --train data/train.csv \
    --output-dir out/
```

Результаты:

- `out/train_paraphrase_partial.csv` — лог аугментов (можно продолжать после прерывания),
- `out/train_paraphrase.csv` — итоговый датасет (оригинал + перефразы).

Формат входного/выходного CSV такой же, как в backtranslate: поля `label,text` на входе и `label,text,source_text,cosine_sim,max_label_cosine_sim,augmentation_type` в partial‑выходе.