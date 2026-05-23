# Результаты экспериментов: метрики классификации

Все эксперименты выполнены на фиксированном train/test-разбиении (random_state=42) корпуса русскоязычных деловых писем из 36 классов. Метрики:

- **Balanced accuracy** — средняя полнота по всем классам, нечувствительна к дисбалансу.
- **Macro F1** — макроусреднённое F1, равный вклад каждого класса.

Условные обозначения:
- **BASELINE** — классический baseline на TF-IDF (word + char n-граммы) с линейным классификатором.
- **AutoModelForSequenceClassification** — стандартная BERT-голова на токене [CLS].
- **MeanPooling** — усреднение скрытых состояний всех токенов с учётом маски внимания.
- **chunkmean** — chunked-классификатор: разбиение длинного документа на перекрывающиеся фрагменты, mean pooling внутри чанка, усреднение по чанкам.
- **default** — предобученный эмбеддер `ai-forever/ruRoberta-large` без доменной адаптации.
- **custom_embedder** — тот же эмбеддер после доменного дообучения через MultipleNegativesRankingLoss.
- **bert_only / tfidf_only / hybrid** — источник признаков: только BERT-эмбеддинг, только TF-IDF, либо их конкатенация.

---

## 1. train.csv

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| **BASELINE** (TF-IDF + линейная модель) | **0.534** | **0.543** |
| rubert-base-cased AutoModelForSequenceClassification | 0.359 | 0.347 |
| rubert-base-cased MeanPooling | 0.406 | 0.408 |
| ruRoberta-large AutoModelForSequenceClassification | 0.388 | 0.377 |
| ruRoberta-large MeanPooling | 0.396 | 0.394 |
| ruRoberta-large chunkmean | 0.454 | 0.443 |

---

## 2. train_augmented.csv

### 2.1. Baseline и BERT-классификаторы

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| BASELINE (TF-IDF + линейная модель) | 0.489 | 0.486 |
| rubert-base-cased AutoModelForSequenceClassification | 0.433 | 0.431 |
| rubert-base-cased MeanPooling | 0.454 | 0.448 |
| ruRoberta-large AutoModelForSequenceClassification | 0.507 | 0.488 |
| ruRoberta-large MeanPooling | 0.479 | 0.469 |
| ruRoberta-large chunkmean | 0.501 | **0.504** |

### 2.2. Косинусные методы и гибридные классификаторы (default эмбеддер)

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| cosine_similarity centroid | 0.103 | 0.103 |
| cosine_similarity nearest | 0.197 | 0.186 |
| hybrid MLP | 0.213 | 0.180 |
| hybrid classical LinearSVC | 0.394 | 0.358 |
| hybrid classical LogReg | 0.272 | 0.232 |

### 2.3. Косинусные методы и гибридные классификаторы (custom эмбеддер)

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| cosine_similarity centroid | 0.476 | 0.470 |
| cosine_similarity nearest | 0.476 | 0.487 |

> Результаты `[custom_embedder] hybrid MLP` и `[custom_embedder] hybrid classical` в логе отсутствуют — добавить, когда будут запуски.

---

## 3. train_summarized.csv

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| **BASELINE** | **0.366** | **0.378** |
| rubert-base-cased AutoModelForSequenceClassification | 0.193 | 0.183 |
| rubert-base-cased MeanPooling | 0.242 | 0.240 |
| ruRoberta-large AutoModelForSequenceClassification | 0.345 | 0.325 |
| ruRoberta-large MeanPooling | 0.333 | 0.326 |

Замена исходного текста на суммаризацию ухудшает все модели; baseline остаётся лучшим.

---

## 4. train_original_plus_summary.csv

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| **BASELINE** | **0.482** | **0.497** |
| rubert-base-cased AutoModelForSequenceClassification | 0.363 | 0.358 |
| rubert-base-cased MeanPooling | 0.376 | 0.386 |
| ruRoberta-large AutoModelForSequenceClassification | 0.384 | 0.381 |
| ruRoberta-large MeanPooling | 0.444 | 0.435 |

Связка original+summary заметно лучше, чем только summary, но всё ещё ниже исходного train.

---

## 5. train_augmented_summarized.csv

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| BASELINE | 0.384 | 0.395 |
| rubert-base-cased AutoModelForSequenceClassification | 0.326 | 0.291 |
| rubert-base-cased MeanPooling | 0.271 | 0.271 |
| ruRoberta-large AutoModelForSequenceClassification | 0.379 | 0.383 |
| ruRoberta-large MeanPooling | **0.411** | **0.413** |

---

## 6. train_augmented_original_plus_summary.csv

| Подход | Balanced accuracy | Macro F1 |
|---|---|---|
| BASELINE | 0.467 | 0.475 |
| rubert-base-cased AutoModelForSequenceClassification | 0.420 | 0.423 |
| rubert-base-cased MeanPooling | 0.461 | 0.453 |
| ruRoberta-large AutoModelForSequenceClassification | 0.478 | 0.459 |
| ruRoberta-large MeanPooling | **0.512** | **0.498** |
| ruRoberta-large chunkmean | 0.483 | 0.473 |

Лучшая конфигурация (ruRoberta-large MeanPooling) приближается к лучшим результатам BERT на полном тексте с аугментацией.

---

## 7. Сводное сравнение baseline по всем вариантам train-набора

| Обучающие данные | Balanced accuracy | Macro F1 |
|---|---|---|
| **train** | **0.534** | **0.543** |
| train_augmented | 0.489 | 0.486 |
| train_original_plus_summary | 0.482 | 0.497 |
| train_augmented_original_plus_summary | 0.467 | 0.475 |
| train_augmented_summarized | 0.384 | 0.395 |
| train_summarized | 0.366 | 0.378 |

Для TF-IDF baseline лучшим остаётся исходный неаугментированный train; любая модификация (суммаризация, аугментация, их комбинация) снижает качество.

---

## 8. Косинусные методы — полная сводка (все эмбеддеры и train-наборы)

| Эмбеддинги | Обучающие данные | Метод | Balanced accuracy | Macro F1 |
|---|---|---|---|---|
| default | train | Centroid | 0.134 | 0.109 |
| default | train | Nearest | 0.198 | 0.187 |
| default | train | CentroidNN | 0.171 | 0.168 |
| default | train_augmented | Centroid | 0.103 | 0.103 |
| default | train_augmented | Nearest | 0.197 | 0.186 |
| default | train_augmented | CentroidNN | 0.168 | 0.170 |
| custom | train | Centroid | **0.578** | **0.508** |
| custom | train | Nearest | 0.504 | 0.504 |
| custom | train | CentroidNN | 0.503 | 0.501 |
| custom | train_augmented | Centroid | 0.476 | 0.470 |
| custom | train_augmented | Nearest | 0.476 | 0.487 |
| custom | train_augmented | CentroidNN | 0.442 | 0.456 |

На baseline-эмбеддингах все методы работают на уровне случайного. После дообучения эмбеддера качество резко возрастает, **Cosine-Centroid на `custom + train`** даёт лучшую balanced accuracy во всей работе.

---

## 9. MLP-классификатор поверх композитных признаков

### 9.1. Все запуски (по 3 на конфигурацию)

| Эмбеддинги | Обучающие данные | Запуск | Balanced accuracy | Macro F1 |
|---|---|---|---|---|
| default | train | 1 | 0.527 | 0.517 |
| default | train | 2 | 0.493 | 0.512 |
| default | train | 3 | 0.542 | 0.513 |
| default | train_augmented | 1 | 0.404 | 0.438 |
| default | train_augmented | 2 | 0.447 | 0.484 |
| default | train_augmented | 3 | 0.490 | 0.505 |
| custom | train | 1 | 0.498 | **0.543** |
| custom | train | 2 | 0.518 | 0.519 |
| custom | train | 3 | 0.518 | 0.525 |
| custom | train_augmented | 1 | 0.371 | 0.456 |
| custom | train_augmented | 2 | 0.458 | 0.469 |
| custom | train_augmented | 3 | 0.410 | 0.459 |

### 9.2. Сводная статистика (mean ± std)

| Эмбеддинги | Обучающие данные | Balanced accuracy | Macro F1 |
|---|---|---|---|
| default | train | 0.520 ± 0.025 | 0.514 ± 0.003 |
| default | train_augmented | 0.447 ± 0.043 | 0.476 ± 0.035 |
| custom | train | 0.511 ± 0.011 | **0.529 ± 0.012** |
| custom | train_augmented | 0.413 ± 0.044 | 0.461 ± 0.007 |

---

## 10. Линейные модели на bert_only / tfidf_only / hybrid

Лучшие варианты по macro F1 при подборе регуляризации, с balanced-взвешиванием классов.

| Эмбеддинги | Обучающие данные | Источник признаков | Метод | Balanced accuracy | Macro F1 |
|---|---|---|---|---|---|
| default | train | bert_only | LinearSVC | 0.333 | 0.317 |
| default | train | bert_only | LogisticRegression | 0.318 | 0.292 |
| default | train | bert_only | RidgeClassifier | 0.361 | 0.288 |
| default | train | tfidf_only | LinearSVC | 0.529 | 0.521 |
| default | train | tfidf_only | LogisticRegression | 0.513 | 0.467 |
| default | train | hybrid | LogisticRegression | 0.442 | 0.451 |
| default | train | hybrid | RidgeClassifier | **0.559** | **0.532** |
| default | train_augmented | bert_only | LinearSVC | 0.403 | 0.373 |
| default | train_augmented | tfidf_only | LogisticRegression | 0.458 | 0.430 |
| default | train_augmented | hybrid | RidgeClassifier | 0.514 | 0.486 |
| custom | train | bert_only | LinearSVC | 0.540 | 0.528 |
| custom | train | bert_only | LogisticRegression | 0.534 | 0.510 |
| custom | train | bert_only | RidgeClassifier | 0.531 | 0.523 |
| custom | train | tfidf_only | RidgeClassifier | 0.521 | 0.508 |
| custom | train | hybrid | LogisticRegression | 0.538 | 0.529 |
| custom | train | hybrid | LinearSVC | 0.526 | **0.541** |
| custom | train_augmented | bert_only | LinearSVC | 0.442 | 0.462 |
| custom | train_augmented | hybrid | LinearSVC | 0.446 | 0.463 |

---

## 11. Сводная таблица лидеров по balanced accuracy

| Подход | Эмбеддинги | Данные | Источник | Метод | Balanced accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| Косинусные | custom | train | bert_only | **Cosine-Centroid** | **0.578** | 0.508 |
| Классические | custom | train | bert_only | LogisticRegression | 0.573 | 0.490 |
| Классические | default | train | tfidf_only | LogisticRegression | 0.566 | 0.424 |
| Классические | custom | train | tfidf_only | LogisticRegression | 0.566 | 0.424 |
| Классические | default | train | tfidf_only | LinearSVC | 0.561 | 0.497 |

---

## 12. Сводная таблица лидеров по macro F1

| Подход | Эмбеддинги | Данные | Источник | Метод | Balanced accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| MLP | custom | train | hybrid | **MLP-StrongMLP** | 0.498 | **0.543** |
| Классические | custom | train | hybrid | LinearSVC | 0.526 | 0.541 |
| Классические | default | train | hybrid | RidgeClassifier | 0.559 | 0.532 |
| Классические | custom | train | hybrid | LogisticRegression | 0.538 | 0.529 |
| Классические | custom | train | bert_only | LinearSVC | 0.540 | 0.528 |

---

## 13. Итоги по результатам экспериментов

- **Абсолютный максимум по macro F1: 0.543** — MLP на `custom + train + hybrid`. Практически не уступает **LinearSVC = 0.541** на той же комбинации, который проще, быстрее и стабильнее.
- **Абсолютный максимум по balanced accuracy: 0.578** — Cosine-Centroid на `custom + train + bert_only`.
- **TF-IDF baseline на n-граммах** формирует сильную базовую линию (0.534/0.543 на train) и остаётся конкурентоспособным относительно нейросетевых подходов на любом отдельно взятом dataset-файле.
- Среди классификационных голов BERT **MeanPooling и chunkmean устойчиво опережают стандартную голову на токене CLS**; chunkmean на `train_augmented` даёт лучший macro F1 среди end-to-end BERT-конфигураций (0.504).
- **BERT-эмбеддинги без доменной адаптации** проигрывают TF-IDF; после дообучения эмбеддера (custom) выходят на сопоставимый уровень (прирост порядка +0.21–0.24 по macro F1).
- **Композитное (hybrid) представление + custom-эмбеддер** даёт наиболее сбалансированный результат по precision и recall.
- **Аугментация** улучшает BERT-классификаторы (особенно стандартную голову CLS), но размывает границы классов в сильных представлениях (custom + hybrid) и ухудшает TF-IDF baseline.
- **Суммаризация** как замена исходного текста снижает качество всех моделей. Связка original+summary с аугментацией приближается к лучшим конфигурациям на полном тексте.
- Среди линейных моделей **LinearSVC** и **RidgeClassifier** чаще лидируют, **LogisticRegression** заметно уступает в ключевых конфигурациях.
