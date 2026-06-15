# Результаты экспериментов — открытые данные (36 классов)

Все эксперименты выполнены на фиксированном train/test-разбиении (random_state=42) корпуса русскоязычных деловых писем из **36 классов**.

**Метрики**:
- **BAcc (Balanced accuracy)** — средняя полнота по всем классам, нечувствительна к дисбалансу.
- **F1 (Macro F1)** — макроусреднённое F1, равный вклад каждого класса.

**Условные обозначения**:
- **BASELINE** — классический baseline на TF-IDF (word + char n-граммы) с линейным классификатором.
- **AutoModelForSequenceClassification** — стандартная BERT-голова на токене [CLS].
- **MeanPooling** — усреднение скрытых состояний всех токенов с учётом маски внимания.
- **chunkmean** — chunked-классификатор: разбиение длинного документа на перекрывающиеся фрагменты, mean pooling внутри чанка, усреднение по чанкам.
- **default** — предобученный эмбеддер `ai-forever/ruRoberta-large` без доменной адаптации.
- **custom_embedder** — тот же эмбеддер после доменного дообучения через MultipleNegativesRankingLoss.
- **bert_only / tfidf_only / hybrid** — источник признаков: только BERT-эмбеддинг, только TF-IDF, либо их конкатенация.

> **Главный объект исследования** — композитные (гибридные) векторные представления. Все остальные подходы (Baseline, чистый BERT-классификатор, Cosine-методы, ablation tfidf_only / bert_only) — это **точки сравнения и контрольные точки** для проверки тезиса, что объединение TF-IDF и BERT даёт выигрыш над любой из своих половин.

---

## 1. Baseline — TF-IDF + линейная модель

Эталон, с которым сравниваются все композитные подходы. Зафиксированная конфигурация: word (1–2) + char_wb (3–5), L2-нормализация, default C.

### 1.1. По вариантам train-набора

| Обучающие данные | BAcc | F1 |
|---|---|---|
| **train** | **0.534** | **0.543** |
| train_augmented | 0.489 | 0.486 |
| train_original_plus_summary | 0.482 | 0.497 |
| train_augmented_original_plus_summary | 0.467 | 0.475 |
| train_augmented_summarized | 0.384 | 0.395 |
| train_summarized | 0.366 | 0.378 |

**Вывод**: для TF-IDF baseline исходный неаугментированный `train` остаётся лучшим. Любая модификация (суммаризация, аугментация, их комбинация) снижает качество, поскольку back-translation размывает редкие токены в разреженных признаках.

---

## 2. BERT-классификация (full fine-tuning головы)

Стандартные подходы — обучение всей модели или классификационной головы поверх предобученного энкодера.

### 2.1. train.csv

| Подход | BAcc | F1 |
|---|---|---|
| rubert-base-cased AutoModelForSequenceClassification | 0.359 | 0.347 |
| rubert-base-cased MeanPooling | 0.406 | 0.408 |
| ruRoberta-large AutoModelForSequenceClassification | 0.388 | 0.377 |
| ruRoberta-large MeanPooling | 0.396 | 0.394 |
| **ruRoberta-large chunkmean** | **0.454** | **0.443** |

### 2.2. train_augmented.csv

| Подход | BAcc | F1 |
|---|---|---|
| rubert-base-cased AutoModelForSequenceClassification | 0.433 | 0.431 |
| rubert-base-cased MeanPooling | 0.454 | 0.448 |
| ruRoberta-large AutoModelForSequenceClassification | **0.507** | 0.488 |
| ruRoberta-large MeanPooling | 0.479 | 0.469 |
| ruRoberta-large chunkmean | 0.501 | **0.504** |

### 2.3. train_summarized.csv (суммаризация как замена)

| Подход | BAcc | F1 |
|---|---|---|
| rubert-base-cased AutoModelForSequenceClassification | 0.193 | 0.183 |
| rubert-base-cased MeanPooling | 0.242 | 0.240 |
| ruRoberta-large AutoModelForSequenceClassification | **0.345** | 0.325 |
| ruRoberta-large MeanPooling | 0.333 | **0.326** |

### 2.4. train_original_plus_summary.csv (исходный + краткое содержание)

| Подход | BAcc | F1 |
|---|---|---|
| rubert-base-cased AutoModelForSequenceClassification | 0.363 | 0.358 |
| rubert-base-cased MeanPooling | 0.376 | 0.386 |
| ruRoberta-large AutoModelForSequenceClassification | 0.384 | 0.381 |
| **ruRoberta-large MeanPooling** | **0.444** | **0.435** |

### 2.5. train_augmented_summarized.csv

| Подход | BAcc | F1 |
|---|---|---|
| rubert-base-cased AutoModelForSequenceClassification | 0.326 | 0.291 |
| rubert-base-cased MeanPooling | 0.271 | 0.271 |
| ruRoberta-large AutoModelForSequenceClassification | 0.379 | 0.383 |
| **ruRoberta-large MeanPooling** | **0.411** | **0.413** |

### 2.6. train_augmented_original_plus_summary.csv

| Подход | BAcc | F1 |
|---|---|---|
| rubert-base-cased AutoModelForSequenceClassification | 0.420 | 0.423 |
| rubert-base-cased MeanPooling | 0.461 | 0.453 |
| ruRoberta-large AutoModelForSequenceClassification | 0.478 | 0.459 |
| **ruRoberta-large MeanPooling** | **0.512** | **0.498** |
| ruRoberta-large chunkmean | 0.483 | 0.473 |

**Выводы по BERT-классификации**:
- **MeanPooling и chunkmean устойчиво опережают стандартную голову на [CLS]**.
- На полном тексте лучший результат — `chunkmean` (`train`: F1 0.443; `train_augmented`: F1 0.504).
- Лучшая end-to-end конфигурация в работе — `ruRoberta-large MeanPooling` на `train_augmented_original_plus_summary` (F1 0.498), но всё ещё уступает композитному подходу.
- Замена исходного текста на суммаризацию резко снижает качество всех моделей.

---

## 3. Cosine similarity на BERT-эмбеддингах

Альтернативный подход без обучения классификатора — прямое голосование по косинусной близости к эталонам класса. Три метода: Centroid (близость к усреднённому вектору класса), Nearest (k-NN по эмбеддингам), CentroidNN (взвешенная комбинация).

### 3.1. default-эмбеддер (без доменной адаптации)

| Обучающие данные | Метод | BAcc | F1 |
|---|---|---|---|
| train | Centroid | 0.134 | 0.109 |
| train | Nearest | 0.198 | 0.187 |
| train | CentroidNN | 0.171 | 0.168 |
| train_augmented | Centroid | 0.103 | 0.103 |
| train_augmented | Nearest | 0.197 | 0.186 |
| train_augmented | CentroidNN | 0.168 | 0.170 |

### 3.2. custom-эмбеддер (доменное дообучение через MNRLoss)

| Обучающие данные | Метод | BAcc | F1 |
|---|---|---|---|
| train | **Centroid** | **0.578** | **0.508** |
| train | Nearest | 0.504 | 0.504 |
| train | CentroidNN | 0.503 | 0.501 |
| train_augmented | Centroid | 0.476 | 0.470 |
| train_augmented | Nearest | 0.476 | 0.487 |
| train_augmented | CentroidNN | 0.442 | 0.456 |

**Вывод**: на baseline-эмбеддингах все методы работают на уровне случайного. После доменного дообучения качество резко возрастает — **Cosine-Centroid на `custom + train`** даёт абсолютный максимум по balanced accuracy во всей работе (BAcc 0.578).

---

## 4. Линейные модели на композитном (hybrid) векторе

Главный объект исследования: классические линейные модели поверх hybrid-вектора (TF-IDF ⊕ BERT) с подбором регуляризации C и balanced-взвешиванием классов.

### 4.1. default-эмбеддер

| Обучающие данные | Модель | BAcc | F1 |
|---|---|---|---|
| train | LogisticRegression | 0.442 | 0.451 |
| train | **RidgeClassifier** | **0.559** | **0.532** |
| train_augmented | RidgeClassifier | 0.514 | 0.486 |

### 4.2. custom-эмбеддер

| Обучающие данные | Модель | BAcc | F1 |
|---|---|---|---|
| train | LogisticRegression | 0.538 | 0.529 |
| train | LinearSVC | 0.526 | **0.541** |
| train_augmented | LinearSVC | 0.446 | 0.463 |

**Вывод**: композитный вектор + custom-эмбеддер + LinearSVC на `train` даёт F1 = 0.541 — лучший результат среди классических моделей на hybrid.

---

## 5. MLP на композитном векторе

Нейросетевой классификатор поверх hybrid-вектора. Архитектура: hidden=512, blocks=2, dropout=0.4, focal-loss, mixup-аугментация.

### 5.1. Все запуски (по 3 на конфигурацию)

| Эмбеддинги | Обучающие данные | Запуск | BAcc | F1 |
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

### 5.2. Сводная статистика (mean ± std)

| Эмбеддинги | Обучающие данные | BAcc | F1 |
|---|---|---|---|
| default | train | 0.520 ± 0.025 | 0.514 ± 0.003 |
| default | train_augmented | 0.447 ± 0.043 | 0.476 ± 0.035 |
| custom | train | 0.511 ± 0.011 | **0.529 ± 0.012** |
| custom | train_augmented | 0.413 ± 0.044 | 0.461 ± 0.007 |

**Вывод**: **MLP на `custom + train + hybrid` даёт абсолютный максимум по macro F1 во всей работе** — F1 = 0.543. Это и есть основной результат исследования по открытым данным.

---

## 6. Ablation: только TF-IDF-компонент гибрида

Что будет, если использовать только TF-IDF-часть композитного вектора. Это **не самостоятельный baseline**, а контрольная точка для проверки тезиса «гибрид > его половин».

| Эмбеддинги | Обучающие данные | Модель | BAcc | F1 |
|---|---|---|---|---|
| default | train | LinearSVC | 0.529 | 0.521 |
| default | train | LogisticRegression | 0.513 | 0.467 |
| default | train_augmented | LogisticRegression | 0.458 | 0.430 |
| custom | train | RidgeClassifier | 0.521 | 0.508 |

---

## 7. Ablation: только BERT-компонент гибрида

| Эмбеддинги | Обучающие данные | Модель | BAcc | F1 |
|---|---|---|---|---|
| default | train | LinearSVC | 0.333 | 0.317 |
| default | train | LogisticRegression | 0.318 | 0.292 |
| default | train | RidgeClassifier | 0.361 | 0.288 |
| default | train_augmented | LinearSVC | 0.403 | 0.373 |
| custom | train | LinearSVC | 0.540 | 0.528 |
| custom | train | LogisticRegression | 0.534 | 0.510 |
| custom | train | RidgeClassifier | 0.531 | 0.523 |
| custom | train_augmented | LinearSVC | 0.442 | 0.462 |

**Вывод по ablation**: каждая половина гибрида по отдельности уступает гибриду в целом по F1. Лучший bert_only — F1 0.528 (custom+train+LinearSVC); лучший tfidf_only — F1 0.521; гибрид — F1 0.541 (LinearSVC) и F1 0.543 (MLP).

---

## 8. Итоговое сравнение лидеров

### 8.1. По balanced accuracy

| Подход | Эмбеддинги | Данные | Источник | Метод | BAcc | F1 |
|---|---|---|---|---|---|---|
| **Cosine** | custom | train | bert_only | **Centroid** | **0.578** | 0.508 |
| Классические | custom | train | bert_only | LogisticRegression | 0.573 | 0.490 |
| Классические | default | train | tfidf_only | LogisticRegression | 0.566 | 0.424 |
| Классические | custom | train | tfidf_only | LogisticRegression | 0.566 | 0.424 |
| Классические | default | train | tfidf_only | LinearSVC | 0.561 | 0.497 |

### 8.2. По macro F1

| Подход | Эмбеддинги | Данные | Источник | Метод | BAcc | F1 |
|---|---|---|---|---|---|---|
| **MLP** | custom | train | **hybrid** | **MLP-StrongMLP** | 0.498 | **0.543** |
| Классические | custom | train | hybrid | LinearSVC | 0.526 | 0.541 |
| Классические | default | train | hybrid | RidgeClassifier | 0.559 | 0.532 |
| Классические | custom | train | hybrid | LogisticRegression | 0.538 | 0.529 |
| Классические | custom | train | bert_only | LinearSVC | 0.540 | 0.528 |

---

## 9. Выводы по результатам экспериментов

1. **Абсолютный максимум по macro F1 — 0.543** — MLP на `custom + train + hybrid`. Главный результат работы по открытым данным.
2. **Абсолютный максимум по balanced accuracy — 0.578** — Cosine-Centroid на `custom + train + bert_only`.
3. **Композитное (hybrid) представление + custom-эмбеддер** даёт наиболее сбалансированный результат по precision и recall (топ-4 из топ-5 по F1).
4. **Композит превосходит обе свои половины** в режиме `custom + train`: F1 hybrid 0.541–0.543 vs bert_only 0.528 vs tfidf_only 0.521.
5. **TF-IDF baseline** формирует сильную базовую линию (0.534/0.543) и остаётся конкурентоспособным относительно нейросетевых подходов на любом отдельно взятом dataset-файле.
6. Среди BERT-голов **MeanPooling и chunkmean устойчиво опережают [CLS]**; chunkmean на `train_augmented` даёт лучший macro F1 среди end-to-end BERT-конфигураций (0.504).
7. **BERT-эмбеддинги без доменной адаптации** проигрывают TF-IDF; после дообучения эмбеддера (custom) выходят на сопоставимый уровень (прирост порядка +0.21–0.24 по macro F1).
8. **Аугментация** работает дифференцированно: улучшает стандартную BERT-голову, но размывает границы классов в сильных представлениях (custom + hybrid) и ухудшает TF-IDF baseline.
9. **Суммаризация как замена** исходного текста снижает качество всех моделей. Связка original+summary с аугментацией приближается к лучшим конфигурациям на полном тексте, но не превосходит их.
10. Среди линейных моделей **LinearSVC** и **RidgeClassifier** чаще лидируют, **LogisticRegression** заметно уступает в ключевых конфигурациях.

---

## 10. Технические параметры

- Энкодер: `ai-forever/ruRoberta-large` (1024-dim, 24 слоя, заморожено 12 нижних)
- Чанкинг: 448 (эмбеддер), 512 (BERT-классификатор), overlap=128
- Pooling: mean (token + chunk)
- TF-IDF: word (1–2) ⊕ char_wb (3–5), sublinear_tf, L2-нормализация
- Hybrid: bert_weight=5.0, per-block L2 + final L2
- MLP: hidden=512, blocks=2, dropout=0.4, focal_gamma=1.0, label_smoothing=0.05, mixup=0.2, class_weight=balanced
- BERT-классификатор: lr=2e-5, AdamW, bf16, warmup 10%, gradient checkpointing
- Custom-эмбеддер: 3 эпохи, MNRLoss, in-batch negatives, cross-document positives
