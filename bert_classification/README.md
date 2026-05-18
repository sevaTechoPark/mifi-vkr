# bert_classification

Chunk-aware классификатор длинных русских писем поверх `ruRoberta-large`.
В проекте называется **`ruRoberta-large-chunkmean`** — это его рабочее имя
в логах и отчётах. От пяти baseline-ноутбуков в `notebooks/bert/` отличается
тем, что **не обрезает длинные документы до 512 токенов**, а режет их на
несколько чанков и собирает один документный вектор иерархическим mean-pool.

## Архитектура
```
input_text → tokenizer (RoBERTa BPE)
→ split into N chunks (max_length=512, stride=128, max_chunks=4)
→ RoBERTa encoder (per chunk) [B*N, T, H]
→ token mean-pool (per chunk) [B*N, H]
→ reshape [B, N, H]
→ chunk mean-pool (по реальному num_chunks) [B, H]
→ LayerNorm → Dropout → Linear(num_labels)
→ CrossEntropyLoss(weight=class_weights, label_smoothing)
```

Двухуровневый mean-pool — это компромисс между скоростью и качеством:

- Иерархическая агрегация даёт документу один плотный вектор той же
  размерности, что и токеновые эмбеддинги (1024 у large).
- Mean-pool — стабильнее max-pool на длинных хвостах, не вытаскивает
  выбросы.
- Альтернативы (attention-pool, кастомный CLS на чанках) на small-data
  обычно переобучаются; mean — sane default.

## Что лежит в модуле

| Файл | Назначение |
| --- | --- |
| `config.py` | `ModelConfig`, `TrainConfig`, `DataConfig`, `PathConfig` |
| `data.py` | Загрузка CSV, label-mappings, class-weights, токенизация в чанки, `DatasetDict` |
| `model.py` | `ChunkMeanPoolRobertaClassifier`, `ChunkDataCollator`, `build_model()` |
| `metrics.py` | `compute_metrics` — `balanced_accuracy`, `f1_macro` |
| `training.py` | Сборка Trainer, кастомный LR-split (энкодер/голова), early stop, rolling resume |
| `utils.py` | Seed, очистка памяти, безопасная загрузка state_dict |
| `main.py` | CLI + `run_from_params(...)` для использования из ноутбука |
| `inference.py` | Загрузка обученной модели и предикт |

## Раздельный learning rate (encoder vs head)

В `WeightedChunkTrainer.create_optimizer` параметры модели разбиты на 4
группы: `(roberta) × (decay/no_decay) ⊕ (head) × (decay/no_decay)`.
Это позволяет учить голову и энкодер с разными LR. Сейчас оба `1e-5`,
но если энкодер начинает забывать pretraining — снижай только
`lr_encoder` (например, `5e-6`), не трогая `lr_head`.

## Чекпоинты на диске

| Файл | Когда пишется | Содержит |
| --- | --- | --- |
| `resume_checkpoint.pt` | После каждой эпохи (перезаписывается) | model_state_dict (без `class_weights`) + optimizer + scheduler + epoch |
| `metrics.json` | Один раз в конце | best epoch, best metrics, configs |

**Принципиально:** нет ни `pytorch_model.bin`, ни per-epoch-копий, ни
`train_history.json`. Лучшие веса хранятся в RAM через
`BestMetricInMemoryCallback` и явно загружаются в `trainer.model`
перед финальным `evaluate()`.

Почему `class_weights` исключается из state_dict: это PyTorch-buffer,
завязанный на распределение классов в **train**. Хранить его в чекпоинте
бессмысленно — он пересчитается при следующем запуске. При загрузке
обратно — `strict=False` (см. `utils.load_state_dict_into_model`).

## Anti-OOM настройки для Colab L4

- `max_chunks=4`, `stride=128` — меньше токенов на батч, чем у предыдущей
  версии (было `max_chunks=6, stride=256` — это давало OOM на эвалюации
  и kernel умирал «тихо»).
- `gradient_checkpointing=True` — включается через `TrainingArguments`,
  а не в `__init__` модели (так корректно работает с recent HF Trainer).
- `bf16` авто-включается на A100 / Ampere+, иначе fallback на `fp16`.
- `dataloader_num_workers=2`, `pin_memory=True`.

## Использование

### CLI

```bash
python -m bert_classification.main \
    --train-file /content/data/train.csv \
    --test-file  /content/data/test.csv \
    --output-dir /content/out_bertcls
```

### Из ноутбука

```python
from bert_classification.main import run_from_params

components, eval_metrics = run_from_params(
    train_file="data/train.csv",
    test_file="data/test.csv",
    output_dir="/content/drive/MyDrive/.../bert_classification_best",
    # любые поля ModelConfig/TrainConfig/DataConfig переопределяются здесь
    num_epochs=15,
    early_stopping_patience=3,
)
```

### Resume после прерывания

```python
import torch
from bert_classification.main import run_from_params

# ... сначала собираешь те же конфиги и trainer, затем:
ckpt = torch.load("out_bertcls/resume_checkpoint.pt", map_location="cpu")
trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
trainer.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
trainer.train(resume_from_checkpoint=False)
```

## Ожидаемые результаты на тебе

- На `train.csv`: `balanced_accuracy ≈ 0.45+`, `f1_macro ≈ 0.44+`.
- На `train_augmented.csv`: цель `0.55+ / 0.55+`.
- Если упирается в потолок — попробуй `max_chunks=6, stride=128` (если
  железо позволяет) и подними `num_epochs` до 20.