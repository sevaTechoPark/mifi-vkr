augmentation
├── common
│   ├── __init__.py
│   ├── config.py          # константы SIM_MIN, TARGET_PER_CLASS и т.д.
│   ├── masks.py           # mask_placeholders / unmask_placeholders
│   ├── embeddings.py      # load_embed_model, cos_sim
│   ├── perplexity.py      # load_rugpt, rugpt_perplexity_list
│   └── augment_loop.py    # run_augmentation_loop — общий цикл
├── backtranslate
│   ├── __init__.py
│   ├── models.py          # загрузка MarianMT с кешированием
│   ├── translate.py       # safe_translate, split_long_sentence, preprocess
│   ├── augment.py         # generate_bt_candidates, choose_best_bt, back_translate_document
│   └── main.py            # --train / --output-dir / --single
└── paraphrase
    ├── __init__.py
    ├── models.py          # загрузка ruT5 с кешированием
    ├── augment.py         # generate_paraphrase, generate_best_paraphrase, paraphrase_document
    └── main.py            # --train / --output-dir / --single