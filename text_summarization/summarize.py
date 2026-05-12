import torch
import pandas as pd
from tqdm import tqdm
from .config import MAX_INPUT_LENGTH, MAX_SUMMARY_LENGTH, MIN_SUMMARY_LENGTH, BATCH_SIZE, TEXT_COLUMN

def summarize_batch(texts, tokenizer, model, device,
                    max_input_length=MAX_INPUT_LENGTH,
                    max_summary_length=MAX_SUMMARY_LENGTH,
                    min_summary_length=MIN_SUMMARY_LENGTH) -> list[str]:
    inputs = tokenizer(texts, max_length=max_input_length, truncation=True,
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_new_tokens=max_summary_length,
            min_new_tokens=min_summary_length,
            num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
        )
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

def summarize_dataframe(df, tokenizer, model, device,
                        text_column=TEXT_COLUMN, batch_size=BATCH_SIZE) -> list[str]:
    # summaries[i] строго соответствует df.iloc[i]
    all_summaries = []
    texts = df[text_column].tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="Суммаризация"):
        all_summaries.extend(summarize_batch(texts[i:i+batch_size], tokenizer, model, device))
    return all_summaries