from .config import *
from .model import load_summarization_model
from .summarize import summarize_dataframe
from .data import load_dataset, save_dataset, build_summarized_dataset, build_combined_dataset, derive_output_paths

def run(input_path: str, output_dir: str, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataset(input_path)
    df = df.reset_index(drop=True)
    df["sample_id"] = df.index          # ключ для безопасного мержа

    tokenizer, model, device = load_summarization_model(model_name)
    summaries = summarize_dataframe(df, tokenizer, model, device, ...)

    df_summarized = build_summarized_dataset(df, summaries, ...)
    df_combined   = build_combined_dataset(df, summaries, ...)

    summarized_path, combined_path = derive_output_paths(input_path, output_dir)
    save_dataset(df_summarized, summarized_path)
    save_dataset(df_combined, combined_path)
    return df_summarized, df_combined