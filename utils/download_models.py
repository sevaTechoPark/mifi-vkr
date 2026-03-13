from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    # https://huggingface.co/Qwen/Qwen3.5-0.8B
    ("Qwen/Qwen3.5-0.8B", "qwen3_5_0_8b", None),
    # https://huggingface.co/Qwen/Qwen3.5-2B
    ("Qwen/Qwen3.5-2B", "qwen3_5_2b", None),
    # https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
    ("unsloth/Qwen3.5-9B-GGUF", "qwen3_5_9b", None),
    # https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
    ("Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled", "claude-opus_27b", None),
    # https://huggingface.co/AlekseyCalvin/Lyrical_ru2_en_NanBeige_3B
    ("AlekseyCalvin/Lyrical_ru2_en_NanBeige_3B", "lyrical_ru_2_en_3b", None),

    # не полноценные HF-модели для AutoModel*:
    # https://huggingface.co/SentiChain/aparecium-seq2seq-reverser
    # https://huggingface.co/xummer/deepseek-r1-8b-belebele-lora-rus-cyrl
]

def is_model_downloaded(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    has_config = (model_dir / "config.json").exists()
    has_weights = any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").exists()
    return has_config and has_weights

def download(repo_id: str, local_name: str, revision: str | None):
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    model_dir = models_dir / local_name

    if is_model_downloaded(model_dir):
        print(f"SKIP (already downloaded): {repo_id} -> {model_dir}")
        return

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    print(f"OK: {repo_id} -> {model_dir}")

def main():
    for repo_id, local_name, revision in MODELS:
        download(repo_id, local_name, revision)

if __name__ == "__main__":
    main()