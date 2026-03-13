import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

PROMPT = "привет, мир, ты работаешь?"
MAX_NEW_TOKENS = 64


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_any(model_dir: Path):
    """
    Универсальная загрузка:
    1) пробуем как CausalLM (LLM)
    2) если не вышло — пробуем как Seq2SeqLM (text2text/перевод)
    """
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )

    common = dict(
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_dir, **common)
        mdl.eval()
        return tok, mdl, "causal"
    except Exception:
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir, **common)
        mdl.eval()
        return tok, mdl, "seq2seq"


@torch.no_grad()
def run_model(local_name: str, model_dir: Path, prompt: str = PROMPT):
    tok, mdl, kind = load_any(model_dir)

    device = get_device()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)

    # если model загружен через device_map="auto", у него может не быть .device
    if device == "cuda" and not hasattr(mdl, "device"):
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    if kind == "seq2seq":
        gen_kwargs["num_beams"] = 4
    else:
        gen_kwargs["num_beams"] = 1

    t0 = time.perf_counter()
    out = mdl.generate(**inputs, **gen_kwargs)
    dt = time.perf_counter() - t0

    text = tok.decode(out[0], skip_special_tokens=True)
    return {"model": local_name, "kind": kind, "seconds": dt, "out": text}


def test_one(local_name: str, models_dir: Path):
    model_dir = models_dir / local_name
    if not model_dir.exists():
        print(f"[SKIP] {local_name}: dir not found: {model_dir}")
        return

    print(f"\n=== {local_name} ===")
    try:
        res = run_model(local_name, model_dir)
        print(f"Kind: {res['kind']}")
        print(f"Time: {res['seconds']:.3f}s")
        print("Output:\n", res["out"])
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[ERROR] {local_name}: {type(e).__name__}: {e}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"

    MODELS = [
        "qwen3_5_0_8b",
        "qwen3_5_2b",
        "qwen3_5_9b",  # GGUF: transformers не поддержит, будет ошибка/скип ниже
        "claude-opus_27b",
        "aparecium-seq2seq-reverser_3b",
    ]

    print("Device:", get_device())
    print("Prompt:", PROMPT)

    for local_name in MODELS:
        if "gguf" in local_name.lower() or local_name.endswith("_9b"):
            print(f"\n=== {local_name} ===")
            print("[SKIP] GGUF model: not supported by transformers (use llama.cpp).")
            continue
        test_one(local_name, models_dir)


if __name__ == "__main__":
    main()
