from huggingface_hub import snapshot_download

# Можно добавить несколько моделей списком
MODELS = [
    ("DeepPavlov/rubert-base-cased", "rubert-base-cased"),
]

BASE_DIR = "./models"

def download(repo_id: str, local_name: str, revision: str | None = None):
    local_dir = f"{BASE_DIR}/{local_name}"
    snapshot_download(
        repo_id=repo_id,
        revision="COMMIT_HASH",         # можно зафиксировать commit hash для воспроизводимости
        local_dir=local_dir,
        local_dir_use_symlinks=False,   # реально копировать файлы в папку
    )
    print(f"OK: {repo_id} -> {local_dir}")

def main():
    for repo_id, local_name in MODELS:
        download(repo_id, local_name)

if __name__ == "__main__":
    main()