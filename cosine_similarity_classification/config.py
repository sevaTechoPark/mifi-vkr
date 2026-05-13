BASE_MODEL_NAME: str = "ai-forever/ruRoberta-large"

TEXT_COLUMN: str = "text"
LABEL_COLUMN: str = "label"

METHOD: str = "centroid"

MAX_LENGTH: int = 512
CHUNK_SIZE: int = 448
CHUNK_OVERLAP: int = 96

POOLING: str = "mean_max"
CHUNK_AGGREGATION: str = "mean_max"

BATCH_SIZE: int = 8

DEVICE: str | None = None
MODEL_DIR: str = ""