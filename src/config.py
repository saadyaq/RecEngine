from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

    # Data
    DATASET_CATEGORY: str = "Electronics"
    MIN_INTERACTIONS: str = 5
    TEST_RATION: str = 0.2

    # Chunk
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CODE_CHUNK_SIZE: int = 1000

    # TOKENIZER
    TOKENIZER_MODEL: str = "cl100k_base"

    class config:
        env_file = ".env"


settings = Settings()
