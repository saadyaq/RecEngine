from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_RAW_DIR: Path = Path(__file__).parent.parent / "data" / "raw"
    DATA_PROCESSED_DIR: Path = Path(__file__).parent.parent / "data" / "processed"

    # Data
    DATASET_CATEGORY: str = "Electronics"
    MIN_INTERACTIONS: int = 10
    TEST_RATIO: float = 0.2

    # Dask chunked processing
    USE_DASK: bool = False
    DASK_CHUNK_SIZE: int = 500_000

    # Chunk
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CODE_CHUNK_SIZE: int = 1000

    # Tokenizer
    TOKENIZER_MODEL: str = "cl100k_base"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"


settings = Settings()
