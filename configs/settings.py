from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Model
    model_path: str = Field(default="weights/best.pt", alias="MODEL_PATH")
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    input_size: int = 640
    device: str = Field(default="cpu", alias="DEVICE")  # "cpu" | "cuda" | "cuda:0"

    helmet_class_id: int = 0        # 'With Helmet'    → an toàn
    no_helmet_class_id: int = 1     # 'Without Helmet' → vi phạm


    # Tracker
    tracker_max_age: int = 30       # frames to keep a lost track alive
    tracker_n_init: int = 3         # frames before confirming a new track

    # Violation logic
    head_region_ratio: float = 0.30         # top 30% of person bbox = head
    helmet_iou_threshold: float = 0.15      # min IoU(head, helmet) to count as "wearing"
    violation_confirm_frames: int = 5      # consecutive frames to confirm violation

    # Output
    output_dir: str = Field(default="output", alias="OUTPUT_DIR")
    save_snapshots: bool = True

    # API
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_upload_mb: int = 50


settings = Settings()
