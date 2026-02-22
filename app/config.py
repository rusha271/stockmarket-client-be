"""Application configuration."""
from pathlib import Path

from pydantic_settings import BaseSettings

# Default: app/models folder (all model files stay inside app)
_APP_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _APP_DIR / "models"


class Settings(BaseSettings):
    """App settings (env-loaded)."""

    app_name: str = "AI Stock Prediction API"
    debug: bool = False
    # Model paths: default is app/models/; overridable via env MODEL_DIR, CLF_PATH, REG_PATH
    model_dir: Path | None = None
    clf_path: Path | None = None  # if set, overrides model_dir for classifier
    reg_path: Path | None = None  # if set, overrides model_dir for regression

    @property
    def classifier_model_path(self) -> Path:
        base = self.model_dir if self.model_dir is not None else _MODELS_DIR
        return self.clf_path or (base / "pretrained_model_clf_5min.json")

    @property
    def regression_model_path(self) -> Path:
        base = self.model_dir if self.model_dir is not None else _MODELS_DIR
        return self.reg_path or (base / "pretrained_model_reg_5min.json")

    # Database (MySQL) – set DATABASE_URL or individual vars
    database_url: str | None = None
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "be_stock"

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"mysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
