
from pathlib import Path
from pydantic import BaseModel,Field, model_validator
from pydantic_settings import  BaseSettings, SettingsConfigDict
from svoc.constants import SUPERVISED_MODELS_FILENAME
from svoc.supervised.enums import SupervisedModel
from typing import Optional
import warnings

class DataColumns(BaseModel):
    ID: str = "ID"
    OUTLET_NAME: str = "OUTLET_NAME"
    ADDRESS: str = "ADDRESS"
    POSTCODE: str = "POSTCODE"

class Settings(BaseSettings):

    DATA_DIR: Path = Path(".")
    INPUT_DATA_FILENAME: str =  ""
    BENCHMARK_DATA_FILENAME: str =  ""
    @property
    def INPUT_FILEPATH(self) -> Path:
        return self.DATA_DIR / self.INPUT_DATA_FILENAME
    @property
    def BENCHMARK_FILEPATH(self) -> Path:
        return self.DATA_DIR / self.BENCHMARK_DATA_FILENAME

    INPUT_DATATABLE: str =  ""
    BENCHMARK_DATATABLE: str = ""


    INPUT_COLUMNS: DataColumns = Field(default_factory=DataColumns)
    BENCHMARK_COLUMNS: DataColumns = Field(default_factory=DataColumns)
    @property
    def INPUT_COLUMNS_DICT(self) -> dict[str, str]:
        return self.INPUT_COLUMNS.model_dump()
    @property
    def BENCHMARK_COLUMNS_DICT(self) -> dict[str, str]:
        return self.BENCHMARK_COLUMNS.model_dump()

    MODELS_DIR: Path = Path("./models")
    @property
    def SUPERVISED_MODELS_PATHS(self) -> dict[SupervisedModel, Path]:
        return {
            model: self.MODELS_DIR / filename
            for model, filename in SUPERVISED_MODELS_FILENAME.items()
        }

    N_MATCHES: int = Field(3, ge=1)
    BLOCK_COL: Optional[str] = "POSTCODE"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SVOC_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    @model_validator(mode="after")
    def validate_block_col(cls, values):
        if values.BLOCK_COL is None:
            warnings.warn(
                "BLOCK_COL is set to None. Record matching will be performed "
                "by considering all possible pairs of records.",
                UserWarning,
            )
            return values

        allowed_keys = set(DataColumns.model_fields.keys())

        if values.BLOCK_COL not in allowed_keys:
            raise ValueError(
                f"Invalid BLOCK_COL '{values.BLOCK_COL}'. "
                f"Allowed values: {sorted(allowed_keys)}"
            )

        return values

import yaml

def get_settings(config_path: str | None = None):
    if config_path is None:
        return Settings()
    else:
        with open(config_path) as f:
            return Settings(**yaml.safe_load(f))