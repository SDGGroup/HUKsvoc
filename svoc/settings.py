
from pathlib import Path
from pydantic import BaseModel,Field, model_validator
from pydantic_settings import  BaseSettings, SettingsConfigDict
from svoc.constants import SUPERVISED_MODELS_FILENAME
from svoc.supervised.enums import SupervisedModel

class DataColumns(BaseModel):
    ID: str = "ID"
    OUTLET_NAME: str = "OutletName"
    ADDRESS: str = "OutletAddress"
    POSTCODE: str = "OutletPostcode"

class Settings(BaseSettings):

    DATA_DIR: Path = Path(".")
    INPUT_DATA_FILENAME: str = "HUK_bowimi_data.csv"
    BENCHMARK_DATA_FILENAME: str = "HUK_sap_data.csv"
    @property
    def INPUT_FILEPATH(self) -> Path:
        return self.DATA_DIR / self.INPUT_DATA_FILENAME
    @property
    def BENCHMARK_FILEPATH(self) -> Path:
        return self.DATA_DIR / self.BENCHMARK_DATA_FILENAME

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
    def SUPERVISED_MODEL_PATH(self) -> dict[SupervisedModel, Path]:
        return {
            model: self.MODELS_DIR / filename
            for model, filename in SUPERVISED_MODELS_FILENAME.items()
        }

    N_MATCHES: int = Field(3, ge=1)
    BLOCK_COL: str = "POSTCODE"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SVOC_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    @model_validator(mode="after")
    def validate_block_col(cls, values):
        allowed_keys = set(DataColumns.model_fields.keys())

        if values.BLOCK_COL not in allowed_keys:
            raise ValueError(
                f"Invalid block_col '{values.BLOCK_COL}'. "
                f"Allowed values: {sorted(allowed_keys)}"
            )

        return values
