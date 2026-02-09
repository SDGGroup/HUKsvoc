"""Configuration settings for the SVOC (Sales Venue Outlet Clustering) matching system.

This module defines configuration classes using Pydantic for:
- Data source configuration (CSV files or database tables)
- Column mapping for input and benchmark datasets
- Model storage paths
- Matching algorithm parameters
"""

from pathlib import Path
from pydantic import BaseModel, Field, model_validator#, ConfigDict
from pydantic_settings import  BaseSettings, SettingsConfigDict
from svoc.constants import SUPERVISED_MODELS_FILENAME
from svoc.supervised.enums import SupervisedModel
from typing import Optional
import warnings
import yaml


class DataColumns(BaseModel):
    """Column mapping configuration for data sources.
    
    Defines the mapping between standardized column names and actual column names
    in the input/benchmark datasets. All columns are required except LATITUDE and LONGITUDE.
    
    Attributes:
        ID: Unique identifier column name. Default: "ID"
        OUTLET_NAME: Outlet name column. Default: "OUTLET_NAME"
        ADDRESS: Street address column. Default: "ADDRESS"
        POSTCODE: Postal code column. Default: "POSTCODE"
        LATITUDE: Geographic latitude (optional). Default: "LATITUDE"
        LONGITUDE: Geographic longitude (optional). Default: "LONGITUDE"
    """
    ID: str = "ID"
    OUTLET_NAME: str = "OUTLET_NAME"
    ADDRESS: str = "ADDRESS"
    POSTCODE: str = "POSTCODE"
    LATITUDE: Optional[str] = "LATITUDE"
    LONGITUDE: Optional[str] = "LONGITUDE"

    # model_config = ConfigDict(extra="allow") # Allows extra values


class Settings(BaseSettings):
    """Main configuration class for the SVOC matching system.
    
    Handles all configuration including data sources, column mappings, model paths,
    and algorithm parameters. Supports loading from environment variables, .env files,
    or YAML configuration files.
    
    Environment variables can be set with SVOC_ prefix (e.g., SVOC_DATA_DIR).
    Nested fields use double underscore delimiter (e.g., SVOC_INPUT_COLUMNS__ID).
    
    Attributes:
        DATA_DIR: Directory containing input data files. Default: "."
        INPUT_DATA_FILENAME: Name of input CSV file. Default: ""
        BENCHMARK_DATA_FILENAME: Name of benchmark CSV file. Default: ""
        INPUT_DATATABLE: Name of input database table. Default: ""
        BENCHMARK_DATATABLE: Name of benchmark database table. Default: ""
        INPUT_COLUMNS: Column mapping for input data. Default: DataColumns()
        BENCHMARK_COLUMNS: Column mapping for benchmark data. Default: DataColumns()
        MODELS_DIR: Directory for saving/loading models. Default: "./models"
        N_MATCHES: Number of top matches to return per record. Default: 3
        K_NEIGHBOURS: Number of neighbors for kNN blocking. Default: 6
        BLOCK_COL: Column to use for blocking (None for no blocking). Default: "POSTCODE"
    """

    DATA_DIR: Path = Path(".")
    INPUT_DATA_FILENAME: str =  ""
    BENCHMARK_DATA_FILENAME: str =  ""
    
    @property
    def INPUT_FILEPATH(self) -> Path:
        """Full path to the input CSV file.
        
        Returns:
            Path object combining DATA_DIR and INPUT_DATA_FILENAME
        """
        return self.DATA_DIR / self.INPUT_DATA_FILENAME
    
    @property
    def BENCHMARK_FILEPATH(self) -> Path:
        """Full path to the benchmark CSV file.
        
        Returns:
            Path object combining DATA_DIR and BENCHMARK_DATA_FILENAME
        """
        return self.DATA_DIR / self.BENCHMARK_DATA_FILENAME

    INPUT_DATATABLE: str =  ""
    BENCHMARK_DATATABLE: str = ""

    INPUT_COLUMNS: DataColumns = Field(default_factory=DataColumns)
    BENCHMARK_COLUMNS: DataColumns = Field(default_factory=DataColumns)
    
    @property
    def INPUT_COLUMNS_DICT(self) -> dict[str, str]:
        """Dictionary representation of input column mappings.
        
        Returns:
            Dictionary with column mappings, excluding None values
        """
        return self.INPUT_COLUMNS.model_dump(exclude_none=True)
    
    @property
    def BENCHMARK_COLUMNS_DICT(self) -> dict[str, str]:
        """Dictionary representation of benchmark column mappings.
        
        Returns:
            Dictionary with column mappings, excluding None values
        """
        return self.BENCHMARK_COLUMNS.model_dump(exclude_none=True)
    
    def _core_columns(self, columns: dict[str, str]) -> dict[str, str]:
        """Filter column mappings to exclude geographic coordinates.
        
        Args:
            columns: Dictionary of column mappings
            
        Returns:
            Dictionary excluding LATITUDE and LONGITUDE columns
        """
        return {
            k: v
            for k, v in columns.items()
            if k not in {"LATITUDE", "LONGITUDE"}
        }
    
    @property
    def INPUT_CORE_COLUMNS_DICT(self) -> dict[str, str]:
        """Core input columns excluding geographic coordinates.
        
        Returns:
            Dictionary of input columns without LATITUDE/LONGITUDE
        """
        return self._core_columns(self.INPUT_COLUMNS_DICT)
    
    @property
    def BENCHMARK_CORE_COLUMNS_DICT(self) -> dict[str, str]:
        """Core benchmark columns excluding geographic coordinates.
        
        Returns:
            Dictionary of benchmark columns without LATITUDE/LONGITUDE
        """
        return self._core_columns(self.BENCHMARK_COLUMNS_DICT)
    
    MODELS_DIR: Path = Path("./models")
    
    @property
    def SUPERVISED_MODELS_PATHS(self) -> dict[SupervisedModel, Path]:
        """Full paths to all supervised model files.
        
        Returns:
            Dictionary mapping SupervisedModel enum values to their file paths
        """
        return {
            model: self.MODELS_DIR / filename
            for model, filename in SUPERVISED_MODELS_FILENAME.items()
        }

    N_MATCHES: int = Field(3, ge=1)
    K_NEIGHBOURS: Optional[int] = Field(6, ge=1)
    BLOCK_COL: Optional[str] = "POSTCODE"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SVOC_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    @model_validator(mode="after")
    def validate_block_col(cls, values):
        """Validate that BLOCK_COL matches a defined column if set.
        
        Issues a warning if BLOCK_COL is None (no blocking, all pairs compared).
        Raises an error if BLOCK_COL is set but doesn't match any input column.
        
        Args:
            values: The Settings instance being validated
            
        Returns:
            The validated Settings instance
            
        Raises:
            ValueError: If BLOCK_COL is not a valid column name
        """
        if values.BLOCK_COL is None:
            warnings.warn(
                "BLOCK_COL is set to None. Record matching will be performed "
                "by considering all possible pairs of records.",
                UserWarning,
            )
            return values

        allowed_keys = set(values.INPUT_COLUMNS_DICT.keys())

        if values.BLOCK_COL not in allowed_keys:
            raise ValueError(
                f"Invalid BLOCK_COL '{values.BLOCK_COL}'. "
                f"Allowed values: {sorted(allowed_keys)}"
            )

        return values


def get_settings(config_path: str | None = None) -> Settings:
    """Load settings from a YAML configuration file or environment.
    
    If config_path is provided, loads settings from the YAML file and merges with
    environment variables (env vars take precedence). If config_path is None,
    loads only from environment variables and defaults.
    
    Args:
        config_path: Path to YAML configuration file. Default: None
        
    Returns:
        Configured Settings object
        
    Examples:
        >>> # Load from environment only
        >>> settings = get_settings()
        
        >>> # Load from YAML file
        >>> settings = get_settings("config/dev.yaml")
    """
    if config_path is None:
        return Settings()
    else:
        with open(config_path) as f:
            return Settings(**yaml.safe_load(f))