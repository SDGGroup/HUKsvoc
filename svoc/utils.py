import pickle
from pathlib import Path
import pandas as pd
from svoc.settings import Settings
import logging

def get_logger() -> logging.Logger:
    """
    Creates and returns a configured logger for the current module.

    Sets up basic logging configuration with INFO level and a custom format
    including timestamp, level, and message. Uses logging.getLogger(__name__)
    to create a module-specific logger, ensuring proper hierarchy in logs.

    Returns:
        logging.Logger: The configured logger ready for use.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger(__name__)

def concat_l(l: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates a list of DataFrames, ignoring empty ones.

    Filters out empty DataFrames before concatenation and resets the index.
    Returns an empty DataFrame if all inputs are empty.

    Args:
        l: List of pandas DataFrames to concatenate.

    Returns:
        pd.DataFrame: Concatenated DataFrame or empty DataFrame if all are empty.
        
    Raises:
        TypeError: If l is not a list or contains non-DataFrame elements.
    """
    if not isinstance(l, list):
        raise TypeError(f"Expected a list of DataFrames, got {type(l).__name__}")
    
    for i, df in enumerate(l):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Element at index {i} is not a DataFrame. "
                f"Expected pd.DataFrame, got {type(df).__name__}"
            )

    out = pd.concat(
        [df for df in l if not df.empty],
        ignore_index=True
    ) if any(not df.empty for df in l) else pd.DataFrame()

    return out


def load_pickle(pickle_path: Path | str):
    """
    Loads an object from a pickle file.

    Args:
        pickle_path: Path to the pickle file (converted to Pathlib Path if needed).

    Returns:
        The deserialized object from the pickle file.

    Raises:
        TypeError: If pickle_path is not a valid path type.
    """
    if not isinstance(pickle_path, (Path, str)):
        raise TypeError(
            f"pickle_path must be a Path or string, got {type(pickle_path).__name__}"
        )

    pickle_path = Path(pickle_path)

    with open(pickle_path, "rb") as f:
        out = pickle.load(f)

    return out

def save_pickle(obj, pickle_path: Path | str) -> None:
    """Saves an object to a pickle file, creating parent directories if needed.

    Args:
        obj: The object to serialize.
        pickle_path: Path to save the pickle file (converted to Pathlib Path).

    Returns:
        None
        
    Raises:
        TypeError: If pickle_path is not a valid path type.
    """
    if not isinstance(pickle_path, (Path, str)):
        raise TypeError(
            f"pickle_path must be a Path or string, got {type(pickle_path).__name__}"
        )
    
    pickle_path = Path(pickle_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "wb") as f:
        pickle.dump(obj, f)
    
    return None

def read_data_from_csv(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads input and benchmark CSV data using settings.
    Loads CSVs with comma separator and string dtype. Validates that both filenames are set.

    Args:
        settings: Configuration object with INPUT_DATA_FILENAME, BENCHMARK_DATA_FILENAME,
            INPUT_FILEPATH, and BENCHMARK_FILEPATH.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Input DataFrame and benchmark DataFrame.

    Raises:
        TypeError: If settings is not a Settings object.
        ValueError: If either filename is empty (invalid config).
    """
    if not isinstance(settings, Settings):
        raise TypeError(
            f"settings must be a Settings object, got {type(settings).__name__}"
        )
    
    if not settings.INPUT_DATA_FILENAME:
        raise ValueError(
            "INPUT_DATA_FILENAME is not set in settings. "
            "Please configure the input data filename."
        )
    
    if not settings.BENCHMARK_DATA_FILENAME:
        raise ValueError(
            "BENCHMARK_DATA_FILENAME is not set in settings. "
            "Please configure the benchmark data filename."
        )
    
    try:
        if not settings.INPUT_FILEPATH.exists():
            raise FileNotFoundError(
                f"Input CSV file not found: {settings.INPUT_FILEPATH}"
            )
        df_input = pd.read_csv(settings.INPUT_FILEPATH, sep=",", dtype=str)
    except Exception as e:
        raise RuntimeError(
            f"Error reading input CSV file {settings.INPUT_FILEPATH}: {str(e)}"
        )
    
    try:
        if not settings.BENCHMARK_FILEPATH.exists():
            raise FileNotFoundError(
                f"Benchmark CSV file not found: {settings.BENCHMARK_FILEPATH}"
            )
        df_benchmark = pd.read_csv(settings.BENCHMARK_FILEPATH, sep=",", dtype=str)
    except Exception as e:
        raise RuntimeError(
            f"Error reading benchmark CSV file {settings.BENCHMARK_FILEPATH}: {str(e)}"
        )

    return df_input, df_benchmark

def read_data_from_table(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads input and benchmark data from Databricks tables using settings.

    Designed for Databricks Runtime. Imports Spark tables via databricks.sdk.runtime.spark
    and converts them to pandas DataFrames using toPandas(). Validates both table names are set.

    Args:
        settings: Configuration object with INPUT_DATATABLE, BENCHMARK_DATATABLE attributes.

    Returns:
        tuple: a tuple of pandas DataFrames, the Input DataFrame and benchmark DataFrame from tables.

    Raises:
        TypeError: If settings is not a Settings object.
        ValueError: If either INPUT_DATATABLE or BENCHMARK_DATATABLE is empty.
        RuntimeError: If not running in a Databricks environment.
        Exception: Spark table access errors (table not found, permissions, etc.).

    Note:
        Requires Databricks Runtime environment with databricks.sdk.runtime.spark available.
    """
    if not isinstance(settings, Settings):
        raise TypeError(
            f"settings must be a Settings object, got {type(settings).__name__}"
        )
    
    if not settings.INPUT_DATATABLE:
        raise ValueError(
            "INPUT_DATATABLE is not set in settings. "
            "Please configure the input data table name."
        )
    
    if not settings.BENCHMARK_DATATABLE:
        raise ValueError(
            "BENCHMARK_DATATABLE is not set in settings. "
            "Please configure the benchmark data table name."
        )
    
    try:
        from databricks.sdk.runtime import spark
    except ImportError:
        raise RuntimeError(
            "databricks.sdk.runtime is not available. "
            "This function can only be used in a Databricks environment."
        )
    
    def import_table(table: str) -> pd.DataFrame:
        """Import a Spark table and convert to pandas DataFrame."""
        if not isinstance(table, str):
            raise TypeError(f"table must be a string, got {type(table).__name__}")
        if not table:
            raise ValueError("table name cannot be empty")
        
        try:
            df = spark.table(table)
            return df.toPandas()
        except Exception as e:
            raise RuntimeError(
                f"Failed to import table '{table}': {str(e)}. "
                f"Ensure the table exists and you have proper permissions."
            )

    try:
        df_input = import_table(settings.INPUT_DATATABLE)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading input table: {str(e)}")
    
    try:
        df_benchmark = import_table(settings.BENCHMARK_DATATABLE)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading benchmark table: {str(e)}")

    return df_input, df_benchmark