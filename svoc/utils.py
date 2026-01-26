import pickle
from pathlib import Path
import pandas as pd
from svoc.settings import Settings

def concat_l(l):
    out = pd.concat(
        [df for df in l if not df.empty],
        ignore_index=True
    ) if any(not df.empty for df in l) else pd.DataFrame()
    return out


def load_pickle(pickle_path: Path):

    pickle_path = Path(pickle_path)

    with open(pickle_path, "rb") as f:
        out = pickle.load(f)

    return out

def save_pickle(obj, pickle_path: Path):

    pickle_path = Path(pickle_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "wb") as f:
        pickle.dump(obj, f)
    
    return None


def read_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Case 1: load from CSV files
    if (
        settings.INPUT_DATA_FILENAME != ""
        and settings.BENCHMARK_DATA_FILENAME != ""
    ):
        df_input = pd.read_csv(settings.INPUT_FILEPATH, sep=",", dtype=str)
        df_benchmark = pd.read_csv(settings.BENCHMARK_FILEPATH, sep=",", dtype=str)

        return df_input, df_benchmark

    # Case 2: load from database tables
    if (
        settings.INPUT_DATA_FILENAME == ""
        and settings.BENCHMARK_DATA_FILENAME == ""
        and settings.INPUT_DATATABLE != ""
        and settings.BENCHMARK_DATATABLE != ""
    ):
        from databricks.sdk.runtime import spark
        
        def import_table(table: str) -> pd.DataFrame:
            df = spark.table(table)
            return df.toPandas()

        df_input = import_table(settings.INPUT_DATATABLE)
        df_benchmark = import_table(settings.BENCHMARK_DATATABLE)

        return df_input, df_benchmark

    # Invalid configuration
    raise ValueError(
        "Invalid data source configuration. "
        "Either both INPUT_DATA_FILENAME and BENCHMARK_DATA_FILENAME must be set, "
        "or both INPUT_DATATABLE and BENCHMARK_DATATABLE must be set."
    )
