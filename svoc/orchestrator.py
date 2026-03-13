"""Main orchestrator module for SVOC outlet matching pipeline.

This module provides high-level orchestration functions that coordinate the complete
outlet matching workflow:
- Geographic clustering using k-nearest neighbors (kNN) for blocking optimization
- Complete record linkage pipeline integrating automatic and supervised matching
- Data preparation, feature computation, and output formatting

These functions serve as the primary entry points for the SVOC matching system.
"""

import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors

from svoc.settings import Settings
from svoc.datapreparation import check_duplicates, prepare_data, rename_and_select_cols, make_upper_str, remove_accents_and_regex
from svoc.rl import get_matches_with_clusters, prepare_output
from svoc.constants import DISTANCES, FILTERS_AUTO
from logging import Logger

def svoc_knn(
        settings: Settings, 
        df_input: pd.DataFrame, 
        df_benchmark: pd.DataFrame, 
        k: int = 6,
        save: bool = True,
        logger: Logger | None = None
        ) -> dict[str, list[str]]:
    """Create geographic neighborhood groups using k-nearest neighbors.
    
    Builds a dictionary mapping each postcode to its k-nearest neighbor postcodes
    based on geographic distance (haversine). This enables efficient clustering-based
    blocking for record linkage, reducing the number of comparisons while maintaining
    high recall by comparing records in nearby geographic areas.
    
    The algorithm:
    1. Extracts postcode and lat/lon from both datasets
    2. Converts coordinates to radians for haversine distance
    3. Fits kNN model on input data coordinates
    4. For each benchmark postcode, finds k nearest input postcodes
    5. Adds missing postcodes and self-references
    6. Optionally saves the neighborhood mapping to JSON
    
    Args:
        settings: Settings object with column mappings and configuration.
                 Must include LATITUDE, LONGITUDE, and BLOCK_COL in both
                 INPUT_COLUMNS_DICT and BENCHMARK_COLUMNS_DICT.
        df_input: Input DataFrame with outlets to match (must have lat/lon columns)
        df_benchmark: Benchmark DataFrame with reference outlets (must have lat/lon columns)
        k: Number of nearest neighbors to find for each postcode. Default: 6
        save: If True, save neighborhood mapping to 'postcode_neighbourhood.json'. Default: True
        
    Returns:
        Dictionary mapping each benchmark postcode to a list of nearby input postcodes.
        Format: {benchmark_postcode: [neighbor1, neighbor2, ...]}
        Missing postcodes get single-item lists containing themselves.
        
    Raises:
        TypeError: If inputs are not of expected types
        KeyError: If LATITUDE or LONGITUDE columns are not specified in settings
        ValueError: If k < 1 or if required columns are missing from DataFrames
        
    Example:
        >>> settings = Settings(BLOCK_COL='POSTCODE', K_NEIGHBOURS=6)
        >>> groups = svoc_knn(settings, df_input, df_benchmark, k=6, save=True)
        >>> # groups = {'SW1A1AA': ['SW1A1AA', 'SW1A2AA', 'SW1H0TL', ...], ...}
    """
    if not isinstance(settings, Settings):
        raise TypeError(
            f"settings must be a Settings object, got {type(settings).__name__}"
        )
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(
            f"df_input must be a pandas DataFrame, got {type(df_input).__name__}"
        )
    if not isinstance(df_benchmark, pd.DataFrame):
        raise TypeError(
            f"df_benchmark must be a pandas DataFrame, got {type(df_benchmark).__name__}"
        )
    if not isinstance(k, int) or k < 1:
        raise ValueError(
            f"k must be a positive integer, got {k}"
        )

    LATITUDE, LONGITUDE = 'LATITUDE', 'LONGITUDE'
    required_keys = {LATITUDE, LONGITUDE}
    if not (required_keys <= settings.BENCHMARK_COLUMNS_DICT.keys() 
            and required_keys <= settings.INPUT_COLUMNS_DICT.keys()
            ):
        raise KeyError("Missing required keys: both 'LATITUDE' and 'LONGITUDE' columns must be specified in the settings.")
    
    def prepare_data_for_knn(df: pd.DataFrame, cols: dict):
        df_out=check_duplicates(df=df, logger=logger)
        df_out = rename_and_select_cols(df=df_out, dict_cols={k: cols[k] for k in [settings.BLOCK_COL, LATITUDE, LONGITUDE]})
        df_out = make_upper_str(df=df_out)
        df_out = remove_accents_and_regex(
            df=df_out, 
            re_pattern=r'[^a-zA-Z0-9]', 
            l_cols_not_to_apply=[LATITUDE,LONGITUDE]
            )
        return df_out
      
    def get_radians(df: pd.DataFrame):
        df_out = df.dropna(subset=[LATITUDE,LONGITUDE]).copy()
        df_out[[LATITUDE, LONGITUDE]] = np.radians(df_out[[LATITUDE, LONGITUDE]].astype(float))
        return df_out


    pc_bench=prepare_data_for_knn(
        df=df_benchmark, 
        cols=settings.BENCHMARK_COLUMNS_DICT
        )
    all_pc = list(pc_bench[settings.BLOCK_COL].copy())
    pc_bench = get_radians(pc_bench)


    pc_input=prepare_data_for_knn(
        df=df_input, 
        cols=settings.INPUT_COLUMNS_DICT
        )
    pc_input = get_radians(pc_input)


    nn = NearestNeighbors(
        n_neighbors=k,
        metric="haversine"
    )
    nn.fit(pc_input[[LATITUDE,LONGITUDE]])
    distances, indices = nn.kneighbors(pc_bench[[LATITUDE,LONGITUDE]])

    groups = {}
    for i, cap in enumerate(pc_bench[settings.BLOCK_COL]):
        neighbors = pc_input.iloc[indices[i]][settings.BLOCK_COL].tolist()
        neighbors = list(dict.fromkeys(neighbors))
        groups[cap] = neighbors

    # If a postal code does not have any neighbors, we add a group with only itself as a member
    missing_pcs = [
        pc for pc in all_pc
        if pc not in groups and pd.notna(pc)
    ]
    for pc in missing_pcs:
        groups[pc] = [pc] 
    # If the neighbourhood of a postal code does not include itself, we add it to the group
    x = [pc for pc in groups.keys() if pc not in groups[pc]]
    for xpc in x:
        groups[xpc].append(xpc)
  
    if save:
        output_path = settings.DATA_DIR / "postcode_neighbourhood.json"
        with open(output_path, "w") as f:
            json.dump(groups, f, indent=2)

    return groups

def svoc_record_linkage(
        settings: Settings, 
        df_input: pd.DataFrame, 
        df_benchmark: pd.DataFrame, 
        groups: dict | None = None,
        save: bool = False,
        logger: Logger | None = None
        ) -> pd.DataFrame:
    """Execute complete record linkage pipeline for outlet matching.
    
    Main orchestrator function that coordinates the entire matching workflow:
    1. Data preparation and cleaning (standardization, noise removal)
    2. Feature computation with blocking or clustering
    3. Automatic rule-based matching using distance filters
    4. Supervised model-based matching for remaining pairs
    5. Output formatting and optional CSV export
    
    This is the primary entry point for performing outlet matching between
    an input dataset and a benchmark reference dataset.
    
    Args:
        settings: Settings object containing configuration:
                 - Column mappings (INPUT_CORE_COLUMNS_DICT, BENCHMARK_CORE_COLUMNS_DICT)
                 - Blocking column (BLOCK_COL)
                 - Number of matches to return (N_MATCHES)
                 - Model paths (SUPERVISED_MODELS_PATHS)
                 - Data directory for output (DATA_DIR)
        df_input: Input DataFrame with outlet records to match against benchmark
        df_benchmark: Benchmark DataFrame with reference outlet records
        groups: Optional neighborhood mapping from svoc_knn for clustering-based matching.
               If None, uses standard blocking. Default: None
        save: If True, saves output to 'output.csv' in DATA_DIR. Default: False
        
    Returns:
        DataFrame containing matched pairs with columns:
        - Input record IDs (from df_input)
        - Benchmark record IDs (from df_benchmark)
        - match_type: 'auto' or 'supervised'
        - score: Match confidence score
        - Additional metadata from matching process
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If required columns are missing or settings are invalid
        KeyError: If required configuration keys are missing from settings
        
    Example:
        >>> # Basic usage with blocking
        >>> settings = get_settings('config/dev.yaml')
        >>> matches = svoc_record_linkage(
        ...     settings, 
        ...     df_input, 
        ...     df_benchmark,
        ...     save=True
        ... )
        
        >>> # Advanced usage with kNN clustering
        >>> groups = svoc_knn(settings, df_input, df_benchmark, k=6)
        >>> matches = svoc_record_linkage(
        ...     settings, 
        ...     df_input, 
        ...     df_benchmark,
        ...     groups=groups,
        ...     save=True
        ... )
    
    Note:
        Using the groups parameter (from svoc_knn) typically provides better
        performance and matching quality by intelligently grouping geographically
        nearby records for comparison.
    """
    if not isinstance(settings, Settings):
        raise TypeError(
            f"settings must be a Settings object, got {type(settings).__name__}"
        )
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(
            f"df_input must be a pandas DataFrame, got {type(df_input).__name__}"
        )
    if not isinstance(df_benchmark, pd.DataFrame):
        raise TypeError(
            f"df_benchmark must be a pandas DataFrame, got {type(df_benchmark).__name__}"
        )
    if groups is not None and not isinstance(groups, dict):
        raise TypeError(
            f"groups must be a dictionary or None, got {type(groups).__name__}"
        )

    df_benchmark_clean = prepare_data(
        df=df_benchmark, 
        dict_cols=settings.BENCHMARK_CORE_COLUMNS_DICT,
        logger=logger
        )

    df_input_clean = prepare_data(
        df=df_input, 
        dict_cols=settings.INPUT_CORE_COLUMNS_DICT,
        logger=logger
        )

    all_matches, features, remaining_features = get_matches_with_clusters(
        df_input=df_input_clean, 
        df_benchmark=df_benchmark_clean, 
        distances=DISTANCES, 
        filters=FILTERS_AUTO,
        block_col=settings.BLOCK_COL,
        groups=groups,
        n_matches=settings.N_MATCHES, verbose=False,
        models_path_dict=settings.SUPERVISED_MODELS_PATHS,
        logger=logger
        )

    output = prepare_output(
        matches=all_matches,
        distances=DISTANCES,
        filters=FILTERS_AUTO
    )
    
    if save:
        output.to_csv(settings.DATA_DIR / 'output.csv', index=False)    
        if logger:
            logger.info(f"Output saved to {settings.DATA_DIR / 'output.csv'}")
        
    return output