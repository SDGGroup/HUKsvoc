"""Automatic rule-based matching module.

This module provides functionality for automatic outlet matching using distance-based filters.
Includes:
- Filter-based matching using threshold constraints
- Score normalization and ranking
- Group-based parallel processing with blocking
- Sequential filter application to find top-N matches
"""

import pandas as pd
from tqdm import tqdm
from svoc.datapreparation import split_df
from svoc.utils import concat_l
from svoc.automatic.features import get_features
from svoc.automatic.models import Distance
from svoc.constants import DistanceMethod, DistanceFilter


def filter_dataframe(
        df: pd.DataFrame, 
        dict_constraints: DistanceFilter
    ) -> pd.DataFrame:
    """Filter DataFrame rows based on column threshold constraints.
    
    Applies multiple threshold filters to a DataFrame, keeping only rows where
    each specified column value exceeds its threshold. Columns not present in
    the DataFrame are skipped with a warning.
    
    Args:
        df: DataFrame to filter
        dict_constraints: DistanceFilter object mapping column names to minimum threshold values
                         (e.g., {'outlet_name_cosine': 0.8, 'address_cosine': 0.7})
        
    Returns:
        Filtered DataFrame containing only rows meeting all threshold constraints
        
    Raises:
        TypeError: If df is not a DataFrame or dict_constraints is not a DistanceFilter object
        
    Note:
        Prints a warning if any constraint column is not found in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(dict_constraints, DistanceFilter):
        raise TypeError(
            f"dict_constraints must be a DistanceFilter object, got {type(dict_constraints).__name__}"
        )
    for column_name, threshold in dict_constraints.value.items():
        if column_name in df.columns:
            df = df[df[column_name] > threshold].copy()
        else:
            print(f"Warning: Column '{column_name}' not found in DataFrame.")
    return df


def norm_score(
        df: pd.DataFrame, 
        score_cols: list[str] | set[str]
    ) -> pd.Series:
    """Calculate normalized score as mean across specified columns.
    
    Computes the average score across multiple similarity/distance columns.
    Columns not present in the DataFrame are skipped with a warning (except
    'filter_threshold' which is silently ignored).
    
    Args:
        df: DataFrame containing score columns
        score_cols: List or set of column names to average
        
    Returns:
        Series containing the mean score for each row
        
    Raises:
        TypeError: If df is not a DataFrame
        
    Note:
        Missing columns (except 'filter_threshold') trigger a warning.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}"
        )
    score_cols = list(score_cols)
    for column_name in score_cols:
        if column_name not in df.columns:
            if column_name != 'filter_threshold':
                print(f"Warning: Column '{column_name}' not found in DataFrame.")
            score_cols.remove(column_name)
    score = df[score_cols].mean(axis=1)
    return score


def check_matches(
        df_match: pd.DataFrame, 
        filter_label: int | str
    ) -> None:
    """Print match statistics for a specific filter.
    
    Displays the number of unique benchmark IDs and input IDs that were matched
    by a particular filter. Used for verbose output during matching.
    
    Args:
        df_match: DataFrame containing matched pairs with 'ID_1' and 'ID_2' columns
        filter_label: Identifier for the filter (e.g., filter number or name)
        
    Returns:
        None (prints to stdout)
        
    Raises:
        TypeError: If df_match is not a DataFrame
        ValueError: If required columns 'ID_1' or 'ID_2' are missing
    """
    if not isinstance(df_match, pd.DataFrame):
        raise TypeError(
            f"df_match must be a pandas DataFrame, got {type(df_match).__name__}"
        )
    if 'ID_1' not in df_match.columns or 'ID_2' not in df_match.columns:
        raise ValueError(
            "df_match must contain 'ID_1' and 'ID_2' columns"
        )
    ids_1 = len(df_match['ID_1'].drop_duplicates())
    ids_2 = len(df_match['ID_2'].drop_duplicates())
    print(f"Filter {filter_label}: {ids_1} IDs have been matched with {ids_2} IDs from the input dataset.")


def find_automatic_matches(
        filters: list[DistanceFilter], 
        features: pd.DataFrame, 
        n: int = 3, 
        verbose: bool = True
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find matches by sequentially applying distance-based filters.
    
    Applies a series of filters to identify the top-N matches for each record.
    Filters are applied in sequence, and once a record reaches N matches, it's
    excluded from subsequent filters. This creates a priority hierarchy where
    earlier filters are preferred.
    
    Algorithm:
    1. For each filter in sequence:
       - Apply threshold constraints to identify potential matches
       - Calculate normalized scores
       - Rank matches by score within each ID_1 group
       - Keep top matches up to N total per ID_1
       - Remove fully matched records from subsequent processing
    2. Combine all matches from all filters
    
    Args:
        filters: List of DistanceFilter objects, each containing threshold constraints
        features: DataFrame with computed features for all candidate pairs
                 Must include columns: 'ID_1', 'ID_2', and feature columns
        n: Maximum number of matches to find per record. Default: 3
        verbose: If True, print match statistics for each filter. Default: True
        
    Returns:
        tuple containing:
        - all_matches: DataFrame of all matched pairs with columns:
          ['ID_1', 'ID_2', features..., 'ID_filter', 'score', 'match_type']
        - remaining_features: DataFrame of unmatched candidate pairs
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If n < 1 or required columns are missing
        
    Note:
        Filter order matters! Earlier filters take precedence in the matching hierarchy.
    """
    if not isinstance(filters, list):
        raise TypeError(
            f"filters must be a list, got {type(filters).__name__}"
        )
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            f"features must be a pandas DataFrame, got {type(features).__name__}"
        )
    if not isinstance(n, int) or n < 1:
        raise ValueError(
            f"n must be a positive integer, got {n}"
        )
    if 'ID_1' not in features.columns or 'ID_2' not in features.columns:
        raise ValueError(
            "features DataFrame must contain 'ID_1' and 'ID_2' columns"
        )
    l_matches = []
    missing_matches = features[["ID_1"]].drop_duplicates().copy()
    missing_matches['counter'] = n
    for i, f in enumerate(filters):
        # find matches
        matches_filter_i = filter_dataframe(features, f)
        if matches_filter_i.empty:
            if verbose:
                print(f"Filter {i}: Any match found")
            continue
        else:
            matches_filter_i['ID_filter'] = i + 1
            matches_filter_i['score'] = norm_score(matches_filter_i, f.value.keys())
            matches_filter_i = (
                matches_filter_i
                .merge(
                    missing_matches,  # contiene ID_1 + counter
                    on='ID_1',
                    how='left'
                )
                .sort_values(['ID_1', 'score'], ascending=[True, False])
                .assign(rank=lambda x: x.groupby('ID_1').cumcount())
            )
            matches_filter_i = matches_filter_i[
                matches_filter_i['rank'] < matches_filter_i['counter']
            ].drop(columns=['rank', 'counter'])

            # check matches
            if verbose:
                check_matches(matches_filter_i, i)
            
            # append matches
            l_matches.append(matches_filter_i)

            match_count = matches_filter_i.groupby('ID_1')['score'].count()   

            # Removing ID_1 that have already reached the required number of matches
            full_matched = match_count[match_count>=n].index.tolist()
            features = features[~(features['ID_1'].isin(full_matched))]
            # Removing already matched pairs from the features DataFrame to avoid re-matching in subsequent filters
            matches_filter_i = matches_filter_i[~(matches_filter_i['ID_1'].isin(full_matched))]
            features = (
                features
                    .merge(matches_filter_i[['ID_1', 'ID_2']],
                        on=['ID_1', 'ID_2'],
                        how='left',
                        indicator=True)
                    .query('_merge == "left_only"')
                    .drop(columns='_merge')
            )

            missing_matches = (
                missing_matches
                .merge(match_count.reset_index(), on='ID_1', how='left')
                .assign(counter=lambda x: x['counter'] - x['score'].fillna(0).clip(lower=0))
                .drop(columns='score')
                )

    all_matches = concat_l(l_matches)
    all_matches["match_type"] = "auto"
    remaining_features = features
    return all_matches, remaining_features


def get_automatic_matches(
        df_benchmark: pd.DataFrame, 
        df_input: pd.DataFrame, 
        distances: list[Distance], 
        filters: list[DistanceFilter], 
        block_col: str | None = None, 
        n_groups: int = 15, 
        n_matches: int = 3, 
        verbose: bool = True,
        window: int = 1,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complete automatic matching pipeline with group-based processing.
    
    High-level orchestrator for automatic matching that:
    1. Validates inputs and sets up blocking column
    2. Splits benchmark data into balanced groups for parallel processing
    3. For each group:
       - Filters relevant input records
       - Computes features for candidate pairs
       - Applies sequential filters to find top-N matches
    4. Combines results from all groups
    
    The grouping strategy reduces memory usage and enables progress tracking
    for large datasets.
    
    Args:
        df_benchmark: Benchmark DataFrame with reference outlet records
        df_input: Input DataFrame with outlets to match against benchmark
        distances: List of Distance objects specifying features to compute
        filters: List of DistanceFilter objects for sequential matching
        block_col: Column name for blocking/grouping. If None, processes all pairs
                  (creates dummy blocking column). Default: None
        n_groups: Number of groups to split benchmark data into. Default: 15
        n_matches: Maximum matches to find per input record. Default: 3
        verbose: If True, print progress and match statistics. Default: True
        window: Window size for sorted neighbourhood blocking. Default: 1
        
    Returns:
        tuple containing:
        - all_matches: DataFrame of all matched pairs with scores and metadata
        - all_features: DataFrame of all computed features for candidate pairs
        - remaining_features: DataFrame of unmatched candidate pairs
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If block_col is specified but not found in DataFrames,
                   or if n_groups < 1 or n_matches < 1
        
    Warning:
        Using block_col=None processes all possible pairs, which can be very slow
        and memory-intensive for large datasets.
        
    Example:
        >>> matches, features, remaining = get_automatic_matches(
        ...     df_benchmark=df_bench,
        ...     df_input=df_in,
        ...     distances=DISTANCES,
        ...     filters=FILTERS_AUTO,
        ...     block_col='POSTCODE',
        ...     n_matches=3
        ... )
    """
    if not isinstance(df_benchmark, pd.DataFrame):
        raise TypeError(
            f"df_benchmark must be a pandas DataFrame, got {type(df_benchmark).__name__}"
        )
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(
            f"df_input must be a pandas DataFrame, got {type(df_input).__name__}"
        )
    if not isinstance(distances, list):
        raise TypeError(
            f"distances must be a list, got {type(distances).__name__}"
        )
    if not isinstance(filters, list):
        raise TypeError(
            f"filters must be a list, got {type(filters).__name__}"
        )
    if not isinstance(n_groups, int) or n_groups < 1:
        raise ValueError(
            f"n_groups must be a positive integer, got {n_groups}"
        )
    if not isinstance(n_matches, int) or n_matches < 1:
        raise ValueError(
            f"n_matches must be a positive integer, got {n_matches}"
        )
    
    if block_col is not None and block_col not in df_benchmark.columns:
            raise ValueError(
                f"block_col '{block_col}' not found in df_benchmark columns: "
                f"{list(df_benchmark.columns)}"
            )
    
    if block_col is not None and block_col not in df_input.columns:
        raise ValueError(
                f"block_col '{block_col}' not found in df_input columns: "
                f"{list(df_input.columns)}"
            )
       
    if block_col is None:
        df_benchmark = df_benchmark.assign(_DUMMY_BLOCK=1)
        df_input = df_input.assign(_DUMMY_BLOCK=1)
        block_col = '_DUMMY_BLOCK'

    results_df = split_df(df=df_benchmark, split_col=block_col, num_groups=n_groups)
    l_all_matches = []
    l_features = []
    l_remaining_features = []
    for i, group in enumerate(tqdm(results_df['GROUP'].tolist())):#[::-1])):
        
        if group == []:
            continue # skip empty groups
 
        if verbose:
            print("\nElaborating group nr.",i + 1)
        
        df_y_filtered = df_input[df_input[block_col].isin(group)]#.drop_duplicates()
        df_x_filtered = df_benchmark[df_benchmark[block_col].isin(group)]#.drop_duplicates()
        features = get_features(distances, df_x=df_x_filtered, df_y=df_y_filtered, window=window,
                                block_col=(block_col if block_col != '_DUMMY_BLOCK' else None))
        l_features.append(features)

        matches_auto, remaining_features = find_automatic_matches(filters, features, n=n_matches, verbose=verbose)
        l_all_matches.append(matches_auto)
        l_remaining_features.append(remaining_features)

    if not l_all_matches:
        print("⚠️ No matches found")

    return concat_l(l_all_matches), concat_l(l_features), concat_l(l_remaining_features)

