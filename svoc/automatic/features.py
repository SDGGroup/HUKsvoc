"""Feature computation module for automatic outlet matching.

This module provides functionality for computing similarity features between record pairs,
including:
- String distance calculations (cosine, Levenshtein, Jaro-Winkler, Q-gram)
- Exact matching
- Substring and word-level matching
- Blocking and indexing strategies for efficient pair generation
- Integration with the recordlinkage library
"""

import pandas as pd
import numpy as np
from recordlinkage import Compare, Index
from svoc.automatic.enums import DistanceMethod
from svoc.automatic.models import Distance
import logging


def initialize_compare_cl(
        l_compare: list[Distance], 
        n_jobs_param: int = -1
    ) -> Compare:
    """Initialize a recordlinkage Compare object with specified distance methods.
    
    Creates and configures a Compare instance to compute similarity features between
    record pairs. Different distance methods are handled appropriately:
    - Standard methods (cosine, Levenshtein, etc.): Added to Compare object
    - Exact matching: Added using exact comparison
    - Substring/Word matching: Skipped (handled separately by manual_features)
    
    Args:
        l_compare: List of Distance objects specifying which features to compute
        n_jobs_param: Number of parallel jobs for computation. -1 uses all CPUs. Default: -1
        
    Returns:
        Configured Compare object ready for feature computation
        
    Raises:
        TypeError: If l_compare is not a list or n_jobs_param is not an integer
    """
    if not isinstance(l_compare, list):
        raise TypeError(
            f"l_compare must be a list of Distance objects, got {type(l_compare).__name__}"
        )
    if not isinstance(n_jobs_param, int):
        raise TypeError(
            f"n_jobs_param must be an integer, got {type(n_jobs_param).__name__}"
        )
    compare_cl = Compare(n_jobs=n_jobs_param)
    for element in l_compare:
        if (
            element.method is not None 
            and element.method!=DistanceMethod.EXACT
            and element.method!=DistanceMethod.SUBSTRING
            and element.method!=DistanceMethod.WORDSMATCH
        ):
            compare_cl.string(element.col_name, element.col_name, label=element.label,
                              method=element.method, missing_value=0)
        elif element.method == DistanceMethod.EXACT:
            compare_cl.exact(element.col_name, element.col_name, label=element.label, missing_value=0)
        else:
            pass  # Substring and Wordsmatch are handled separately
            
    return compare_cl


def rl_compare_block(
        df_1: pd.DataFrame, 
        df_2: pd.DataFrame, 
        compare_cl: Compare, 
        block_variable: str | None = None,
        window: int = 1,
    ) -> pd.DataFrame:
    """Generate candidate pairs and compute features using blocking strategy.
    
    Creates an index of candidate record pairs using blocking or full indexing,
    then computes similarity features for those pairs. Blocking reduces the number
    of comparisons by only pairing records that share a common blocking key.
    
    Blocking strategies:
    - block_variable=None: Full indexing (all possible pairs) - expensive!
    - window=1: Standard blocking on exact block_variable values
    - window>1: Sorted neighbourhood blocking (compares nearby sorted values)
    
    Args:
        df_1: First DataFrame (typically benchmark data)
        df_2: Second DataFrame (typically input data)
        compare_cl: Configured Compare object with distance methods
        block_variable: Column name to use for blocking. None = full index. Default: None
        window: Window size for sorted neighbourhood blocking. Default: 1
        
    Returns:
        DataFrame with computed features for each candidate pair.
        Index is MultiIndex (ID_1, ID_2) representing record pairs.
        Missing values are filled with 0.0.
        
    Raises:
        TypeError: If DataFrames or compare_cl are not of expected types
        ValueError: If block_variable doesn't exist in both DataFrames
        
    Warning:
        Using block_variable=None generates all possible pairs, which can be extremely
        large for big datasets (O(n*m) complexity).
    """
    if not isinstance(df_1, pd.DataFrame):
        raise TypeError(
            f"df_1 must be a pandas DataFrame, got {type(df_1).__name__}"
        )
    if not isinstance(df_2, pd.DataFrame):
        raise TypeError(
            f"df_2 must be a pandas DataFrame, got {type(df_2).__name__}"
        )
    if not isinstance(compare_cl, Compare):
        raise TypeError(
            f"compare_cl must be a recordlinkage Compare object, got {type(compare_cl).__name__}"
        )
    if block_variable is not None:
        if block_variable not in df_1.columns:
            raise ValueError(
                f"block_variable '{block_variable}' not found in df_1. Available columns: {list(df_1.columns)}"
            )
        if block_variable not in df_2.columns:
            raise ValueError(
                f"block_variable '{block_variable}' not found in df_2. Available columns: {list(df_2.columns)}"
            )

    indexer = Index()
    
    if block_variable is not None:
        if window > 1:
            indexer.sortedneighbourhood(block_variable, window=window)
        else:
            indexer.block(block_variable)
    else:
        rl_logger = logging.getLogger("recordlinkage")
        rl_logger.setLevel(logging.ERROR)
        indexer.full()
    
    candidate_links = indexer.index(df_1, df_2) 
    features = compare_cl.compute(candidate_links, df_1, df_2)
    features = features.fillna(0.0)
    return features


def vec_substring_matching(
        column_x: pd.Series, 
        column_y: pd.Series
    ) -> pd.Series:
    """Vectorized substring matching between two Series.
    
    Checks if either string is a substring of the other for each pair of values.
    Returns 1 if one string contains the other, 0 otherwise.
    
    Args:
        column_x: First Series of string values
        column_y: Second Series of string values (must have same index as column_x)
        
    Returns:
        Series of integers (0 or 1) indicating substring match
        
    Raises:
        TypeError: If inputs are not pandas Series
        
    Note:
        Missing values (NaN) are treated as non-matches (return 0).
    """
    if not isinstance(column_x, pd.Series):
        raise TypeError(
            f"column_x must be a pandas Series, got {type(column_x).__name__}"
        )
    if not isinstance(column_y, pd.Series):
        raise TypeError(
            f"column_y must be a pandas Series, got {type(column_y).__name__}"
        )
    
    out = 1 * np.array([
        (str(x) in str(y)) or (str(y) in str(x))
        if pd.notna(x) and pd.notna(y) else 0
        for x, y in zip(column_x, column_y)
    ])
    return pd.Series(out, index=column_x.index, dtype="int64")


def vec_word_subset(
    column_x: pd.Series, 
    column_y: pd.Series
    ) -> pd.Series:
    """Vectorized word-level subset matching between two Series.
    
    Checks if all words from one string appear in the other (in any order).
    Splits strings by whitespace and compares word sets. Returns 1 if one word
    set is a subset of the other, 0 otherwise.
    
    Args:
        column_x: First Series of string values
        column_y: Second Series of string values (must have same index as column_x)
        
    Returns:
        Series of integers (0 or 1) indicating word subset match
        
    Raises:
        TypeError: If inputs are not pandas Series
        
    Note:
        Missing values (NaN) are treated as non-matches (return 0).
        
    Example:
        >>> x = pd.Series(['MAIN STREET'])
        >>> y = pd.Series(['MAIN STREET LONDON'])
        >>> vec_word_subset(x, y)
        0    1  # "MAIN STREET" is subset of "MAIN STREET LONDON"
    """
    if not isinstance(column_x, pd.Series):
        raise TypeError(
            f"column_x must be a pandas Series, got {type(column_x).__name__}"
        )
    if not isinstance(column_y, pd.Series):
        raise TypeError(
            f"column_y must be a pandas Series, got {type(column_y).__name__}"
        )

    def contains_all_words(x, y):
        if pd.isna(x) or pd.isna(y):
            return 0
        w1 = set(str(x).split())
        w2 = set(str(y).split())
        return int(w1.issubset(w2) or w2.issubset(w1))

    return pd.Series(
        (contains_all_words(x, y) for x, y in zip(column_x, column_y)),
        index=column_x.index,
        dtype="int64"
    )


def manual_features(
        l_compare: list[Distance], 
        features: pd.DataFrame, 
        x: pd.DataFrame, 
        y: pd.DataFrame,
        index_x: str, 
        index_y: str,
    ) -> pd.DataFrame:
    """Add manually computed features for substring and word matching.
    
    Supplements the features computed by recordlinkage with custom features that
    require manual implementation (substring matching and word-level matching).
    Merges the original DataFrames to access raw column values, computes the
    custom features, and returns the augmented feature DataFrame.
    
    Args:
        l_compare: List of Distance objects specifying features to compute
        features: DataFrame with features already computed by recordlinkage
        x: First source DataFrame with original data
        y: Second source DataFrame with original data
        index_x: Column name for ID from first DataFrame (e.g., 'ID_1')
        index_y: Column name for ID from second DataFrame (e.g., 'ID_2')
        
    Returns:
        DataFrame with both original and manually computed features.
        Columns: [index_x, index_y, original_features..., new_features...]
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If required columns are missing
        
    Note:
        Only processes Distance objects with method=SUBSTRING or method=WORDSMATCH.
        Other distance methods are ignored (already handled by recordlinkage).
    """
    if not isinstance(l_compare, list):
        raise TypeError(
            f"l_compare must be a list, got {type(l_compare).__name__}"
        )
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            f"features must be a pandas DataFrame, got {type(features).__name__}"
        )
    if not isinstance(x, pd.DataFrame):
        raise TypeError(
            f"x must be a pandas DataFrame, got {type(x).__name__}"
        )
    if not isinstance(y, pd.DataFrame):
        raise TypeError(
            f"y must be a pandas DataFrame, got {type(y).__name__}"
        )

    original_cols = features.columns.tolist()

    # Merge
    features = features.reset_index(level=[index_x, index_y])
    features = features.merge(x.add_suffix('_x'), how='left', left_on=index_x, right_on='ID')
    features = features.merge(y.add_suffix('_y'), how='left', left_on=index_y, right_on='ID')
    
    new_cols = []
    for element in l_compare:
        if element.method == DistanceMethod.SUBSTRING:
            features[element.label] = vec_substring_matching(
                                   column_x=features[f'{element.col_name}_x'],
                                   column_y=features[f'{element.col_name}_y']
                                   )
            new_cols.append(element.label)
        elif element.method == DistanceMethod.WORDSMATCH:
            features[element.label] = vec_word_subset(
                                   column_x=features[f'{element.col_name}_x'],
                                   column_y=features[f'{element.col_name}_y']
                                   )
            new_cols.append(element.label)
        else:
            pass
    
    return features[[index_x, index_y] + original_cols + new_cols]


def get_features(
        distances: list[Distance], 
        df_x: pd.DataFrame, 
        df_y: pd.DataFrame, 
        block_col: str | None = None,
        window: int = 1,
        njobs: int = -1,
    ) -> pd.DataFrame:
    """Main feature computation pipeline for record pair comparison.
    
    Complete pipeline that:
    1. Initializes Compare object with specified distance methods
    2. Generates candidate pairs using blocking strategy
    3. Computes standard features (cosine, Levenshtein, etc.)
    4. Adds custom features (substring, word matching)
    
    This is the primary entry point for computing all similarity features between
    two datasets for outlet matching.
    
    Args:
        distances: List of Distance objects defining all features to compute
        df_x: First DataFrame (typically benchmark data), must have 'ID' column
        df_y: Second DataFrame (typically input data), must have 'ID' column
        block_col: Column name for blocking. None = full index (all pairs). Default: None
        window: Window size for sorted neighbourhood blocking. Default: 1
        njobs: Number of parallel jobs (-1 = all CPUs). Default: -1
        
    Returns:
        DataFrame with computed features for each candidate pair.
        Columns: ['ID_1', 'ID_2', feature_1, feature_2, ...]
        ID_1 refers to records from df_x, ID_2 from df_y.
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If DataFrames are missing required columns or block_col is invalid
        
    Example:
        >>> from svoc.automatic.models import Distance
        >>> from svoc.automatic.enums import DistanceMethod
        >>> 
        >>> distances = [
        ...     Distance('OUTLET_NAME', DistanceMethod.COSINE, 'name_cosine'),
        ...     Distance('ADDRESS', DistanceMethod.LEVENSHTEIN, 'addr_lev')
        ... ]
        >>> features = get_features(distances, df_benchmark, df_input, block_col='POSTCODE')
    """
    if not isinstance(distances, list):
        raise TypeError(
            f"distances must be a list, got {type(distances).__name__}"
        )
    if not isinstance(df_x, pd.DataFrame):
        raise TypeError(
            f"df_x must be a pandas DataFrame, got {type(df_x).__name__}"
        )
    if not isinstance(df_y, pd.DataFrame):
        raise TypeError(
            f"df_y must be a pandas DataFrame, got {type(df_y).__name__}"
        )
    
    compare_cl = initialize_compare_cl(distances, n_jobs_param=njobs)
    features = rl_compare_block(df_x, df_y, compare_cl, block_col, window)
    features = manual_features(distances, features, df_x, df_y, index_x="ID_1", index_y="ID_2")
    return features