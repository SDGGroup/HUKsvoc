"""Record linkage workflow coordination module.

This module provides high-level coordination functions for the complete record linkage
workflow, integrating both automatic and supervised matching approaches:
- Blocking-based matching with group partitioning
- Clustering-based matching using neighborhood groups
- Output formatting for standardized results

These functions orchestrate the sequential application of automatic filters followed
by supervised models, ranking and filtering results to return top-N matches.
"""

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from svoc.datapreparation import split_df
from svoc.utils import concat_l
from svoc.automatic.features import get_features
from svoc.automatic.match import find_automatic_matches
from svoc.automatic.models import Distance
from svoc.supervised.enums import SupervisedModel
from svoc.supervised.match import find_supervised_matches
from svoc.constants import DEFAULT_DISTANCES, DistanceMethod


def get_matches_with_blocking(
        df_benchmark: pd.DataFrame, 
        df_input: pd.DataFrame, 
        distances: list[Distance], 
        filters: list[DistanceMethod], 
        block_col: str | None = None, 
        n_groups: int = 15, 
        n_matches: int = 3, 
        verbose: bool = True,
        models_path_dict: dict[SupervisedModel, Path] | None = None,
        window: int = 1, 
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete matching pipeline using blocking strategy.
    
    Coordinates the full matching workflow with blocking-based group partitioning:
    1. Splits benchmark data into balanced groups by blocking column
    2. For each group:
       - Filters relevant input records
       - Computes similarity features
       - Applies automatic rule-based matching
       - Applies supervised model-based matching to remaining pairs
    3. Combines and ranks all matches across groups
    4. Returns top-N matches per benchmark record
    
    Args:
        df_benchmark: Benchmark DataFrame with reference outlet records. Index is used as ID.
        df_input: Input DataFrame with outlets to match. Index is used as ID.
        distances: List of Distance objects defining similarity features to compute
        filters: List of DistanceFilter objects for automatic matching
        block_col: Column name for blocking/grouping. If None, creates dummy blocking
                  (processes all pairs - expensive!). Default: None
        n_groups: Number of groups to split benchmark data into. Default: 15
        n_matches: Maximum number of matches to return per benchmark record. Default: 3
        verbose: If True, print progress and match statistics. Default: True
        models_path_dict: Dictionary mapping SupervisedModel to model file paths. Default: None
        window: Window size for sorted neighbourhood blocking. Default: 1
        
    Returns:
        tuple containing:
        - all_matches: DataFrame of top-N matched pairs with scores, ranks, and metadata
        - all_features: DataFrame of all computed features for candidate pairs
        - remaining_features: DataFrame of unmatched candidate pairs
        
    Raises:
        ValueError: If block_col is specified but not found in DataFrames
    """
    
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
        matches_auto, remaining_features = find_automatic_matches(filters, features, n=n_matches, verbose=verbose)
        matches_supervised, remaining_features = find_supervised_matches(
            remaining_features, 
            models_path_dict=models_path_dict)
        
        l_all_matches.append(matches_auto)
        l_all_matches.append(matches_supervised)
        l_features.append(features)
        l_remaining_features.append(remaining_features)

    if not l_all_matches:
        print("⚠️ No matches found")
    else:
        all_matches = (
            concat_l(l_all_matches)
            .sort_values(by=['ID_1', 'ID_filter', 'score'], ascending=[True, True, False], na_position='last')
            .assign(rank=lambda x: x.groupby('ID_1').cumcount() + 1)
        )
        all_matches = all_matches[all_matches['rank'] <= n_matches]

    return all_matches, concat_l(l_features), concat_l(l_remaining_features)


def get_matches_with_clusters(
        df_benchmark: pd.DataFrame, 
        df_input: pd.DataFrame, 
        distances: list[Distance], 
        filters: list[DistanceMethod], 
        block_col: str,
        groups: dict | None = None, 
        n_matches: int = 3, 
        verbose: bool = True,
        models_path_dict: dict[SupervisedModel, Path] | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete matching pipeline using geographic clustering strategy.
    
    Coordinates the full matching workflow using pre-computed neighborhood groups
    (typically from kNN clustering). This provides better performance than simple
    blocking by intelligently grouping geographically nearby records.
    
    Workflow:
    1. For each benchmark postcode and its neighborhood:
       - Filters benchmark records for the postcode
       - Filters input records for the neighborhood postcodes
       - Computes similarity features for these pairs
       - Applies automatic rule-based matching
       - Applies supervised model-based matching to remaining pairs
    2. Combines and ranks all matches across neighborhoods
    3. Returns top-N matches per benchmark record
    
    Args:
        df_benchmark: Benchmark DataFrame with reference outlet records. Index is used as ID.
        df_input: Input DataFrame with outlets to match. Index is used as ID.
        distances: List of Distance objects defining similarity features to compute
        filters: List of DistanceFilter objects for automatic matching
        block_col: Column name used in groups dictionary (typically 'POSTCODE')
        groups: Dictionary mapping each value to its neighbors.
               Format: {value: [neighbor1, neighbor2, ...]}.
               Typically from svoc_knn function. Required. Default: None
        n_matches: Maximum number of matches to return per benchmark record. Default: 3
        verbose: If True, print progress and match statistics. Default: True
        models_path_dict: Dictionary mapping SupervisedModel to model file paths. Default: None
        
    Returns:
        tuple:
        - all_matches: DataFrame of top-N matched pairs with scores, ranks, and metadata
        - all_features: DataFrame of all computed features for candidate pairs
        - remaining_features: DataFrame of unmatched candidate pairs
        
    Raises:
        ValueError: If groups is None (required parameter)
        
    Note:
        Prints matching statistics including total, matched, and unmatched IDs.
    """
    
    if groups is None:
        raise ValueError("groups parameter must be provided for clustering-based matching.")

    l_all_matches = []
    l_features = []
    l_remaining_features = []

    for pc, neighbors in tqdm(groups.items()):
        
        if verbose:
            print(f"{pc}: {neighbors}")       
        
        df_x_filtered = df_benchmark[df_benchmark[block_col].isin([pc])]#.drop_duplicates()
        df_y_filtered = df_input[df_input[block_col].isin(neighbors)]#.drop_duplicates()
        
        if (df_x_filtered.empty or df_y_filtered.empty):
            if verbose:
                print(f"⚠️ Skipping postcode {pc} due to empty benchmark or input group.")
            continue # skip empty groups
            
        features = get_features(distances, df_x=df_x_filtered, df_y=df_y_filtered, njobs=1)
        matches_auto, remaining_features = find_automatic_matches(filters, features, n=n_matches, verbose=verbose)
        matches_supervised, remaining_features = find_supervised_matches(
            remaining_features, 
            models_path_dict=models_path_dict
            )
        
        l_all_matches.append(matches_auto)
        l_all_matches.append(matches_supervised)
        l_features.append(features)
        l_remaining_features.append(remaining_features)

    if not l_all_matches:
        print("⚠️ No matches found")
    else:
        all_matches = (
            concat_l(l_all_matches)
            .sort_values(by=['ID_1', 'ID_filter', 'score'], ascending=[True, True, False], na_position='last')
            .assign(rank=lambda x: x.groupby('ID_1').cumcount() + 1)
        )
        all_matches = all_matches[all_matches['rank'] <= n_matches]

        print(
f"""""
Total benchmark IDs: {df_benchmark.shape[0]}
Matched benchmark IDs: {all_matches[['ID_1']].drop_duplicates().shape[0]}
Un-matched benchmark IDs: {df_benchmark[~df_benchmark.index.isin(all_matches['ID_1'].drop_duplicates())].shape[0]}
"""""
)


    return all_matches, concat_l(l_features), concat_l(l_remaining_features)


def prepare_output( 
        matches: pd.DataFrame,
        distances: list[Distance],
        filters: list[DistanceMethod]
    ) -> pd.DataFrame:
    """Format matching results into standardized output structure.
    
    Transforms raw matching results into a clean, standardized output format by:
    1. Processing automatic matches: For each filter that produced matches,
       extracts the relevant similarity scores and methods used
    2. Processing supervised matches: Adds default distance scores and methods
    3. Renaming columns to remove '_CLEAN' suffix for clarity
    4. Sorting results by benchmark ID and rank
    
    The output includes both the match metadata (rank, score, type) and the
    specific similarity scores and methods that led to each match.
    
    Args:
        matches: DataFrame containing raw matching results with columns:
                - ID_1, ID_2: Matched record pair IDs
                - ID_filter: Filter number (for automatic matches) or NaN (for supervised)
                - rank: Match rank within ID_1 group
                - score: Overall match confidence score
                - match_type: 'auto' or 'supervised'
                - model: Model name (for supervised matches)
                - Feature columns (similarity scores)
        distances: List of all Distance objects used in matching
        filters: List of all DistanceFilter objects used in automatic matching
        
    Returns:
        DataFrame with standardized output format containing:
        - ID_1, ID_2: Matched record IDs
        - ID_filter: Filter ID or NaN
        - rank: Match rank (1 to n_matches)
        - score: Overall match confidence
        - match_type: 'auto' or 'supervised'
        - model: Model name (for supervised matches)
        - {field}_score: Similarity score for each field used in matching
        - {field}_method: Distance method used for each field
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If required columns are missing from matches DataFrame
        
    Note:
        Column names are cleaned by removing '_CLEAN' suffixes to improve readability
        in the final output.
        
    Example:
        >>> output = prepare_output(
        ...     matches=all_matches,
        ...     distances=DISTANCES,
        ...     filters=FILTERS_AUTO
        ... )
        >>> # Output columns: ['ID_1', 'ID_2', 'rank', 'score', 'match_type',
        >>> #                  'OUTLET_NAME_score', 'OUTLET_NAME_method',
        >>> #                  'ADDRESS_score', 'ADDRESS_method', ...]
    """
    if not isinstance(matches, pd.DataFrame):
        raise TypeError(
            f"matches must be a pandas DataFrame, got {type(matches).__name__}"
        )
    if not isinstance(distances, list):
        raise TypeError(
            f"distances must be a list, got {type(distances).__name__}"
        )
    if not isinstance(filters, list):
        raise TypeError(
            f"filters must be a list, got {type(filters).__name__}"
        )
    
    required_cols = ['ID_1', 'ID_2', 'ID_filter', 'rank', 'score', 'match_type']
    missing_cols = [col for col in required_cols if col not in matches.columns]
    if missing_cols:
        raise ValueError(
            f"matches DataFrame missing required columns: {missing_cols}"
        )
    
    out = pd.DataFrame()
    # LABEL_TO_COL = {d.label: d.col_name_x for d in distances}
    
    LABEL_TO_COL = {
        d.label: d.col_name_x+"_"+d.col_name_y 
        if d.col_name_x != d.col_name_y 
        else d.col_name_x 
        for d in distances
        }

    LABEL_TO_DIST = {d.label: d.method.value for d in distances}
    to_keep = ['ID_1','ID_2','ID_filter','rank','score','match_type','model']
    for idx, f in enumerate(filters):
        filter = f.value
        filter_fields = list(filter.keys())
        aux = matches.loc[
                    matches['ID_filter'] == idx + 1, 
                    to_keep + filter_fields
                    ].copy()
        
        for i in range(len(filter_fields)):
            field = filter_fields[i]
            new_name = LABEL_TO_COL.get(field).replace("_CLEAN", "")
            aux = aux.rename(columns={field: new_name+'_score'})
            aux[new_name+"_method"]= LABEL_TO_DIST.get(field)
        
        out = pd.concat([out, aux], axis=0, ignore_index=True)

    LABEL_TO_COL = {d.label: d.col_name_x for d in DEFAULT_DISTANCES}
    LABEL_TO_DIST = {d.label: d.method.value for d in DEFAULT_DISTANCES}
    
    columns_method = [c+'_method' for c in list(LABEL_TO_COL.values())]
    aux = (matches.loc[
        matches['ID_filter'].isna(), 
        to_keep+list(LABEL_TO_COL.keys())
        ]
        .rename(columns={d.label: d.col_name_x+'_score' for d in DEFAULT_DISTANCES})
        .copy())
    aux.loc[:, columns_method]=list(LABEL_TO_DIST.values())

    out = pd.concat([out, aux], axis=0, ignore_index=True)
    out.sort_values(by=['ID_1', 'rank'], ascending=[True, True], na_position='last', inplace=True)

    return out

