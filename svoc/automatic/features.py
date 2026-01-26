import pandas as pd
import numpy as np
from recordlinkage import Compare, Index
from svoc.automatic.enums import DistanceMethod
from svoc.automatic.models import Distance

def initialize_compare_cl(
        l_compare: list[Distance], 
        n_jobs_param: int = -1
    ) -> Compare:
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
        block_variable: str|None = None
    ) -> pd.DataFrame:
    indexer = Index()
    if block_variable is not None:
        indexer.block(block_variable)
    candidate_links = indexer.index(df_1, df_2)
    features = compare_cl.compute(candidate_links, df_1, df_2)
    features = features.fillna(0.0)
    return features

def vec_substring_matching(
        column_x: pd.Series, 
        column_y: pd.Series
    ) -> pd.Series:
    
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
        block_col: str|None = None
    ) -> pd.DataFrame:
    compare_cl = initialize_compare_cl(distances, n_jobs_param=-1)
    features = rl_compare_block(df_x, df_y, compare_cl, block_col)
    features = manual_features(distances, features, df_x, df_y, index_x="ID_1", index_y="ID_2")
    return features