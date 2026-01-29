import pandas as pd
from svoc.datapreparation import split_df
from svoc.utils import concat_l
from svoc.automatic.features import get_features
from svoc.automatic.match import find_automatic_matches
from svoc.automatic.models import Distance
from svoc.supervised.enums import SupervisedModel
from svoc.supervised.match import find_supervised_matches
from tqdm import tqdm
from svoc.constants import DEFAULT_DISTANCES, DistanceMethod
from pathlib import Path

def get_matches(
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
        ):
    
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


def prepare_output( 
        matches: pd.DataFrame,
        distances: list[Distance],
        filters: list[DistanceMethod]
    ):
    
    out = pd.DataFrame()
    LABEL_TO_COL = {d.label: d.col_name for d in distances}
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

    LABEL_TO_COL = {d.label: d.col_name for d in DEFAULT_DISTANCES}
    LABEL_TO_DIST = {d.label: d.method.value for d in DEFAULT_DISTANCES}
    
    columns_method = [c+'_method' for c in list(LABEL_TO_COL.values())]
    aux = (matches.loc[
        matches['ID_filter'].isna(), 
        to_keep+list(LABEL_TO_COL.keys())
        ]
        .rename(columns={d.label: d.col_name+'_score' for d in DEFAULT_DISTANCES})
        .copy())
    aux.loc[:, columns_method]=list(LABEL_TO_DIST.values())

    out = pd.concat([out, aux], axis=0, ignore_index=True)
    out.sort_values(by=['ID_1', 'rank'], ascending=[True, True], na_position='last', inplace=True)

    return out
