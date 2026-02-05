import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors

from svoc.settings import Settings
from svoc.datapreparation import prepare_data, rename_and_select_cols, make_upper_str, remove_accents_and_regex
from svoc.rl import get_matches_with_clusters, prepare_output
from svoc.constants import DISTANCES, FILTERS_AUTO


def svoc_knn(
        settings: Settings, 
        df_input: pd.DataFrame, 
        df_benchmark: pd.DataFrame, 
        k: int=6,
        save: bool=True
        ):

    LATITUDE, LONGITUDE = 'LATITUDE', 'LONGITUDE'
    required_keys = {LATITUDE, LONGITUDE}
    if not (required_keys <= settings.BENCHMARK_COLUMNS_DICT.keys() 
            and required_keys <= settings.INPUT_COLUMNS_DICT.keys()
            ):
        raise KeyError("Missing required keys: both 'LATITUDE' and 'LONGITUDE' columns must be specified in the settings.")
    
    def prepare_data_for_knn(df: pd.DataFrame, cols: dict):
        df_out = rename_and_select_cols(df=df, dict_cols={k: cols[k] for k in [settings.BLOCK_COL, LATITUDE, LONGITUDE]})
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
        save: bool = False
        ):

    df_benchmark_clean = prepare_data(
        df=df_benchmark, 
        dict_cols=settings.BENCHMARK_CORE_COLUMNS_DICT,
        )

    df_input_clean = prepare_data(
        df=df_input, 
        dict_cols=settings.INPUT_CORE_COLUMNS_DICT,
        )

    all_matches, features, remaining_features = get_matches_with_clusters(
        df_input=df_input_clean, 
        df_benchmark=df_benchmark_clean, 
        distances=DISTANCES, 
        filters=FILTERS_AUTO,
        block_col=settings.BLOCK_COL,
        groups=groups,
        n_matches=settings.N_MATCHES, verbose=False,
        models_path_dict=settings.SUPERVISED_MODELS_PATHS
        )

    output = prepare_output(
        matches=all_matches,
        distances=DISTANCES,
        filters=FILTERS_AUTO
    )
    
    if save:
        output.to_csv(settings.DATA_DIR / 'output.csv', index=False)    
    
    return output