import pandas as pd
from tqdm import tqdm
from svoc.datapreparation import split_df
from svoc.utils import concat_l
from svoc.automatic.features import get_features
from svoc.automatic.models import Distance
from svoc.constants import DistanceMethod

def filter_dataframe(df, dict_constraints):
    for column_name, threshold in dict_constraints.items():
        if column_name in df.columns:
            df = df[df[column_name] > threshold].copy()
        else:
            print(f"Warning: Column '{column_name}' not found in DataFrame.")
    return df

def norm_score(df, score_cols):
    score_cols = list(score_cols)
    for column_name in score_cols:
        if column_name not in df.columns:
            if column_name != 'filter_threshold':
                print(f"Warning: Column '{column_name}' not found in DataFrame.")
            score_cols.remove(column_name)
    score = df[score_cols].mean(axis=1)
    return score

def check_matches(df_match, filter_label):
    ids_1 = len(df_match['ID_1'].drop_duplicates())
    ids_2 = len(df_match['ID_2'].drop_duplicates())
    print(f"Filter {filter_label}: {ids_1} IDs have been matched with {ids_2} IDs from the input dataset.")

def find_automatic_matches(
        filters: list[DistanceMethod], 
        features: pd.DataFrame, 
        n: int = 3, 
        verbose: bool = True
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    l_matches = []
    missing_matches = features[["ID_1"]].drop_duplicates().copy()
    missing_matches['counter'] = n
    for i, f in enumerate(filters):
        filter = f.value
        # find matches
        matches_filter_i = filter_dataframe(features, filter)
        if matches_filter_i.empty:
            if verbose:
                print(f"Filter {i}: Any match found")
            continue
        else:
            matches_filter_i['ID_filter'] = i + 1
            matches_filter_i['score'] = norm_score(matches_filter_i, filter.keys())
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

            # Tolgo ID_1 che hanno già raggiunto il numero di match richiesti
            full_matched = match_count[match_count>=n].index.tolist()
            features = features[~(features['ID_1'].isin(full_matched))]
            # Tolgo le coppie già matchate
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
        filters: list[DistanceMethod], 
        block_col: str|None = None, 
        n_groups: int = 15, 
        n_matches: int = 3, 
        verbose: bool = True,
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
        l_features.append(features)

        matches_auto, remaining_features = find_automatic_matches(filters, features, n=n_matches, verbose=verbose)
        l_all_matches.append(matches_auto)
        l_remaining_features.append(remaining_features)

    if not l_all_matches:
        print("⚠️ No matches found")

    return concat_l(l_all_matches), concat_l(l_features), concat_l(l_remaining_features)

