import pandas as pd
import numpy as np
import recordlinkage as rl

def initialize_compare_cl(l_compare, n_jobs_param):
    compare_cl = rl.Compare(n_jobs=n_jobs_param)
    for element in l_compare:
        if 'method' in element:
            compare_cl.string(element['col_name'], element['col_name'], label=element['label'],
                              method=element['method'], missing_value=0)
        else:
            compare_cl.exact(element['col_name'], element['col_name'], label=element['label'], missing_value=0)
    return compare_cl

def rl_compare_block(df_1, df_2, compare_cl, block_variable):
    indexer = rl.Index()
    indexer.block(block_variable)
    candidate_links = indexer.index(df_1, df_2)
    features = compare_cl.compute(candidate_links, df_1, df_2)
    features = features.fillna(0.0)
    return features

def arrange_features(features, df_1, df_2, index_x, index_y):
    original_columns_list = features.columns.tolist()

    features.reset_index(level=[index_x, index_y], inplace=True)
    features = features.merge(df_1, how='left', left_on=index_x, right_on="ID")
    features = features.merge(df_2, how='left', left_on=index_y, right_on="ID")

    features['outlet_name_in'] = 1 * np.array(
        [(str(x) in str(y)) or (str(y) in str(x)) if pd.notna(x) and pd.notna(y) else 0 for x, y in
         zip(features['OUTLET_NAME_x'], features['OUTLET_NAME_y'])])
    features['outlet_name_clean_in'] = 1 * np.array(
        [(str(x) in str(y)) or (str(y) in str(x)) if pd.notna(x) and pd.notna(y) else 0 for x, y in
         zip(features['OUTLET_NAME_CLEAN_x'], features['OUTLET_NAME_CLEAN_y'])])
    features['address_in'] = 1 * np.array(
        [(str(x) in str(y)) or (str(y) in str(x)) if pd.notna(x) and pd.notna(y) else 0 for x, y in
         zip(features['ADDRESS_x'], features['ADDRESS_y'])])
    features['address_clean_in'] = 1 * np.array(
        [(str(x) in str(y)) or (str(y) in str(x)) if pd.notna(x) and pd.notna(y) else 0 for x, y in
         zip(features['ADDRESS_CLEAN_x'], features['ADDRESS_CLEAN_y'])])

    def contains_all_words(x, y):
        if pd.isna(x) or pd.isna(y):
            return 0
        w1 = set(str(x).split())
        w2 = set(str(y).split())
        return int(w1.issubset(w2) or w2.issubset(w1))

    features['outlet_name_in2'] = [
        contains_all_words(x, y)
        for x, y in zip(features['OUTLET_NAME_x'], features['OUTLET_NAME_y'])
    ]
    features['outlet_name_clean_in2'] = [
        contains_all_words(x, y)
        for x, y in zip(features['OUTLET_NAME_CLEAN_x'], features['OUTLET_NAME_CLEAN_y'])
    ]
    features['address_in2'] = [
        contains_all_words(x, y)
        for x, y in zip(features['ADDRESS_x'], features['ADDRESS_y'])
    ]
    features['address_clean_in2'] = [
        contains_all_words(x, y)
        for x, y in zip(features['ADDRESS_CLEAN_x'], features['ADDRESS_CLEAN_y'])
    ]

    
    final_columns_list = [index_x, index_y] + original_columns_list + \
        ['outlet_name_in', 'outlet_name_clean_in', 'address_in', 'address_clean_in'] + \
        ['outlet_name_in2', 'outlet_name_clean_in2', 'address_in2', 'address_clean_in2']

    return features[final_columns_list]

def get_features(distances_dict, df_x, df_y, block_col):
    compare_cl = initialize_compare_cl(distances_dict, n_jobs_param=-1)
    features = rl_compare_block(df_x, df_y, compare_cl, block_col)
    features = arrange_features(features, df_1=df_x, df_2=df_y, index_x="ID_1", index_y="ID_2")
    return features