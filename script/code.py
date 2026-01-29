
# import importlib
# importlib.reload()
from svoc.settings import get_settings
from svoc.utils import read_data
from svoc.datapreparation import prepare_data, make_upper_str, rename_and_select_cols
from svoc.automatic.match import get_automatic_matches
from svoc.supervised.match import predict_supervised
from svoc.rl import get_matches, prepare_output
import pandas as pd
import numpy as np
from svoc.constants import DISTANCES, FILTERS_AUTO


settings = get_settings()
# settings = get_settings("./config/dev2.yaml")

df_input, df_benchmark = read_data(settings)

# ## Modifico Location SAP (TEST)
# df_loc = pd.read_csv('./data/HUK_sap_location.csv', sep=',', dtype=str)
# cols_to_join = ["AddressLine1", "AddressLine2", "AddressLine3", "AddressLine4"]
# df_loc[settings.BENCHMARK_COLUMNS.ADDRESS] = df_loc[cols_to_join].apply(
#     lambda x: ', '.join(x.dropna().astype(str)), 
#     axis=1
# ) 
# df_benchmark = df_benchmark.drop(columns=[settings.BENCHMARK_COLUMNS.ADDRESS]).merge(
#     df_loc[["CustomerCode", settings.BENCHMARK_COLUMNS.ADDRESS]],
#     left_on=settings.BENCHMARK_COLUMNS.ID, right_on="CustomerCode",
#     how='left'
# ).drop(columns=['CustomerCode'])

## TODO 
# Prendo tutte le coppie di postalcode e misuro le differenze. Se la differenza è piccola considero il postal code uguale
# prova a usare index.sortedneighbourhood()
# https://recordlinkage.readthedocs.io/en/latest/ref-index.html#recordlinkage.index.SortedNeighbourhood

#----------------------------------------------------------
## Data Preparation

## SAP Benchmark - BOWIMI Input(dev.yaml)
df_benchmark_clean = prepare_data(
    df=df_benchmark, dict_cols=settings.BENCHMARK_COLUMNS_DICT)
df_input_clean = prepare_data(
    df=df_input, dict_cols=settings.INPUT_COLUMNS_DICT)

df_inner = (df_input_clean
            .merge(
                make_upper_str(df_input[[settings.INPUT_COLUMNS.ID, "sapcode"]]).replace(['NAN', 'NONE'], np.nan), 
                left_on='ID', right_on=settings.INPUT_COLUMNS.ID, 
                how='left'
                )
                .merge(
                    df_benchmark_clean, 
                    left_on="sapcode", right_on='ID', 
                    how='inner', suffixes=('_input', '_benchmark'))
            )

# ## BOWIMI Benchmark - CGA Input (dev2.yaml)
# df_benchmark_clean = prepare_data(
#     df=df_benchmark, dict_cols=settings.BENCHMARK_COLUMNS_DICT,
#     parse_address=True, get_town=False, rm_address_noise=True)
# df_input_clean = prepare_data(
#     df=df_input, dict_cols=settings.INPUT_COLUMNS_DICT,
#     parse_address=False, get_town=False, rm_address_noise=True)


#----------------------------------------------------------

## Auto + Superv Matching wrapper
all_matches, features, remaining_features = get_matches(
    df_input=df_input_clean, 
    df_benchmark=df_benchmark_clean, 
    block_col=settings.BLOCK_COL, 
    distances=DISTANCES, 
    filters=FILTERS_AUTO,
    # n_groups = 1, window = 3,
    n_groups = 15, window = 1,
    n_matches=settings.N_MATCHES, verbose=False,
    models_path_dict=settings.SUPERVISED_MODELS_PATHS
    )

output = prepare_output(
    matches=all_matches,
    distances=DISTANCES,
    filters=FILTERS_AUTO
)

missing_matches = (make_upper_str(df_benchmark)
 .merge(output[["ID_1"]].drop_duplicates(), 
        left_on = settings.BENCHMARK_COLUMNS.ID, 
        right_on = "ID_1",
        how="left",
        indicator=True)
).query("_merge == 'left_only'") \
 .drop(columns="_merge")

output.to_excel("./data/output_1.xlsx", index=False)

#----------------------------------------------------------
## Automatic Matching
all_matches_auto, features, remaining_features = get_automatic_matches(
    df_input=df_input_clean, 
    df_benchmark=df_benchmark_clean, 
    block_col=settings.BLOCK_COL, 
    distances=DISTANCES, 
    filters=FILTERS_AUTO,
    n_groups=15, n_matches=settings.N_MATCHES, verbose=False)

all_matches_auto = (
    all_matches_auto
        .sort_values(by=['ID_1', 'score'], ascending=[True, False])
        .assign(rank=lambda x: x.groupby('ID_1').cumcount() + 1)
)

## Check filters
all_matches_auto['ID_filter'].value_counts().sort_index()
[FILTERS_AUTO[i-1] for i in all_matches_auto['ID_filter'].drop_duplicates().sort_values().tolist()]

all_matches_auto

#--------------------------------------------------------------------------------------------------------
## Supervised Matching

from svoc.utils import concat_l
from svoc.supervised.match import SupervisedModel

test_features = (remaining_features
                 .drop(columns=[settings.BLOCK_COL.lower()])
                 .set_index(["ID_1","ID_2"])
                 .copy())

all_matches_supervised_l = []
features_superv = test_features.copy()

for mdl in SupervisedModel:
    matches_supervised = predict_supervised(features_superv, model=mdl)
    features_superv = features_superv.loc[~features_superv.index.isin(matches_supervised.index)]
    all_matches_supervised_l.append(matches_supervised.reset_index())

all_matches_supervised = (test_features
                      .reset_index()
                      .merge(
                          concat_l(all_matches_supervised_l), 
                          on=['ID_1','ID_2'], 
                          how='inner'
                          ))

all_matches = pd.concat([all_matches_auto, all_matches_supervised], axis=0, ignore_index=True)

#--------------------------------------------------------------------------------------------------------

def prepare_output(
        all_matches: pd.DataFrame, 
        df_benchmark: pd.DataFrame, 
        df_input: pd.DataFrame,
        benchmark_cols: dict,
        input_cols: dict,
        labels: list = ("_benchmark", "_input"),
        keep_features: bool = False
    ):

    df_benchmark = rename_and_select_cols(df_benchmark, benchmark_cols)
    df_input = rename_and_select_cols(df_input, input_cols)
    all_matches[['ID_1']] = all_matches[['ID_1']].apply(lambda col: col.str.lower(), axis=1)
    all_matches[['ID_2']] = all_matches[['ID_2']].apply(lambda col: col.str.lower(), axis=1)
    df_input[['ID']] = df_input[['ID']].apply(lambda col: col.str.lower(), axis=1)
    df_benchmark[['ID']] = df_benchmark[['ID']].apply(lambda col: col.str.lower(), axis=1)
    
    if not keep_features:
        all_matches = all_matches[['ID_1', 'ID_2', 'ID_filter', 'rank', "score" ,"match_type", "model"]]

    df_match_final = (all_matches
                    .merge(
                         df_benchmark,
                         left_on='ID_1',
                         right_on='ID'
                          )
                    .drop(columns=['ID'])
                    .merge(
                        df_input,
                        left_on='ID_2',
                        right_on='ID', 
                        suffixes=labels
                        )
                    .drop(columns=['ID'])
                    .rename(columns={'ID_1': benchmark_cols['ID'], 'ID_2': input_cols['ID']})
                    )

    return df_match_final

df_match_final = prepare_output(
    all_matches=output,
    df_benchmark=df_benchmark,
    df_input=df_input,
    benchmark_cols=settings.BENCHMARK_COLUMNS_DICT,
    input_cols=settings.INPUT_COLUMNS_DICT,
    # labels=('_sap', '_bowimi'),
    keep_features=True
)

f = 1
output[output["ID_filter"]==f]
df_match_final[df_match_final["ID_filter"]==f][["OUTLET_NAME_input", "OUTLET_NAME_benchmark","rank","score"]]

# df_match_final.to_excel("./data/match_final1.xlsx", index=False)
# from svoc.utils import save_pickle
# save_pickle(df_match_final, "./data/match_final.pkl")

# CONTROLLO RISULTATI ---------------------------------------------------------------------------------------------------------

df_match_final=df_match_final.rename(columns={settings.BENCHMARK_COLUMNS_DICT['ID']: 'ID_benchmark', settings.INPUT_COLUMNS_DICT['ID']: 'ID_input'})
# df_match_final=df_match_final.rename(columns={'BowimiId': 'ID_benchmark', 'CgaId': 'ID_input'})

df_match_final[['OUTLET_NAME_input', 'OUTLET_NAME_benchmark','rank', 'ID_filter']]
df_match_final[['ADDRESS_input', 'ADDRESS_benchmark','rank', 'ID_filter']]

(df_benchmark_clean.reset_index())["ID"].nunique()
df_match_final["ID_benchmark"].nunique()

## Check multiple matches
n_matches = all_matches.groupby('ID_1').size().reset_index(name='num_matches').sort_values(by='num_matches', ascending=False)
multiple_matches = df_match_final[
    df_match_final["ID_benchmark"].isin(
        n_matches[n_matches['num_matches'] > 1]['ID_1']
        )
    ]

multiple_matches[['OUTLET_NAME_benchmark','OUTLET_NAME_input','rank']]
multiple_matches[['ADDRESS_benchmark', 'ADDRESS_input','rank']]


## Controllo vs. match certi
totali=(df_inner
        .rename(columns={'sapcode': 'ID_benchmark', 'bowimiid': 'ID_input'})
        .merge(
            make_upper_str(df_match_final[["ID_benchmark","ID_input","rank"]]),
            on = ["ID_benchmark"],
            suffixes=('','_auto'),
            how='outer'
        ))


nmatches=totali.groupby('ID_benchmark').size()
multiple_matches[multiple_matches["ID_benchmark"].isin(nmatches[nmatches>1].index.tolist())][["ID_benchmark","OUTLET_NAME_benchmark","OUTLET_NAME_input","rank"]]
multiple_matches[multiple_matches["ID_benchmark"].isin(nmatches[nmatches>1].index.tolist())][["ID_benchmark","ADDRESS_benchmark","ADDRESS_input","rank"]]


# Corretti
c=totali[~totali['ID_input_auto'].isna() & ~totali['ID_input'].isna() & (totali['ID_input_auto'] == totali['ID_input'])]

# Non trovati
m = ((totali[totali['ID_input_auto'].isna()][['ID_input','ID_benchmark']])
 .merge(
     df_benchmark_clean.reset_index(),
     how="left", left_on = ["ID_benchmark"], right_on = ["ID"]
 )
 .drop(columns=['ID'])
 .merge(
    df_input_clean.reset_index(),
    how="left", left_on = ["ID_input"], right_on = ["ID"],
    suffixes=('_benchmark','_input'))
 .drop(columns=['ID']
 ))

m[["POSTCODE_input",'OUTLET_NAME_CLEAN_input',
   "POSTCODE_benchmark",'OUTLET_NAME_CLEAN_benchmark']]
m[["POSTCODE_input",'ADDRESS_CLEAN_input',
   "POSTCODE_benchmark",'ADDRESS_CLEAN_benchmark']]
m
## PostCode SBAGLIATO
m[m["POSTCODE_input"]!=m["POSTCODE_benchmark"]].to_excel("./data/m_postcode_wrong.xlsx", index=False)

m[m["POSTCODE_input"]!=m["POSTCODE_benchmark"]][["ADDRESS_CLEAN_input",'ADDRESS_CLEAN_benchmark']]
m[m["POSTCODE_input"]!=m["POSTCODE_benchmark"]][["OUTLET_NAME_CLEAN_input",'OUTLET_NAME_CLEAN_benchmark']]
## PostoCode GIUSTO
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ADDRESS_CLEAN_input",'ADDRESS_CLEAN_benchmark']]
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["OUTLET_NAME_CLEAN_input",'OUTLET_NAME_CLEAN_benchmark']]

m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_input","POSTCODE_input","OUTLET_NAME_CLEAN_input","ID_benchmark","POSTCODE_benchmark",'OUTLET_NAME_CLEAN_benchmark']]
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_input","ADDRESS_CLEAN_input","ID_benchmark",'ADDRESS_CLEAN_benchmark']]
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_input","OUTLET_NAME_input","ID_benchmark",'OUTLET_NAME_benchmark']]
mfeat = m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_benchmark","ID_input"]].merge(
    features,
    how="left", right_on=["ID_1","ID_2"], left_on=["ID_benchmark","ID_input"]
)
mfeat.iloc[0]

num = mfeat.select_dtypes(include='number')
(
   num[num > 0.8] 
    .stack()
    .reset_index()
    .rename(columns={'level_0': 'row_id', 'level_1': 'column', 0: 'value'})
)

mfeat[mfeat['address_clean_levenshtein']>.8]['outlet_name_clean_cosine']

# Sbagliati
s = (totali[
    ~totali['ID_input_auto'].isna() 
    & ~totali['ID_input'].isna() 
    & (totali['ID_input_auto'] != totali['ID_input'])
    & ~totali["ID_benchmark"].isin(c["ID_benchmark"])
    ])[['ID_benchmark','ID_input','ID_input_auto']]

s=(s
   .merge(
     df_benchmark_clean.reset_index(),
     how="left", left_on = ["ID_benchmark"], right_on = ["ID"]
     )
    .drop(columns=['ID'])
   .merge(
    df_input_clean.reset_index(),
    how="left", left_on = ["ID_input"], right_on = ["ID"],
    suffixes=('_benchmark','')
    )
    .drop(columns=['ID'])
    .merge(
     df_input_clean.reset_index(),
     how="left", left_on = ["ID_input_auto"], right_on = ["ID"],
     suffixes=('_input_GIUSTO','_input_SBAGLIATO')
     ))

s.columns
s[["POSTCODE_benchmark",
   'OUTLET_NAME_benchmark',
   "POSTCODE_input_GIUSTO",
   'OUTLET_NAME_input_GIUSTO',
   "POSTCODE_input_SBAGLIATO",
   'OUTLET_NAME_input_SBAGLIATO']]
s[["POSTCODE_benchmark",
   'ADDRESS_benchmark',
   #"POSTCODE_input_GIUSTO",
   'ADDRESS_input_GIUSTO',
   #"POSTCODE_input_SBAGLIATO",
   'ADDRESS_input_SBAGLIATO']]

## Quanti hanno postcode matchato?
s[s["POSTCODE_input_GIUSTO"]==s["POSTCODE_benchmark"]]

# Trovati in più
totali[totali['ID_input'].isna()] 
totali[totali['ID_input'].isna()]["ID_benchmark"].nunique()

#---------------------------------------------------------------------------------------------------------------------------
# Check su quelli non matchati
matched_indexes = all_matches_auto['ID_1'].drop_duplicates().tolist()
df_benchmark_unmatched = df_benchmark_clean[~df_benchmark_clean.index.isin(matched_indexes)]
features[features['ID_1'].isin(df_benchmark_unmatched.index)]
features[features['ID_1'].isin(matched_indexes)]
# Non ci sono gli id in features, perchè?

# Controllo che i podtocede esistano in input
df_benchmark_unmatched.index.nunique()
counts_postcode=(df_benchmark_unmatched["POSTCODE"].value_counts().reset_index().sort_values(by='count', ascending=False))
counts_postcode
df_input_clean[df_input_clean["POSTCODE"].isin(counts_postcode["POSTCODE"].tolist())]

## Vedo se tolgo i filtri extremi che succ
all_matches_auto['ID_filter'].value_counts().sort_index()
all_matches_auto_filtered = all_matches_auto[all_matches_auto['ID_filter']<53]
matched_indexes = all_matches_auto_filtered['ID_1'].drop_duplicates().tolist()

df_benchmark_unmatched = df_benchmark_clean[~df_benchmark_clean.index.isin(matched_indexes)]
mfeat = features[features['ID_1'].isin(df_benchmark_unmatched.index)]
mfeat["ID_1"].nunique() # 34
num = mfeat.select_dtypes(include='number')
(
   num[(num > 0.5)]# & (num < 0.6)] 
    .stack()
    .reset_index()
    .rename(columns={'level_0': 'row_id', 'level_1': 'column', 0: 'value'})
).sort_values(by="row_id")

# address_(clean)_jarowinkler > 0.7
# address_(clean)_cosine > 0.6
# outlet_name_(clean)_ jarowinkler > .6

(
    (all_matches_auto[all_matches_auto["ID_1"].isin(df_benchmark_unmatched.index)].sort_values(by=['ID_1','score'], ascending=[True, False]))
    .merge(df_benchmark_clean.reset_index(), left_on='ID_1', right_on='ID').drop(columns=['ID'])
    .merge(df_input_clean.reset_index(), left_on='ID_2', right_on='ID', suffixes=('_benchmark','_input')).drop(columns=['ID'])
 ).to_excel("./data/last_matches.xlsx", index=False) # Schifo, posso togliere i filtri

(
    all_matches_auto_filtered.merge(df_benchmark_clean.reset_index(), left_on='ID_1', right_on='ID').drop(columns=['ID'])
    .merge(df_input_clean.reset_index(), left_on='ID_2', right_on='ID', suffixes=('_benchmark','_input')).drop(columns=['ID'])
    
).to_excel("./data/matches.xlsx", index=False)

#------------------------------------------------------------
# Seleziono per ogni benchmark id il match con overall score più alto
## Schifo
labels = [d["label"] for d in settings.DISTANCES if d["label"] != "postcode"]
features["overall_score"] = features[labels].max(axis=1)
best_overall_score = (
    features
        .sort_values(by=['ID_1', 'overall_score'], ascending=[True, False])
        .groupby('ID_1')
        .first()
        .reset_index()
)
# all_matches_auto.merge(best_overall_score)

all_matches = pd.concat(
    [
        best_overall_score[~best_overall_score['ID_1'].isin(all_matches_auto["ID_1"].drop_duplicates().to_list())], 
        all_matches_auto.merge(best_overall_score)
    ], 
    axis=0, ignore_index=True)

(all_matches
 .merge(df_benchmark_clean.reset_index(), left_on='ID_1', right_on='ID').drop(columns=['ID'])
 .merge(df_input_clean.reset_index(), left_on='ID_2', right_on='ID', suffixes=('_benchmark','_input'))
 .drop(columns=['ID'])).to_excel("./data/matches.xlsx", index=False)
# %%
