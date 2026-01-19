
import importlib
# importlib.reload(cons)
from svoc.datapreparation import prepare_data, make_upper_str, rename_and_select_cols
from svoc.automatic.match import get_automatic_matches
from svoc.supervised.match import train_supervised_model, predict_supervised
import svoc.constants as cons
import pandas as pd
import numpy as np
import openpyxl


df_input = pd.read_csv(cons.INPUT_FILEPATH, sep=',', dtype=str)
df_benchmark = pd.read_csv(cons.BENCHMARK_FILEPATH, sep=',', dtype=str)

## Modifico Location SAP
df_loc = pd.read_csv('./data/HUK_sap_location.csv', sep=',', dtype=str)
cols_to_join = ["AddressLine1", "AddressLine2", "AddressLine3", "AddressLine4"]
df_loc[cons.BENCHMARK_COLUMNS['ADDRESS']] = df_loc[cols_to_join].apply(
    lambda x: ', '.join(x.dropna().astype(str)), 
    axis=1
) 
df_benchmark = df_benchmark.drop(columns=[cons.BENCHMARK_COLUMNS['ADDRESS']]).merge(
    df_loc[["CustomerCode", cons.BENCHMARK_COLUMNS['ADDRESS']]],
    left_on=cons.BENCHMARK_COLUMNS['ID'], right_on="CustomerCode",
    how='left'
).drop(columns=['CustomerCode'])

#----------------------------------------------------------
## Data Preparation
df_input_clean = prepare_data(
    df=df_input, dict_cols=cons.INPUT_COLUMNS,
    parse_address=True, get_town=False, rm_address_noise=True)

df_benchmark_clean = prepare_data(df=df_benchmark, dict_cols=cons.BENCHMARK_COLUMNS, parse_address=False, rm_address_noise=True)

df_inner = (df_input_clean
            .merge(
                make_upper_str(df_input[[cons.INPUT_COLUMNS['ID'], "SapCode"]]).replace(['NAN', 'NONE'], np.nan), 
                left_on='ID', right_on=cons.INPUT_COLUMNS['ID'], 
                how='left'
                )
                .merge(
                    df_benchmark_clean, 
                    left_on=cons.BENCHMARK_COLUMNS['ID'], right_on='ID', 
                    how='inner', suffixes=('_input', '_benchmark'))
            )
#----------------------------------------------------------
## Automatic Matching
all_matches_auto, features, remaining_features = get_automatic_matches(
    df_input=df_input_clean, 
    df_benchmark=df_benchmark_clean, 
    block_col=cons.BLOCK_COL, 
    distances_dict=cons.DISTANCES, 
    filters_dict=cons.FILTERS_AUTO,
    n_groups=15, n_matches=3, verbose=False)

all_matches_auto = (
    all_matches_auto
        .sort_values(by=['ID_1', 'score'], ascending=[True, False])
        .assign(rank=lambda x: x.groupby('ID_1').cumcount() + 1)
)

## Check filters
all_matches_auto['ID_filter'].value_counts().sort_index()
[cons.FILTERS_AUTO[i-1] for i in all_matches_auto['ID_filter'].drop_duplicates().sort_values().tolist()]

all_matches_auto

#--------------------------------------------------------------------------------------------------------
## Training Supervised Models

# ## Usando i match automatici
# training_matches = (
#     [all_matches_auto["rank"]==1][["ID_1","ID_2"]]
#     .set_index(["ID_1","ID_2"])
#     .index
# )
# matched_indexes = all_matches_auto["ID_1"].drop_duplicates().tolist()

## Usando i match certi
training_matches = (
    df_inner[[cons.BENCHMARK_COLUMNS['ID'], cons.INPUT_COLUMNS['ID']]]
    .rename(columns={cons.BENCHMARK_COLUMNS['ID']: 'ID_1', cons.INPUT_COLUMNS['ID']: 'ID_2'})
    .set_index(["ID_1","ID_2"])
    .index
)
matched_indexes = df_inner[cons.BENCHMARK_COLUMNS['ID']].drop_duplicates().tolist()

training_features = (features[features["ID_1"].isin(matched_indexes)]
                     .drop(columns=[cons.BLOCK_COL.lower()])
                     .set_index(["ID_1","ID_2"])
                     .copy())
 
model_logreg = train_supervised_model(
    supervised_model='logreg',
    train_set_matches_index=training_matches,
    train_set_features=training_features,
    save=True,
    pickle_path=cons.SUPERVISED_MODEL_PATHS['logreg']
)
model_svm = train_supervised_model(
    supervised_model='svm',
    train_set_matches_index=training_matches,
    train_set_features=training_features,
    save=True,
    pickle_path=cons.SUPERVISED_MODEL_PATHS['svm']
)
model_bayes = train_supervised_model(
    supervised_model='naive-bayes',
    train_set_matches_index=training_matches,
    train_set_features=training_features,
    save=True,
    pickle_path=cons.SUPERVISED_MODEL_PATHS['naive-bayes']
)

#--------------------------------------------------------------------------------------------------------
## Supervised Matching

# test_features = (features[~features["ID_1"].isin(matched_indexes)]
#                  .drop(columns=["postcode"])
#                  .set_index(["ID_1","ID_2"])
#                  .copy())

test_features = (remaining_features
                 .drop(columns=[cons.BLOCK_COL.lower()])
                 .set_index(["ID_1","ID_2"])
                 .copy())

matches_pred_logreg = predict_supervised(test_features, cons.SUPERVISED_MODEL_PATHS['logreg'])
matches_pred_svm = predict_supervised(test_features, cons.SUPERVISED_MODEL_PATHS['svm'])
matches_pred_bayes = predict_supervised(test_features, cons.SUPERVISED_MODEL_PATHS['naive-bayes'])


all_matches_pred = test_features.loc[matches_pred_logreg].copy()
all_matches_pred["score"] = all_matches_pred.mean(axis=1)
all_matches_pred["match_type"] = "supervised"
all_matches_pred = (
    all_matches_pred
        .reset_index()
        .sort_values(by=['ID_1', 'score'], ascending=[True, False])
        .assign(rank=lambda x: x.groupby('ID_1').cumcount() + 1)
)
all_matches_pred=all_matches_pred[all_matches_pred["rank"]<=3]


# all_matches = pd.concat(
#     [
#         auto, 
#         all_matches_pred
#     ]
#     , axis=0, ignore_index=True).sort_values(by=['ID_1', 'ID_2'])

## Quali sono state mathcate dai modelli ma non dai match automatici (e viceversa)?

auto = all_matches_auto[~all_matches_auto["ID_1"].isin(matched_indexes)][["ID_1","ID_2", "rank", "match_type"]]
pred = all_matches_pred[["ID_1","ID_2", "match_type"]]
matches_cfr = auto.merge(pred, on=["ID_1","ID_2"], how='outer', suffixes=('_auto','_pred'))

matches_cfr = (
    matches_cfr
    .merge(
            make_upper_str(rename_and_select_cols(df_benchmark, cons.BENCHMARK_COLUMNS)),
            left_on='ID_1',
            right_on='ID'
            )
    .drop(columns=['ID'])
    .merge(
        make_upper_str(rename_and_select_cols(df_input, cons.INPUT_COLUMNS)),
        left_on='ID_2',
        right_on='ID', 
        suffixes=('_benchmark', '_input'))
    .drop(columns=['ID'])
    .rename(columns={'ID_1': 'ID_benchmark', 'ID_2': 'ID_input'})
)
# matches_cfr.to_excel("./data/cfr_matches.xlsx", index=False)
(matches_cfr[matches_cfr["match_type_pred"].isna() & (matches_cfr["rank"]==1)][["ID_benchmark","ID_input"]]
 .merge(
     all_matches_auto,
     right_on=['ID_1','ID_2'],
    left_on=['ID_benchmark','ID_input']
 )).to_excel("./data/cfr_matches_3.xlsx", index=False)
matches_cfr[matches_cfr["match_type_pred"].isna() & (matches_cfr["rank"]==1)].to_excel("./data/cfr_matches.xlsx", index=False)

# (df_inner[[cons.BENCHMARK_COLUMNS['ID'], cons.INPUT_COLUMNS['ID']]]
#  .rename(columns={cons.BENCHMARK_COLUMNS['ID']: 'ID_1', cons.INPUT_COLUMNS['ID']: 'ID_2'}))
#--------------------------------------------------------------------------------------------------------


df_match_final = (all_matches_auto[['ID_1', 'ID_2', 'ID_filter', 'rank', "score" ,"match_type"]]
                .merge(
                     make_upper_str(rename_and_select_cols(df_benchmark, cons.BENCHMARK_COLUMNS)),
                      left_on='ID_1',
                      right_on='ID'
                      )
                .drop(columns=['ID'])
                .merge(
                    make_upper_str(rename_and_select_cols(df_input, cons.INPUT_COLUMNS)),
                    left_on='ID_2',
                    right_on='ID', 
                    suffixes=('_benchmark', '_input'))
                .drop(columns=['ID'])
                .rename(columns={'ID_1': 'ID_benchmark', 'ID_2': 'ID_input'})
                )



# CONTROLLO RISULTATI ---------------------------------------------------------------------------------------------------------

df_match_final[['OUTLET_NAME_input', 'OUTLET_NAME_benchmark','rank', 'ID_filter']]
df_match_final[['ADDRESS_input', 'ADDRESS_benchmark','rank', 'ID_filter']]

(df_benchmark_clean.reset_index())["ID"].nunique()
df_match_final["ID_benchmark"].nunique()

## Check multiple matches
n_matches = all_matches_auto.groupby('ID_1').size().reset_index(name='num_matches').sort_values(by='num_matches', ascending=False)
multiple_matches = df_match_final[
    df_match_final["ID_benchmark"].isin(
        n_matches[n_matches['num_matches'] > 1]['ID_1']
        )
    ]

multiple_matches[['OUTLET_NAME_benchmark','OUTLET_NAME_input','rank']]
multiple_matches[['ADDRESS_benchmark', 'ADDRESS_input','rank']]

totali=(df_inner
        .rename(columns={cons.BENCHMARK_COLUMNS['ID']: 'ID_benchmark', cons.INPUT_COLUMNS['ID']: 'ID_input'})
        .merge(
            df_match_final[["ID_benchmark","ID_input","rank"]],
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

## PostCode SBAGLIATO
m[m["POSTCODE_input"]!=m["POSTCODE_benchmark"]][["ADDRESS_CLEAN_input",'ADDRESS_CLEAN_benchmark']]
m[m["POSTCODE_input"]!=m["POSTCODE_benchmark"]][["OUTLET_NAME_CLEAN_input",'OUTLET_NAME_CLEAN_benchmark']]
## PostoCode GIUSTO
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ADDRESS_CLEAN_input",'ADDRESS_CLEAN_benchmark']]
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["OUTLET_NAME_CLEAN_input",'OUTLET_NAME_CLEAN_benchmark']]

m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_input","POSTCODE_input","OUTLET_NAME_CLEAN_input","ID_benchmark","POSTCODE_benchmark",'OUTLET_NAME_CLEAN_benchmark']]
m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_input","ADDRESS_CLEAN_input","ID_benchmark",'ADDRESS_CLEAN_benchmark']]
mfeat = m[m["POSTCODE_input"]==m["POSTCODE_benchmark"]][["ID_benchmark","ID_input"]].merge(
    features,
    how="left", right_on=["ID_1","ID_2"], left_on=["ID_benchmark","ID_input"]
)
mfeat.iloc[0]

num = mfeat.select_dtypes(include='number')
(
   num[num > 0.1] 
    .stack()
    .reset_index()
    .rename(columns={'level_0': 'row_id', 'level_1': 'column', 0: 'value'})
)

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

s[s["POSTCODE_input_SBAGLIATO"]!=s["POSTCODE_benchmark"]][["ADDRESS_CLEAN_input_SBAGLIATO",'ADDRESS_CLEAN_benchmark']]

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
labels = [d["label"] for d in cons.DISTANCES if d["label"] != "postcode"]
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
