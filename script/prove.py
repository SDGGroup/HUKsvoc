import pandas as pd
from svoc.settings import get_settings
from svoc.utils import read_data_from_csv, concat_l
from svoc.datapreparation import prepare_data
from svoc.orchestrator import svoc_knn

import json
import tqdm

settings = get_settings(("./config/dev3.yaml"))

input, bench = read_data_from_csv(settings)

cl = svoc_knn(
    settings=settings, 
    df_input=input, 
    df_benchmark=bench, 
    k=settings.K_NEIGHBOURS,
    save=False,
)
## or
# with open('./data/postcode_neighbourhood.json', 'r', encoding='utf-8') as f:
#     cl = json.load(f)

input[[settings.INPUT_COLUMNS.ID, 'sapcode']].drop_duplicates()
bench = prepare_data(df=bench, dict_cols=settings.BENCHMARK_COLUMNS_DICT).reset_index().rename(columns={'ID':'ID_1'})
input = prepare_data(df=input, dict_cols=settings.INPUT_COLUMNS_DICT).reset_index().rename(columns={'ID':'ID_2'})

output = pd.read_csv('./data/output.csv')
bench['ID_1'] = pd.to_numeric(bench['ID_1'], errors='coerce').astype('Int64')
input['ID_2'] = pd.to_numeric(input['ID_2'], errors='coerce').astype('Int64')
output.merge(
    bench, on='ID_1', how='left'
).merge(
    input, on='ID_2', how='left', suffixes=('_bench', '_input')
).to_excel('./data/output_STGT.xlsx', index=False)


# Missing matches
missing_m = bench[
    ~bench['ID_1'].isin(list(output['ID_1'].astype(str).drop_duplicates()))
]
missing_m.to_excel('./data/missing_CGA_cds_STGT.xlsx', index=False)
input.to_excel('./data/stonegt.xlsx', index=False)


pcs = missing_m[
    ~missing_m['POSTCODE'].isna()
    ]['POSTCODE'].drop_duplicates().tolist()
pairs = []
for idx, zipcode in tqdm.tqdm(enumerate(pcs)):
    # print(f'{idx} - {zipcode}')
    b = missing_m[missing_m['POSTCODE'] == zipcode].copy()
    i = input[input['POSTCODE'].isin(cl[zipcode])].copy()
    
    b['aux'] = 1; i['aux'] = 1
    res = b.merge(i, on='aux', how = 'outer', suffixes=('_1', '_2')).drop('aux', axis=1)
    pairs.append(res)

concat_l(pairs).to_excel('./data/missing_STGT_matches.xlsx', index=False)



