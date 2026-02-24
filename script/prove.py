import pandas as pd
from svoc.settings import get_settings
from svoc.utils import read_data_from_csv, concat_l
from svoc.datapreparation import prepare_data

import json
import tqdm

settings = get_settings()

input, bench = read_data_from_csv(settings)
input[[settings.INPUT_COLUMNS.ID, 'sapcode']].drop_duplicates()
bench = prepare_data(df=bench, dict_cols=settings.BENCHMARK_COLUMNS_DICT).reset_index().rename(columns={'ID':'ID_1'})
input = prepare_data(df=input, dict_cols=settings.INPUT_COLUMNS_DICT).reset_index().rename(columns={'ID':'ID_2'})

output = pd.read_csv('./data/output.csv')
output

# Missing matches
missing_m = bench[
    ~bench['ID_1'].isin(output['ID_1'].drop_duplicates())
]
missing_m.to_excel('./data/missing_sap_matches_cds.xlsx', index=False)

input.to_excel('./data/bowimi.xlsx', index=False)

with open('./data/postcode.json', 'r', encoding='utf-8') as f:
    cl = json.load(f)

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

concat_l(pairs).to_excel('./data/missing_sap_matches.xlsx', index=False)


