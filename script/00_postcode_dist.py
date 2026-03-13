from svoc.settings import get_settings
from svoc.utils import read_data_from_csv
from svoc.datapreparation import rename_and_select_cols, make_upper_str, remove_accents_and_regex
import numpy as np
from sklearn.neighbors import NearestNeighbors


settings = get_settings()
# settings = get_settings("./config/dev2.yaml")

df_input, df_benchmark = read_data_from_csv(settings)

# - Non ci sono coordinane NaN per i dati di input
# - Ci sono alcune coordinate NaN per i dati di benchmark (poche)

## Data Prep

pc_bench=rename_and_select_cols(
    df=df_benchmark, 
    dict_cols={'POSTCODE':'outletpostcode', 'LAT':'latitude', 'LONG':'longitude'}
    ).drop_duplicates()
pc_bench=make_upper_str(df=pc_bench)
pc_bench=remove_accents_and_regex(
    df=pc_bench, 
    re_pattern=r'[^a-zA-Z0-9]', 
    l_cols_not_to_apply=['LAT','LONG']
    )
all_pc = list(pc_bench['POSTCODE'].copy())

pc_bench=pc_bench.dropna(subset=['LAT','LONG'])
pc_bench[['LAT','LONG']] = np.radians(
    pc_bench[['LAT','LONG']].astype(float)
)

pc_input=rename_and_select_cols(
    df=df_input, 
    dict_cols={'POSTCODE':'outletpostcode', 'LAT':'latitude', 'LONG':'longitude'}
    ).drop_duplicates()
pc_input=make_upper_str(df=pc_input)
pc_input=remove_accents_and_regex(
    df=pc_input, 
    re_pattern=r'[^a-zA-Z0-9]', 
    l_cols_not_to_apply=['LAT','LONG']
    ).dropna(subset=['LAT','LONG'])
pc_input[['LAT','LONG']] = np.radians(
    pc_input[['LAT','LONG']].astype(float)
)

## Neighbors

## Trova K vicini
k = 6
nn = NearestNeighbors(
    n_neighbors=k,
    metric="haversine"
)
## Trova vicini entro tot KM
def km_to_radians(km):
    return km / 6371.0

km = 5
nn = NearestNeighbors(
    radius=km_to_radians(km),  
    metric="haversine"
)


nn.fit(pc_input[['LAT','LONG']])
distances, indices = nn.kneighbors(pc_bench[['LAT','LONG']])

groups = {}
for i, cap in enumerate(pc_bench["POSTCODE"]):
    neighbors = pc_input.iloc[indices[i]]["POSTCODE"].tolist()
    neighbors = list(dict.fromkeys(neighbors))
    groups[cap] = neighbors


# groups['W72DT']
# pc_bench[pc_bench["POSTCODE"]=='W72DT']

# b=make_upper_str(df=df_benchmark)
# b=remove_accents_and_regex(
#     df=b, 
#     re_pattern=r'[^a-zA-Z0-9]', 
#     l_cols_not_to_apply=['OutletName','LAT','LONG']
#     )
# b[b["OutletPostcode"]=='W1U5JZ']

# Alcuni Postalcode non hanno vicini ad esempio perchè non hanno coordinate
# - Prendo i postalcode mancanti dalle chiavi

missing_pcs = [ pc for pc in all_pc if pc not in groups.keys() and pc is not np.nan]

# - Creo un gruppo con solo se stessi come vicino

for pc in missing_pcs:
    groups[pc] = [pc] 

# - Controllo che per ogni chiave ci sia sè stesso come vicino

x = [pc for pc in groups.keys() if pc not in groups[pc]]
for xpc in x:
    groups[xpc].append(xpc)


# rows = []
# for i, cap_A in enumerate(pc_bench["POSTCODE"]):
#     for j, idx in enumerate(indices[i]):
#         rows.append({
#             "POSTCODE_benchmark": cap_A,
#             "POSTCODE_input": pc_input.iloc[idx]["POSTCODE"]
#         })
# result = pd.DataFrame(rows).drop_duplicates()

import json
with open("./data/postcode_new.json", "w") as f:
    json.dump(groups, f, indent=2)
