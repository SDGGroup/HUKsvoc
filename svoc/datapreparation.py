
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from svoc.constants import NOISE_WORDS_OUTLETNAME, NOISE_WORDS_ADDRESS, NOISE_WORDS_ADDRESS_REPLACE

def rename_and_select_cols(df, dict_cols):
    inv_dict_cols = {v: k for k, v in dict_cols.items()}
    df_out = df.rename(columns=inv_dict_cols)
    df_out = df_out[dict_cols.keys()]
    return df_out

def make_upper_str(df):
    df_out = df.applymap(lambda x: str(x).upper())
    return df_out

# def parse_address_components(df):

#     out = df.copy()
    
#     out['ADDRESS'] = out['ADDRESS'].astype(str)
#     out['POSTCODE'] = out['POSTCODE'].astype(str)

#     def process_row(row):
#         address = row['ADDRESS']
#         postcode = row['POSTCODE']
        
#         # 1. Removing Postcode from Address
#         if postcode and postcode not in ['NAN', 'NONE', '']:
#             safe_pcode = [re.escape(char) for char in postcode if char.strip()]
#             pcode_pattern = r'\s*'.join(safe_pcode)
#             address = re.sub(pcode_pattern, '', address)
#             address = address.strip().strip(',').strip()
        
#         # 2. Extracting Town from Address
#         parts = address.rsplit(',', 1)
#         if len(parts) == 2:
#             addr_new = parts[0].strip()
#             town = parts[1].strip()
#         else:
#             addr_new = parts[0].strip()
#             town = None 
            
#         return pd.Series([town, addr_new])

#     # Applichiamo la funzione riga per riga
#     out[['TOWN', 'ADDRESS_NEW']] = out.apply(process_row, axis=1)
    
#     return out

def parse_address_components(df, get_town=True):
    
    out = df.copy()
    
    out["processAdd"] = out['POSTCODE'].str.contains(r'\d', na=True)

    out['ADDRESS'] = out['ADDRESS'].astype(str)

    def process_row(address, get_town=True):
        # --- 1. RIMOZIONE "UK" ---
        address = re.sub(r',?\s*UK$', '', address)
        address = address.strip()
        
        # --- 2. ESTRAZIONE POSTCODE (ULTIME 2 STRINGHE) ---
        parts = address.split()
        postcode = None
        remaining_address = address
        if len(parts) >= 2:
            postcode = f"{parts[-2]} {parts[-1]}"
            remaining_address = " ".join(parts[:-2])
        else:
            remaining_address = address

        remaining_address = remaining_address.strip().strip(',').strip()

        # --- 3. ESTRAZIONE CITTA (ULTIMA VIRGOLA) ---
        if get_town:
            town = None
            final_address = remaining_address
            if ',' in remaining_address:
                split_addr = remaining_address.rsplit(',', 1)
                final_address = split_addr[0].strip()
                town = split_addr[1].strip()
            else:
                final_address = remaining_address
                town = None
            return pd.Series([postcode, town, final_address])
        else:
            return pd.Series([postcode, remaining_address])
        
    if get_town:
        new_cols = ['POSTCODE', 'TOWN', 'ADDRESS']
    else:
        new_cols = ['POSTCODE', 'ADDRESS']        
    
    # out[new_cols] = out['ADDRESS'].apply(process_row, get_town=get_town)

    mask = out['processAdd']
    df_new = out.loc[mask, 'ADDRESS'].apply(process_row, get_town=get_town)
    out.loc[mask, new_cols] = pd.DataFrame(df_new.values, index=df_new.index, columns=new_cols)
    
    return out.drop(columns=['processAdd'])

def remove_accents_and_regex(df, re_pattern, l_id_cols, l_cols_not_to_apply=[]):
    df_out = df.replace(['NAN', 'nan', 'NONE'], '')
    df_out[df_out.columns.difference(l_id_cols+l_cols_not_to_apply)] = df_out[df_out.columns.difference(l_id_cols+l_cols_not_to_apply)].applymap(lambda x: unidecode(x) if pd.notna(x) else x)
    df_out[df_out.columns.difference(l_id_cols+l_cols_not_to_apply)] = df_out[df_out.columns.difference(l_id_cols+l_cols_not_to_apply)].apply(lambda col: col.str.replace(re_pattern, '', regex=True))
    df_out = df_out.replace('', np.nan)
    return df_out

def remove_noise_words(df, col, words_to_remove, name = None):
    if name is None:
        name = col
    df[name] = df[col].apply(
        lambda x: ' '.join([word for word in str(x).split() if word not in words_to_remove]))
    return df

def clean_address_noise_words(df, col, name=None):
    out = df.copy()
    
    if name is None:
        name = col

    def process_row(address):
        if not isinstance(address, str):
            return ""
        
        # 2. Normalizza i punti (ST. -> ST) ma TIENI LE VIRGOLE per ora
        address = address.replace('.', '') 
        
        # --- LOGICA INTELLIGENTE PER "ST" ---
        
        # CASO A: SAINT (Inizio riga)
        # Es: "ST PAULS ROAD" -> "SAINT PAULS ROAD"
        address = re.sub(r'^ST\b', 'SAINT', address)
        
        # CASO B: SAINT (Dopo un numero civico)
        # Es: "10 ST JOHNS" -> "10 SAINT JOHNS"
        # (?<=\d) è un lookbehind: cerca se c'è un numero prima dello spazio
        address = re.sub(r'(?<=\d)\s+ST\b', ' SAINT', address)
        
        # CASO C: SAINT (Dopo una virgola, probabile città)
        # Es: "HIGH ST, ST ALBANS" -> "HIGH ST, SAINT ALBANS"
        address = re.sub(r',\s*ST\b', ', SAINT', address)
        
        # CASO D: STREET (Tutti gli altri casi rimasti)
        # Se "ST" è sopravvissuto alle regole sopra, è quasi sicuramente Street
        # Es: "REGENT ST LONDON" -> "REGENT STREET LONDON"
        address = re.sub(r'\bST\b', 'STREET', address)
        
        # ------------------------------------

        # 3. Ora puoi rimuovere la punteggiatura residua (virgole, etc)
        address = re.sub(r'[^\w\s]', '', address)

        # 4. ESPANSIONE ALTRE ABBREVIAZIONI (Standard)        
        for pattern, replacement in NOISE_WORDS_ADDRESS_REPLACE.items():
            address = re.sub(pattern, replacement, address)
            
        # 5. Normalizza spazi doppi
        address = re.sub(r'\s+', ' ', address).strip()
        
        return address
    
    out[name] = out[col].apply(process_row)
    return out

def prepare_data(df, dict_cols, parse_address = True, rm_address_noise = False, get_town = True):
    out=rename_and_select_cols(df=df, dict_cols=dict_cols)
    out=make_upper_str(df=out)

    out = out[~out['OUTLET_NAME'].str.contains("DO NOT USE", case=False, na=False)]

    if parse_address:
        out=parse_address_components(df=out, get_town=get_town)

    if rm_address_noise:
        out=remove_accents_and_regex(df=out, re_pattern=r'[^a-zA-Z0-9\s]', l_id_cols=['ID'], l_cols_not_to_apply=['ADDRESS'])
        out=remove_noise_words(df=out, col='OUTLET_NAME', name = 'OUTLET_NAME_CLEAN',words_to_remove=NOISE_WORDS_OUTLETNAME)
        out=clean_address_noise_words(df=out, col='ADDRESS', name='ADDRESS_CLEAN')
    else:
        out=remove_accents_and_regex(df=out, re_pattern=r'[^a-zA-Z0-9\s]', l_id_cols=['ID'])
        out=remove_noise_words(df=out, col='OUTLET_NAME', name = 'OUTLET_NAME_CLEAN',words_to_remove=NOISE_WORDS_OUTLETNAME)
        out=remove_noise_words(df=out, col='ADDRESS', name = 'ADDRESS_CLEAN', words_to_remove=NOISE_WORDS_ADDRESS)
    
    out=remove_accents_and_regex(df=out, re_pattern=r'[^a-zA-Z0-9]', l_id_cols=['ID'], 
                                 l_cols_not_to_apply=['OUTLET_NAME','OUTLET_NAME_CLEAN', 'ADDRESS','ADDRESS_CLEAN'])
    out = out.replace('', np.nan)
    out = out.set_index('ID')
    return out

def split_df(
        df: pd.DataFrame, 
        split_col: str, 
        num_groups: int
    ):
    df_split = pd.DataFrame({'COUNT': df[[split_col]].value_counts()}).reset_index()
    df_split = df_split.rename(columns={split_col: 'GROUP'})
    groups = [[] for _ in range(num_groups)]
    sums = [0] * num_groups

    # Distribute rows to each group
    for row in df_split.itertuples(index=False):
        min_sum_index = sums.index(min(sums))
        groups[min_sum_index].append(row.GROUP)
        sums[min_sum_index] += row.COUNT

    # Create a new DataFrame to show the result
    result_df = pd.DataFrame()
    result_df['GROUP'] = [group for group in groups]
    result_df['N_ELEMENTS'] = result_df['GROUP'].str.len()
    result_df['N_ROWS'] = sums

    return result_df