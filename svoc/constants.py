from pathlib import Path
from svoc.supervised.enums import SupervisedModel

DATA_DIR = Path("./data")
INPUT_FILEPATH = DATA_DIR / "HUK_bowimi_data.csv"
BENCHMARK_FILEPATH = DATA_DIR / "HUK_sap_data.csv"

INPUT_COLUMNS = {
    'ID': 'BowimiId',
    'OUTLET_NAME': 'OutletName',
    'POSTCODE': 'OutletPostCode',
    'ADDRESS': 'OutletAddress'
}
BENCHMARK_COLUMNS = {
    'ID': 'SapCode',
    'OUTLET_NAME': 'OutletName',
    'POSTCODE': 'OutletPostcode',
    'ADDRESS': 'OutletAddress'
}

MODELS_DIR = Path("./models")
SUPERVISED_MODEL_PATHS: dict[SupervisedModel, Path] = {
    SupervisedModel.LOGREG: MODELS_DIR / "logreg_model.pkl",
    SupervisedModel.SVM: MODELS_DIR / "svm_model.pkl",
    SupervisedModel.NAIVE_BAYES: MODELS_DIR / "bayes_model.pkl",
}

N_MATCHES = 3
BLOCK_COL = 'POSTCODE'

NOISE_WORDS_OUTLETNAME = ["THE","BAR","PUB", "LTD", "TA"]
NOISE_WORDS_ADDRESS = ["OF","ROAD", "RD", "STREET", "ST","AVENUE", "AV", "DRIVE", "DR", "LANE", "LN", "BOULEVARD", "BLVD", "COURT", "CT", "PLACE", "PL", "SQUARE", "SQ", "TERRACE", "TER", "CRESCENT", "CRES", "HIGHWAY", "HWY"]
NOISE_WORDS_ADDRESS_REPLACE = {
            r'\bRD\b': 'ROAD',
            r'\bAVE\b': 'AVENUE',
            r'\bLN\b': 'LANE',
            r'\bDR\b': 'DRIVE',
            r'\bCT\b': 'COURT',
            r'\bSQ\b': 'SQUARE',
            r'\bPDE\b': 'PARADE',
            r'\bCL\b': 'CLOSE',
            r'\bCRES\b': 'CRESCENT',
            r'\bPL\b': 'PLACE',
            r'\bTER\b': 'TERRACE',
            r'\bHSE\b': 'HOUSE',
            r'\bGDN(S)?\b': 'GARDENS',
            r'\bPK\b': 'PARK',
            r'\bIND EST\b': 'INDUSTRIAL ESTATE',
            r'\bBLVD\b': 'BOULEVARD',
            r'\bHWY\b': 'HIGHWAY'
        }

DISTANCES = [
    {
    'col_name': 'OUTLET_NAME',
    'method': 'cosine',
    'label': 'outlet_name_cosine'
    },
    {
    'col_name': 'OUTLET_NAME',
    'method': 'jarowinkler',
    'label': 'outlet_name_jarowinkler'
    },
    {
    'col_name': 'OUTLET_NAME',
    'method': 'levenshtein',
    'label': 'outlet_name_levenshtein'
    },
    {
    'col_name': 'OUTLET_NAME',
    'method': 'qgram',
    'label': 'outlet_name_qgram'
    },
    {
    'col_name': 'OUTLET_NAME_CLEAN',
    'method': 'cosine',
    'label': 'outlet_name_clean_cosine'
    },
    {
    'col_name': 'OUTLET_NAME_CLEAN',
    'method': 'jarowinkler',
    'label': 'outlet_name_clean_jarowinkler'
    },
    {
    'col_name': 'OUTLET_NAME_CLEAN',
    'method': 'levenshtein',
    'label': 'outlet_name_clean_levenshtein'
    },
    {
    'col_name': 'OUTLET_NAME_CLEAN',
    'method': 'qgram',
    'label': 'outlet_name_clean_qgram'
    },
    {
    'col_name': 'ADDRESS',
    'method': 'cosine',
    'label': 'address_cosine'
    },
    {
    'col_name': 'ADDRESS',
    'method': 'jarowinkler',
    'label': 'address_jarowinkler'
    },
    {
    'col_name': 'ADDRESS',
    'method': 'levenshtein',
    'label': 'address_levenshtein'
    },
    {
    'col_name': 'ADDRESS',
    'method': 'qgram',
    'label': 'address_qgram'
    },
    {
    'col_name': 'ADDRESS_CLEAN',
    'method': 'cosine',
    'label': 'address_clean_cosine'
    },
    {
    'col_name': 'ADDRESS_CLEAN',
    'method': 'jarowinkler',
    'label': 'address_clean_jarowinkler'
    },
    {
    'col_name': 'ADDRESS_CLEAN',
    'method': 'levenshtein',
    'label': 'address_clean_levenshtein'
    },
    {
    'col_name': 'ADDRESS_CLEAN',
    'method': 'qgram',
    'label': 'address_clean_qgram'
    },
    {
    'col_name': 'OUTLET_NAME',
    'label': 'outlet_name'
    },
    {
    'col_name': 'ADDRESS',
    'label': 'address'
    },
    {
    'col_name': 'OUTLET_NAME_CLEAN',
    'label': 'outlet_name_clean'
    },
    {
    'col_name': 'ADDRESS_CLEAN',
    'label': 'address_clean'
    },
    {
    'col_name': 'POSTCODE',
    'label': 'postcode'
    }
]

## Identici
filter_eq_outletname_eq_address = {
    "outlet_name": 0.5,  
    "address": 0.5  
}

## Outlet name uguale, Address simile
filter_eq_outletname_address_cosine = {
    "outlet_name": 0.5,
    "address_cosine": 0.70,
} 
filter_eq_outletname_address_levenshtein = {
    "outlet_name": 0.5,
    "address_levenshtein": 0.70,
}
filter_eq_outletname_address_qgram = {
    "outlet_name": 0.5,
    "address_qgram": 0.5,
}

## Outlet name uguale, Address CLEAN simile
filter_eq_outletname_addressclean_cosine = {
    "outlet_name": 0.5, 
    "address_clean_cosine": 0.70 # 0.8
}
filter_eq_outletname_addressclean_levenshtein = {
    "outlet_name": 0.5,  
    "address_clean_levenshtein": 0.70 # 0.8
}
filter_eq_outletname_addressclean_qgram = {
    "outlet_name": 0.5, 
    "address_clean_qgram": 0.65
}

## Address uguale, Outlet name simile
filter_eq_address_outletname_cosine = {
    "outlet_name_cosine": 0.70,
    "address": 0.5, 
}
filter_eq_address_outletname_levenshtein = {
    "outlet_name_levenshtein": 0.70,
    "address": 0.5, 
}
filter_eq_address_outletname_qgram = {
    "outlet_name_qgram": 0.65,
    "address": 0.5, 
}

## Address uguale, Outlet name CLEAN simile
filter_eq_address_outletnameclean_cosine = {
    "outlet_name_clean_cosine": 0.70, #0.8,
    "address": 0.5  
}
filter_eq_address_outletnameclean_levenshtein = {
    "outlet_name_clean_levenshtein":  0.70, #0.8,
    "address": 0.5  
}
filter_eq_address_outletnameclean_qgram = {
    "outlet_name_clean_qgram": 0.65,
    "address": 0.5  
}

## Address uguale, Outlet Name (CLEAN) condiviso
filter_eq_address_eq_outletnamein = {
    "outlet_name_in": 0.50,
    "address": 0.50,
}
filter_eq_addressclean_eq_outletnamecleanin = {
    'outlet_name_clean_in': 0.50,
    "address_clean": 0.50,
}

## Outlet Name uguale, Address (CLEAN) condiviso
filter_eq_outletname_eq_addressin = {
    "outlet_name": 0.50,
    "address_in": 0.50, 
}
filter_eq_outletnameclean_eq_addresscleanin = {
    "outlet_name_clean": 0.50,
    "address_clean_in": 0.50,
}

## Outlet Name condiviso, Address (CLEAN) condiviso
filter_eq_outletnamein_eq_addresscleanin = {
    "outlet_name_in": 0.50,
    "address_clean_in": 0.50,
}
filter_eq_outletnamein_eq_addressin = {
    "outlet_name_in": 0.50,
    "address_in": 0.50,
}

## Outlet Name (CLEAN) condiviso, Address (CLEAN) condiviso 2
filter_eq_outletnamein2_eq_addressin2 = {
    "outlet_name_in2": 0.50,
    "address_in2": 0.50,
}
filter_eq_outletnamecleanin2_eq_addresscleanin2 = {
    "outlet_name_clean_in2": 0.50,
    "address_clean_in2": 0.50,
}

## Entrambi simili
# - CLEAN cosine
filter_outlet_nameclean_cosine_addressclean_cosine = {
    "outlet_name_clean_cosine": 0.85,
    "address_clean_cosine": 0.85,
} 
# - CLEAN levenshtein
filter_outlet_nameclean_levenshtein_addressclean_levenshtein = {
    "outlet_name_clean_levenshtein": 0.80,
    "address_clean_levenshtein": 0.80,
}
# - CLEAN jarowinkler
filter_outlet_nameclean_jarowinkler_addressclean_jarowinkler = {
    "outlet_name_clean_jarowinkler":  0.80,
    "address_clean_jarowinkler": 0.80,
}
# - CLEAN qgram
filter_outlet_nameclean_qgram_addressclean_qgram = {
    "outlet_name_clean_qgram": 0.65,
    "address_clean_qgram": 0.65, 
}
# - Originali cosine
filter_outletname_cosine_address_cosine = {
    "outlet_name_cosine": 0.80,
    "address_cosine": 0.80,
}
# - Originali levenshtein
filter_outletname_levenshtein_address_levenshtein = {
    "outlet_name_levenshtein": 0.80,
    "address_levenshtein": 0.80,
}
# - Originali jarowinkler
filter_outletname_jarowinkler_address_jarowinkler = {
    "outlet_name_jarowinkler": 0.80,
    "address_jarowinkler": 0.80,
}
# - Originali qgram
filter_outletname_qgram_address_qgram = {
    "outlet_name_qgram": 0.65,
    "address_qgram": 0.65
}
# - Misti CLEAN e Originali
filter_outletname_cosine_addressclean_cosine = {
    "outlet_name_cosine": 0.80,
    "address_clean_cosine": 0.80,
}
filter_outletname_levenshtein_addressclean_levenshtein = {
    "outlet_name_levenshtein": 0.80,
    "address_clean_levenshtein": 0.80,
}
filter_outletname_jarowinkler_addressclean_jarowinkler = {
    "outlet_name_jarowinkler": 0.80, 
    "address_clean_jarowinkler": 0.80, 
}
filter_outletname_qgram_addressclean_qgram = {
    "outlet_name_qgram": 0.65,
    "address_clean_qgram": 0.65,
}
filter_outletnameclean_cosine_address_cosine = {
    "outlet_name_clean_cosine": 0.80,
    "address_cosine": 0.80,
}
filter_outletnameclean_levenshtein_address_levenshtein = {
    "outlet_name_clean_levenshtein": 0.80,
    "address_levenshtein": 0.80,
}
filter_outletnameclean_jarowinkler_address_jarowinkler = {
    "outlet_name_clean_jarowinkler": 0.80, 
    "address_jarowinkler": 0.80, 
}
filter_outletnameclean_qgram_address_qgram = {
    "outlet_name_clean_qgram": 0.65,
    "address_qgram": 0.65,
}

filter_outletnameclean_jaro_addressclean_levenshtein = {
    "outlet_name_clean_jarowinkler": 0.65, #0.75
    "address_clean_levenshtein": 0.65,
}
filter_outletnameclean_levenshtein_addressclean_cosine = {
    "outlet_name_clean_levenshtein": 0.65, #0.75
    "address_clean_cosine": 0.65, #0.75
}
filter_outletnameclean_jaro_addressclean_cosine = {
    "outlet_name_clean_jarowinkler": 0.65, #0.75
    "address_clean_cosine": 0.65, #0.75
}
filter_outletnameclean_jaro_addressclean_jaro = {
    "outlet_name_clean_jarowinkler": 0.60, #0.75
    "address_clean_jarowinkler": 0.60, #0.75
}

## Univariati
filter_eq_outletname = {
    "outlet_name": 0.5
}
filter_eq_address = {
    "address": 0.5
}
filter_eq_outletnameclean = {
    "outlet_name_clean": 0.5
}
filter_eq_addressclean = {
    "address_clean": 0.5
}
filter_outletnamecleanin2 = {
    "outlet_name_clean_in2": 0.5
}
filter_addresscleanin2 = {
    "address_clean_in2": 0.5
}
filter_addressclean_cosine = {
    "address_clean_cosine": 0.80,
    "outlet_name_clean_cosine": 0.30
}
filter_addressclean_levenshtein = {
    "address_clean_levenshtein": 0.80,
    "outlet_name_clean_cosine": 0.30
}
filter_addressclean_jarowinkler = {
    "address_clean_jarowinkler": 0.80,
    "outlet_name_clean_cosine": 0.30
}
filter_addressclean_levenshtein2 = {
    "address_clean_levenshtein": 0.80,
    "outlet_name_clean_levenshtein": 0.20
}





filter_outletnameclean_cosine={
    "outlet_name_clean_cosine": 0.70,
}
filter_outletnameclean_levenshtein = {
    "outlet_name_clean_levenshtein": 0.80
}
filter_outletnameclean_jarowinkler = {
    "outlet_name_clean_jarowinkler": 0.80
}
filter_addressclean_cosine2 = {
    "address_clean_cosine": 0.70
}
filter_addressclean_levenshtein2 = {
    "address_clean_levenshtein": 0.80
}
filter_addressclean_jarowinkler2 = {
    "address_clean_jarowinkler": 0.80
}
## Casi limite
filter_EX_outletnameclean_jaro_addressclean_levenshtein = {
    "outlet_name_clean_jarowinkler": 0.30,
    "address_clean_levenshtein": 0.30,
}
filter_EX_addressclean_cosine = {
    "address_clean_cosine": 0.30
}
filter_EX_addressclean_levenshtein = {
    "address_clean_levenshtein": 0.30
}
filter_EX_outletnameclean_cosine = {
    "outlet_name_clean_cosine": 0.30
}
filter_EX_outletnameclean_jarowinkler = {
    "outlet_name_clean_jarowinkler": 0.30
}
#47, 28, 24, 49, 32
FILTERS_AUTO= [
    filter_eq_outletname_eq_address,
    filter_eq_outletname_addressclean_cosine,   
    filter_eq_outletname_addressclean_levenshtein,
    filter_eq_outletname_addressclean_qgram,    
    filter_eq_address_outletnameclean_cosine, 
    filter_eq_address_outletnameclean_levenshtein,
    filter_eq_address_outletnameclean_qgram,
    filter_eq_address_outletname_cosine,
    filter_eq_address_outletname_levenshtein,
    filter_eq_address_outletname_qgram,
    filter_eq_outletname_address_cosine,    
    filter_eq_outletname_address_levenshtein,
    filter_eq_outletname_address_qgram, 
    filter_eq_address_eq_outletnamein,
    filter_eq_addressclean_eq_outletnamecleanin,
    filter_eq_outletname_eq_addressin,  
    filter_eq_outletnameclean_eq_addresscleanin, 
    filter_eq_outletnamein_eq_addresscleanin,   
    filter_eq_outletnamein_eq_addressin,
    filter_eq_outletnamein2_eq_addressin2,          # NEW
    filter_eq_outletnamecleanin2_eq_addresscleanin2,     # NEW
    filter_outlet_nameclean_cosine_addressclean_cosine, 
    filter_outlet_nameclean_levenshtein_addressclean_levenshtein,
    filter_outlet_nameclean_jarowinkler_addressclean_jarowinkler, # NEW
    filter_outlet_nameclean_qgram_addressclean_qgram,   
    filter_outletname_cosine_address_cosine,    
    filter_outletname_levenshtein_address_levenshtein, 
    filter_outletname_jarowinkler_address_jarowinkler, # NEW 
    filter_outletname_qgram_address_qgram,
    filter_outletname_cosine_addressclean_cosine,   
    filter_outletname_levenshtein_addressclean_levenshtein,
    filter_outletname_jarowinkler_addressclean_jarowinkler, # NEW
    filter_outletname_qgram_addressclean_qgram,
    filter_outletnameclean_cosine_address_cosine,   
    filter_outletnameclean_levenshtein_address_levenshtein,
    filter_outletnameclean_jarowinkler_address_jarowinkler, # NEW
    filter_outletnameclean_qgram_address_qgram,
    filter_outletnameclean_jaro_addressclean_levenshtein, # NEW 
    filter_outletnameclean_levenshtein_addressclean_cosine,
    filter_outletnameclean_jaro_addressclean_cosine, # NEW 
    filter_outletnameclean_jaro_addressclean_jaro,
    filter_eq_outletname,   # NEW
    filter_eq_address,  # NEW
    filter_eq_outletnameclean,  # NEW
    filter_eq_addressclean ,  # NEW
    filter_outletnamecleanin2,  # NEW
    filter_addresscleanin2,  # NEW
    filter_addressclean_cosine,  # NEW
    filter_addressclean_levenshtein,    # NEW
    filter_addressclean_jarowinkler,    # NEW
    filter_addressclean_levenshtein2
    # filter_outletnameclean_cosine,   # NEW
    # filter_outletnameclean_levenshtein, # NEW
    # filter_outletnameclean_jarowinkler, # NEW
    # filter_addressclean_cosine2,  # NEW
    # filter_addressclean_levenshtein2,  # NEW
    # filter_addressclean_jarowinkler2  # NEW
]
