
from pathlib import Path
from pydantic import BaseModel, field_validator
from typing import Dict
from svoc.supervised.enums import SupervisedModel
from svoc.automatic.enums import DistanceMethod
from svoc.automatic.models import Distance

SUPERVISED_MODELS_FILENAME: dict[SupervisedModel, Path] = {
    SupervisedModel.LOGREG: "logreg_model.pkl",
    SupervisedModel.SVM: "svm_model.pkl",
    SupervisedModel.NAIVE_BAYES: "bayes_model.pkl",
}

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

DEFAULT_DISTANCES: list[Distance] = [
    Distance('OUTLET_NAME', DistanceMethod.COSINE, 'outlet_name_cosine'),
    Distance('ADDRESS', DistanceMethod.COSINE, 'address_cosine'),
]

DISTANCES: list[Distance] = DEFAULT_DISTANCES + [
    Distance('OUTLET_NAME', DistanceMethod.JAROWINKLER, 'outlet_name_jarowinkler'),
    Distance('OUTLET_NAME', DistanceMethod.LEVENSHTEIN, 'outlet_name_levenshtein'),
    Distance('OUTLET_NAME', DistanceMethod.QGRAM, 'outlet_name_qgram'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.COSINE, 'outlet_name_clean_cosine'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.JAROWINKLER, 'outlet_name_clean_jarowinkler'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.LEVENSHTEIN, 'outlet_name_clean_levenshtein'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.QGRAM, 'outlet_name_clean_qgram'),
    Distance('ADDRESS', DistanceMethod.JAROWINKLER, 'address_jarowinkler'),
    Distance('ADDRESS', DistanceMethod.LEVENSHTEIN, 'address_levenshtein'),
    Distance('ADDRESS', DistanceMethod.QGRAM, 'address_qgram'),
    Distance('ADDRESS_CLEAN', DistanceMethod.COSINE, 'address_clean_cosine'),
    Distance('ADDRESS_CLEAN', DistanceMethod.JAROWINKLER, 'address_clean_jarowinkler'),
    Distance('ADDRESS_CLEAN', DistanceMethod.LEVENSHTEIN, 'address_clean_levenshtein'),
    Distance('ADDRESS_CLEAN', DistanceMethod.QGRAM, 'address_clean_qgram'),
    Distance('OUTLET_NAME', DistanceMethod.EXACT, 'outlet_name'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.EXACT, 'outlet_name_clean'),
    Distance('ADDRESS', DistanceMethod.EXACT, 'address'),
    Distance('ADDRESS_CLEAN', DistanceMethod.EXACT, 'address_clean'),
    Distance('POSTCODE', DistanceMethod.EXACT, 'postcode'),
    Distance('OUTLET_NAME', DistanceMethod.SUBSTRING, 'outlet_name_in'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.SUBSTRING, 'outlet_name_clean_in'),
    Distance('ADDRESS', DistanceMethod.SUBSTRING, 'address_in'),
    Distance('ADDRESS_CLEAN', DistanceMethod.SUBSTRING, 'address_clean_in'),
    Distance('OUTLET_NAME', DistanceMethod.WORDSMATCH, 'outlet_name_in2'),
    Distance('OUTLET_NAME_CLEAN', DistanceMethod.WORDSMATCH, 'outlet_name_clean_in2'),
    Distance('ADDRESS', DistanceMethod.WORDSMATCH, 'address_in2'),
    Distance('ADDRESS_CLEAN', DistanceMethod.WORDSMATCH, 'address_clean_in2'),
]

DISTANCE_LABELS: frozenset[str] = frozenset(
    d.label for d in DISTANCES
)

class DistanceFilter(BaseModel):
    value: Dict[str, float]

    model_config = {
        "frozen": True
    }

    @field_validator("value")
    @classmethod
    def validate_keys_and_values(cls, v):
        for k, val in v.items():
            if k not in DISTANCE_LABELS:
                raise ValueError(
                    f"Filter '{k}' not valid. "
                    f"Allowed values: {sorted(DISTANCE_LABELS)}"
                )

            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Invalid weight for '{k}': {val}. "
                    "Must be between 0 and 1."
                )
        return v

## Identical
filter_eq_outletname_eq_address = {
    "outlet_name": 0.5,  
    "address": 0.5  
}

## Outlet name identical, Address similar
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

## Outlet name identical, Address CLEAN similar
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

## Address identical, Outlet name similar
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

## Address identical, Outlet name CLEAN similar
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

## Address identical, Outlet Name (CLEAN) shared
filter_eq_address_eq_outletnamein = {
    "outlet_name_in": 0.50,
    "address": 0.50,
}
filter_eq_addressclean_eq_outletnamecleanin = {
    'outlet_name_clean_in': 0.50,
    "address_clean": 0.50,
}

## Outlet Name identical, Address (CLEAN) shared
filter_eq_outletname_eq_addressin = {
    "outlet_name": 0.50,
    "address_in": 0.50, 
}
filter_eq_outletnameclean_eq_addresscleanin = {
    "outlet_name_clean": 0.50,
    "address_clean_in": 0.50,
}

## Outlet Name shared, Address (CLEAN) shared
filter_eq_outletnamein_eq_addresscleanin = {
    "outlet_name_in": 0.50,
    "address_clean_in": 0.50,
}
filter_eq_outletnamein_eq_addressin = {
    "outlet_name_in": 0.50,
    "address_in": 0.50,
}

## Outlet Name (CLEAN) shared, Address (CLEAN) shared (2)
filter_eq_outletnamein2_eq_addressin2 = {
    "outlet_name_in2": 0.50,
    "address_in2": 0.50,
}
filter_eq_outletnamecleanin2_eq_addresscleanin2 = {
    "outlet_name_clean_in2": 0.50,
    "address_clean_in2": 0.50,
}

## Both similar
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
# - Original cosine
filter_outletname_cosine_address_cosine = {
    "outlet_name_cosine": 0.80,
    "address_cosine": 0.80,
}
# - Original levenshtein
filter_outletname_levenshtein_address_levenshtein = {
    "outlet_name_levenshtein": 0.80,
    "address_levenshtein": 0.80,
}
# - Original jarowinkler
filter_outletname_jarowinkler_address_jarowinkler = {
    "outlet_name_jarowinkler": 0.80,
    "address_jarowinkler": 0.80,
}
# - Original qgram
filter_outletname_qgram_address_qgram = {
    "outlet_name_qgram": 0.65,
    "address_qgram": 0.65
}
# - Mixed CLEAN and Original
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
    "outlet_name_clean_jarowinkler": 0.65,
    "address_clean_levenshtein": 0.65,
}
filter_outletnameclean_levenshtein_addressclean_cosine = {
    "outlet_name_clean_levenshtein": 0.65,
    "address_clean_cosine": 0.65,
}
filter_outletnameclean_jaro_addressclean_cosine = {
    "outlet_name_clean_jarowinkler": 0.65,
    "address_clean_cosine": 0.65,
}
filter_outletnameclean_jaro_addressclean_jaro = {
    "outlet_name_clean_jarowinkler": 0.60,
    "address_clean_jarowinkler": 0.60,
}

## Univariate
filter_eq_outletname = {
    "outlet_name": 0.5,
    "address_cosine": 0
}
filter_eq_address = {
    "outlet_name_cosine": 0,
    "address": 0.5
}
filter_eq_outletnameclean = {
    "outlet_name_clean": 0.5,
    "address_cosine": 0
}
filter_eq_addressclean = {
    "outlet_name_cosine": 0,
    "address_clean": 0.5
}
filter_outletnamecleanin2 = {
    "outlet_name_clean_in2": 0.5,
    "address_cosine": 0
}
filter_addresscleanin2 = {
    "outlet_name_cosine": 0,
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

FILTERS_AUTO= [
    DistanceFilter(value=filter_eq_outletname_eq_address),
    DistanceFilter(value=filter_eq_outletname_addressclean_cosine),   
    DistanceFilter(value=filter_eq_outletname_addressclean_levenshtein),
    DistanceFilter(value=filter_eq_outletname_addressclean_qgram),    
    DistanceFilter(value=filter_eq_address_outletnameclean_cosine), 
    DistanceFilter(value=filter_eq_address_outletnameclean_levenshtein),
    DistanceFilter(value=filter_eq_address_outletnameclean_qgram),
    DistanceFilter(value=filter_eq_address_outletname_cosine),
    DistanceFilter(value=filter_eq_address_outletname_levenshtein),
    DistanceFilter(value=filter_eq_address_outletname_qgram),
    DistanceFilter(value=filter_eq_outletname_address_cosine),    
    DistanceFilter(value=filter_eq_outletname_address_levenshtein),
    DistanceFilter(value=filter_eq_outletname_address_qgram), 
    DistanceFilter(value=filter_eq_address_eq_outletnamein),
    DistanceFilter(value=filter_eq_addressclean_eq_outletnamecleanin),
    DistanceFilter(value=filter_eq_outletname_eq_addressin),  
    DistanceFilter(value=filter_eq_outletnameclean_eq_addresscleanin), 
    DistanceFilter(value=filter_eq_outletnamein_eq_addresscleanin),   
    DistanceFilter(value=filter_eq_outletnamein_eq_addressin),
    DistanceFilter(value=filter_eq_outletnamein2_eq_addressin2),
    DistanceFilter(value=filter_eq_outletnamecleanin2_eq_addresscleanin2),
    DistanceFilter(value=filter_outlet_nameclean_cosine_addressclean_cosine), 
    DistanceFilter(value=filter_outlet_nameclean_levenshtein_addressclean_levenshtein),
    DistanceFilter(value=filter_outlet_nameclean_jarowinkler_addressclean_jarowinkler),
    DistanceFilter(value=filter_outlet_nameclean_qgram_addressclean_qgram),   
    DistanceFilter(value=filter_outletname_cosine_address_cosine),    
    DistanceFilter(value=filter_outletname_levenshtein_address_levenshtein), 
    DistanceFilter(value=filter_outletname_jarowinkler_address_jarowinkler),
    DistanceFilter(value=filter_outletname_qgram_address_qgram),
    DistanceFilter(value=filter_outletname_cosine_addressclean_cosine),   
    DistanceFilter(value=filter_outletname_levenshtein_addressclean_levenshtein),
    DistanceFilter(value=filter_outletname_jarowinkler_addressclean_jarowinkler),
    DistanceFilter(value=filter_outletname_qgram_addressclean_qgram),
    DistanceFilter(value=filter_outletnameclean_cosine_address_cosine),   
    DistanceFilter(value=filter_outletnameclean_levenshtein_address_levenshtein),
    DistanceFilter(value=filter_outletnameclean_jarowinkler_address_jarowinkler),
    DistanceFilter(value=filter_outletnameclean_qgram_address_qgram),
    DistanceFilter(value=filter_outletnameclean_jaro_addressclean_levenshtein),
    DistanceFilter(value=filter_outletnameclean_levenshtein_addressclean_cosine),
    DistanceFilter(value=filter_outletnameclean_jaro_addressclean_cosine),
    DistanceFilter(value=filter_outletnameclean_jaro_addressclean_jaro),
    DistanceFilter(value=filter_eq_outletname),
    DistanceFilter(value=filter_eq_address),
    DistanceFilter(value=filter_eq_outletnameclean),
    DistanceFilter(value=filter_eq_addressclean),
    DistanceFilter(value=filter_outletnamecleanin2),
    DistanceFilter(value=filter_addresscleanin2),
    DistanceFilter(value=filter_addressclean_cosine),
    DistanceFilter(value=filter_addressclean_levenshtein),
    DistanceFilter(value=filter_addressclean_jarowinkler),
    DistanceFilter(value=filter_addressclean_levenshtein2)
]
