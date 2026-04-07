
"""Data preparation module for outlet matching.

This module provides functions for cleaning and standardizing outlet and address data,
including:
- Column renaming and selection
- Text normalization (uppercase conversion, accent removal)
- Address parsing and component extraction
- Noise word removal from outlet names and addresses
- Data splitting for parallel processing
"""

import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from svoc.constants import NOISE_WORDS_OUTLETNAME, NOISE_WORDS_ADDRESS, NOISE_WORDS_ADDRESS_REPLACE
from logging import Logger

def rename_and_select_cols(
        df: pd.DataFrame, 
        dict_cols: dict[str, str]
        ) -> pd.DataFrame:
    """Rename and select specific columns from a DataFrame.
    
    Args:
        df: Input DataFrame to process
        dict_cols: Dictionary mapping desired column names to current column names
        
    Returns:
        DataFrame with renamed and selected columns
        
    Raises:
        ValueError: If required columns are missing from the DataFrame
        TypeError: If df is not a DataFrame or dict_cols is not a dictionary
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(dict_cols, dict):
        raise TypeError(f"dict_cols must be a dictionary, got {type(dict_cols).__name__}")
    
    inv_dict_cols = {v: k for k, v in dict_cols.items()}
    missing_cols = set(inv_dict_cols.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Required columns missing from DataFrame: {sorted(missing_cols)}")
    
    df_out = df.rename(columns=inv_dict_cols)
    df_out = df_out[dict_cols.keys()]
    return df_out

def make_upper_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all string columns to uppercase.
    
    Args:
        df: Input DataFrame to process
        
    Returns:
        DataFrame with all string columns converted to uppercase
        
    Raises:
        TypeError: If df is not a DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    
    df_out = df.copy()

    for col in df_out.select_dtypes(include="object"):
        df_out[col] = df_out[col].str.upper()

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

def parse_address_components(df: pd.DataFrame, get_town: bool = True) -> pd.DataFrame:
    """Parse address components to extract the clean postcode, address and, optionally, town.
    
    Processes addresses to:
    - Remove 'UK' suffix
    - Extract postcode (last two words of the address, if they contain digits)
    - Extract town (text after last comma in the address)
    - Clean the remaining address
    
    Args:
        df: DataFrame containing 'ADDRESS' and optionally 'POSTCODE' columns
        get_town: If True, extract town from address. Default is True.
        
    Returns:
        DataFrame with parsed address components (POSTCODE, TOWN, ADDRESS) if get_town is True,
        otherwise (POSTCODE, ADDRESS).
        
    Raises:
        ValueError: If required 'ADDRESS' and 'POSTCODE' columns are missing
        TypeError: If df is not a DataFrame or get_town is not a boolean
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(get_town, bool):
        raise TypeError(f"get_town must be a boolean, got {type(get_town).__name__}")
    if 'ADDRESS' not in df.columns:
        raise ValueError("DataFrame must contain 'ADDRESS' column")
    if 'POSTCODE' not in df.columns:
        raise ValueError("DataFrame must contain 'POSTCODE' column")
    
    out = df.copy()
    
    out["processAdd"] = out['POSTCODE'].str.contains(r'\d', na=True)

    out['ADDRESS'] = out['ADDRESS'].astype(str)

    def process_row(address, get_town=True):
        # --- 1. Removing "UK" suffix ---
        address = re.sub(r',?\s*UK$', '', address)
        address = address.strip()
        
        # --- 2. Extracting postcode (last two words if they contain digits) ---
        parts = address.split()
        postcode = None
        remaining_address = address
        if len(parts) >= 2:
            postcode = f"{parts[-2]} {parts[-1]}"
            remaining_address = " ".join(parts[:-2])
        else:
            remaining_address = address

        remaining_address = remaining_address.strip().strip(',').strip()

        # --- 3. Extracting town (text after last comma) ---
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

def remove_accents_and_regex(
        df: pd.DataFrame, 
        re_pattern: str, 
        l_id_cols: list[str] | None = None,
        l_cols_not_to_apply: list[str] | None = None
    ) -> pd.DataFrame:
    """Remove accents and apply regex pattern to clean text columns.
    
    Processes DataFrame by:
    - Replacing 'NAN', 'nan', 'NONE' with empty strings
    - Removing accents using unidecode
    - Applying regex pattern to remove unwanted characters
    - Replacing empty strings with NaN
    
    Args:
        df: Input DataFrame to process
        re_pattern: Regex pattern to apply for character removal
        l_id_cols: List of ID columns to exclude from processing. Default is None.
        l_cols_not_to_apply: List of columns to exclude from processing. Default is None.
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If re_pattern is empty or invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(re_pattern, str):
        raise TypeError(f"re_pattern must be a string, got {type(re_pattern).__name__}")
    if not re_pattern:
        raise ValueError("re_pattern cannot be empty")
    
    if l_cols_not_to_apply is None:
        l_cols_not_to_apply = []
    if l_id_cols is None:
        l_id_cols = []
    
    if not isinstance(l_id_cols, list):
        raise TypeError(f"l_id_cols must be a list, got {type(l_id_cols).__name__}")
    if not isinstance(l_cols_not_to_apply, list):
        raise TypeError(f"l_cols_not_to_apply must be a list, got {type(l_cols_not_to_apply).__name__}")

    df_out = df.replace(['NAN', 'nan', 'NONE'], '')

    cols_to_clean = df_out.columns.difference(l_id_cols + l_cols_not_to_apply)

    df_out[cols_to_clean] = df_out[cols_to_clean].apply(
        lambda col: col.map(lambda x: unidecode(x) if pd.notna(x) else x)
    )

    df_out[cols_to_clean] = df_out[cols_to_clean].apply(
        lambda col: col.str.replace(re_pattern, '', regex=True)
    )

    df_out = df_out.replace('', np.nan)
    return df_out

def remove_noise_words(
        df: pd.DataFrame, 
        col: str, 
        words_to_remove: list[str], 
        name: str | None = None
    ) -> pd.DataFrame:
    """Remove specified noise words from a text column.
    
    Args:
        df: DataFrame to process
        col: Column name to clean
        words_to_remove: List of words to remove from the text
        name: Name for the output column. If None, uses col name. Default is None.
        
    Returns:
        DataFrame with cleaned column added
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If specified column doesn't exist
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(col, str):
        raise TypeError(f"col must be a string, got {type(col).__name__}")
    if not isinstance(words_to_remove, list):
        raise TypeError(f"words_to_remove must be a list, got {type(words_to_remove).__name__}")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    if name is None:
        name = col
    df[name] = df[col].apply(
        lambda x: ' '.join([word for word in str(x).split() if word not in words_to_remove]))
    return df

def clean_address_noise_words(
        df: pd.DataFrame, 
        col: str, 
        name: str | None = None
    ) -> pd.DataFrame:
    """Clean address by standardizing abbreviations and removing noise.
    
    Performs intelligent address cleaning:
    - Intelligently handles 'ST' (Street vs Saint based on context)
    - Expands common address abbreviations (RD -> ROAD, AVE -> AVENUE, etc.) specified in the NOISE_WORDS_ADDRESS_REPLACE constant
    - Removes punctuation and normalizes spacing
    
    Args:
        df: Input DataFrame to process
        col: Column name containing addresses to clean
        name: Name for the output column. If None, uses col name. Default is None.
        
    Returns:
        DataFrame with cleaned address column added
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If specified column doesn't exist
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(col, str):
        raise TypeError(f"col must be a string, got {type(col).__name__}")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    out = df.copy()
    
    if name is None:
        name = col

    def process_row(address):
        if not isinstance(address, str):
            return ""
        
        # ------------------------------------
        # 1. Replace ST with SAINT or STREET based on context

        address = address.replace('.', '') 
        
        # CASE A: SAINT (Start of the line)
        # Example: "ST PAULS ROAD" -> "SAINT PAULS ROAD"
        address = re.sub(r'^ST\b', 'SAINT', address)
        
        # CASE B: SAINT (After a house number)
        # Example: "10 ST JOHNS" -> "10 SAINT JOHNS"
        # (?<=\d) is a lookbehind: checks if there is a number before the space
        address = re.sub(r'(?<=\d)\s+ST\b', ' SAINT', address)
        
        # CASE C: SAINT (After a comma, likely a city)
        # Example: "HIGH ST, ST ALBANS" -> "HIGH ST, SAINT ALBANS"
        address = re.sub(r',\s*ST\b', ', SAINT', address)
        
        # CASE D: STREET (All other remaining cases)
        # If "ST" survived the rules above, it is almost certainly Street
        # Example: "REGENT ST LONDON" -> "REGENT STREET LONDON"
        address = re.sub(r'\bST\b', 'STREET', address)
        
        # ------------------------------------

        # 2. Remove punctuation (except spaces)
        address = re.sub(r'[^\w\s]', '', address)

        # 3. Expand other abbreviations        
        for pattern, replacement in NOISE_WORDS_ADDRESS_REPLACE.items():
            address = re.sub(pattern, replacement, address)
            
        # 4. Normalize double spaces
        address = re.sub(r'\s+', ' ', address).strip()
        
        return address
    
    out[name] = out[col].apply(process_row)
    return out

def check_duplicates(df: pd.DataFrame, logger: Logger | None = None) -> pd.DataFrame:
    """
    Checks and removes duplicates from a pandas DataFrame.
    
    Identifies completely duplicate rows (all columns) and removes them,
    keeping the first occurrence. If duplicates are found, issues a warning
    via logger (if provided) or console print.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check and clean from duplicates.
    logger : Logger | None, default None
        Optional logger for warnings. If None, uses print().
        
    Returns
    -------
    pd.DataFrame
        DataFrame without duplicates (modified copy).
        
    Notes
    -----
    - Uses `drop_duplicates(keep='first')` implicitly.
    - Warnings show duplicate count and the duplicate rows found.
    - Does not modify the original input DataFrame.
        
    Examples
    --------
    >>> df_clean = check_duplicates(df)
    >>> df_clean = check_duplicates(df, logger=my_logger)
    """
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        if logger:
            logger.warning(f"Duplicates found in the data: {len(duplicates)} rows. The duplicated rows have been removed.")
            # logger.warning(duplicates)
        else:
            print(f"Duplicates found in the data: {len(duplicates)} rows. The duplicated rows have been removed.")
            # print(duplicates)
    
    return df.drop_duplicates()


def check_id(df: pd.DataFrame, logger: Logger | None = None) -> pd.DataFrame:
    """
    Checks for duplicate IDs in the DataFrame.
    
    Verifies that the 'ID' column contains only unique values. If duplicate IDs
    are found, raises a ValueError with details about the duplicates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for duplicate IDs. Must contain an 'ID' column.
    logger : Logger | None, default None
        Optional logger for error messages. If None, only raises error.
        
    Returns
    -------
    pd.DataFrame
        The input DataFrame if no duplicates are found.
        
    Raises
    ------
    ValueError
        If duplicate IDs are found in the 'ID' column.
    ValueError
        If 'ID' column is not present in the DataFrame.
        
    Notes
    -----
    The ID column must have unique values for each row to ensure data integrity.
        
    Examples
    --------
    >>> df_checked = check_id(df)
    >>> df_checked = check_id(df, logger=my_logger)
    """
    if 'ID' not in df.columns:
        raise ValueError("DataFrame must contain an 'ID' column")
    
    duplicate_ids = df[df['ID'].duplicated(keep=False)]
    
    if not duplicate_ids.empty:
        duplicate_id_values = duplicate_ids['ID'].unique()
        error_msg = (
            f"Duplicate IDs found in the data: {len(duplicate_ids)} rows with "
            f"{len(duplicate_id_values)} duplicate ID values. "
            f"The ID column must have unique values for each row."
        )
        
        if logger:
            logger.error(error_msg)
        
        raise ValueError(error_msg)
    
    return df


def prepare_data(
        df: pd.DataFrame, 
        dict_cols: dict[str, str], 
        rm_address_noise: bool = True, 
        parse_address: bool = False, 
        get_town: bool = False,
        logger: Logger | None = None
    ) -> pd.DataFrame:
    """Main data preparation pipeline for outlet matching.
    
    Comprehensive data preparation including:
    1. Column renaming and selection
    2. Uppercase conversion
    3. Filtering out the outlets with 'DO NOT USE' in the name
    4. Optional address parsing
    5. Noise word removal and text cleaning
    6. Accent removal and regex cleaning
    7. Index setting by ID
    
    Args:
        df: Input DataFrame containing outlet data
        dict_cols: Dictionary mapping desired column names to current names
        rm_address_noise: If True, clean addresses with smart abbreviation handling. Default is True.
        parse_address: If True, parse address components. Default is False.
        get_town: If True, extract town from address during parsing. Default is False.
        
    Returns:
        Cleaned and prepared DataFrame indexed by ID
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If required columns are missing
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(dict_cols, dict):
        raise TypeError(f"dict_cols must be a dictionary, got {type(dict_cols).__name__}")
    if not isinstance(parse_address, bool):
        raise TypeError(f"parse_address must be a boolean, got {type(parse_address).__name__}")
    if not isinstance(rm_address_noise, bool):
        raise TypeError(f"rm_address_noise must be a boolean, got {type(rm_address_noise).__name__}")
    if not isinstance(get_town, bool):
        raise TypeError(f"get_town must be a boolean, got {type(get_town).__name__}")
    
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

    out = check_duplicates(df=out, logger=logger)
    out = check_id(df=out, logger=logger)

    return out.set_index('ID')

def split_df(
        df: pd.DataFrame, 
        split_col: str, 
        num_groups: int
    ) -> pd.DataFrame:
    """Split DataFrame into balanced groups based on value counts.
    
    Distributes unique values from split_col into num_groups groups,
    balancing by total row count using a greedy algorithm.
    
    Args:
        df: DataFrame to analyze
        split_col: Column name to use for grouping
        num_groups: Number of groups to create
        
    Returns:
        DataFrame with columns:
        - GROUP: List of values assigned to each group
        - N_ELEMENTS: Number of unique values in each group
        - N_ROWS: Total row count for each group
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If split_col doesn't exist or num_groups is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(split_col, str):
        raise TypeError(f"split_col must be a string, got {type(split_col).__name__}")
    if not isinstance(num_groups, int):
        raise TypeError(f"num_groups must be an integer, got {type(num_groups).__name__}")
    if split_col not in df.columns:
        raise ValueError(f"Column '{split_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    if num_groups < 1:
        raise ValueError(f"num_groups must be at least 1, got {num_groups}")
    
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