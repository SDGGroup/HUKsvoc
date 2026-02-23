"""Data models for automatic matching distance calculations.

This module defines data structures used to configure and represent
distance/similarity calculations in the automatic matching system.
"""

from dataclasses import dataclass
from svoc.automatic.enums import DistanceMethod
from typing import Optional

@dataclass(frozen=True)
class Distance:
    """Configuration for a single distance/similarity calculation.
    
    Represents a specific distance or similarity metric to be computed between
    two data fields. This immutable dataclass is used to configure which columns
    should be compared using which distance methods, and how the results should
    be labeled.
    
    The frozen=True parameter makes instances immutable, ensuring configuration
    cannot be changed after creation.
    
    Attributes:
        col_name_x: Name of the first column/field to compare (e.g., 'OUTLET_NAME', 'ADDRESS').
                    Always required.
        col_name_y: Name of the second column/field to compare. Optional; if not specified,
                    defaults to the same value as col_name_x (auto-set in __post_init__).
        method: Distance calculation method to use (from DistanceMethod enum).
        label: Unique label for this distance calculation, used in filter configurations
               and result columns (e.g., 'outlet_name_cosine', 'address_levenshtein').
    
    Example:
        >>> Distance('OUTLET_NAME', DistanceMethod.COSINE, 'outlet_name_cosine')
        # col_name_y auto-set to 'OUTLET_NAME'
        >>> Distance('ADDRESS', DistanceMethod.LEVENSHTEIN, 'address_levenshtein', col_name_y='ADDRESS_2')
    """
    
    col_name_x: str
    method: DistanceMethod
    label: str
    col_name_y: Optional[str] = None 

    def __post_init__(self):         
        if self.col_name_y is None:
            object.__setattr__(self, "col_name_y", self.col_name_x)
