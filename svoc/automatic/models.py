"""Data models for automatic matching distance calculations.

This module defines data structures used to configure and represent
distance/similarity calculations in the automatic matching system.
"""

from dataclasses import dataclass
from svoc.automatic.enums import DistanceMethod


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
        col_name: Name of the column/field to compare (e.g., 'OUTLET_NAME', 'ADDRESS')
        method: Distance calculation method to use (from DistanceMethod enum)
        label: Unique label for this distance calculation, used in filter configurations
               and result columns (e.g., 'outlet_name_cosine', 'address_levenshtein')
    
    Example:
        >>> Distance('OUTLET_NAME', DistanceMethod.COSINE, 'outlet_name_cosine')
        >>> Distance('ADDRESS', DistanceMethod.LEVENSHTEIN, 'address_levenshtein')
    """
    
    col_name: str
    method: DistanceMethod
    label: str