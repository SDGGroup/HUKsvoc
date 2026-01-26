from dataclasses import dataclass
from svoc.automatic.enums import DistanceMethod

@dataclass(frozen=True)
class Distance:
    col_name: str
    method: DistanceMethod
    label: str