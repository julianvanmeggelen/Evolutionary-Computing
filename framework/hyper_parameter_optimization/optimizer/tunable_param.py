from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List


class TunableDataType(IntEnum):
    float = auto()
    category = auto()


class TunableParameter:
    type: TunableDataType


@dataclass
class TunableFloat(TunableParameter):
    min: float
    max: float
    type: TunableDataType = TunableDataType.float
    init: float = None

@dataclass
class TunableCategory(TunableParameter):
    categories: List[str]
    type: TunableDataType = TunableDataType.category
    init: str = None
