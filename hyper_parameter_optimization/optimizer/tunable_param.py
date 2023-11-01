from enum import IntEnum, auto
from dataclasses import dataclass


class TunableDataType(IntEnum):
    float = auto()


class TunableParameter:
    type: TunableDataType


@dataclass
class TunableFloat(TunableParameter):
    min: float
    max: float
    type: TunableDataType = TunableDataType.float
    init: float = None
