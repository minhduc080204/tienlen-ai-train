from enum import Enum, auto

class MoveType(Enum):
    PASS = auto()
    SINGLE = auto()
    PAIR = auto()
    TRIPLE = auto()
    STRAIGHT = auto()
    DOUBLE_STRAIGHT = auto()
    FOUR_OF_KIND = auto()
    TWO = auto()
