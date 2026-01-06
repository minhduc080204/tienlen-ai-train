# action/action_spec.py
from dataclasses import dataclass
from typing import List
from core.move_type import MoveType

@dataclass(frozen=True)
class ActionSpec:
    """
    Mô tả 1 hành động trừu tượng
    """
    move_type: MoveType
    ranks: List[int]          # rank_value (0–12)
    length: int               # số lá dùng
