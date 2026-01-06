# state/trick_encoder.py
import numpy as np
from core.card import Card
from core.rules import detect_move_type
from core.move_type import MoveType

TRICK_VECTOR_SIZE = 17


def encode_trick(cards: list[Card] | None) -> np.ndarray:
    vec = np.zeros(TRICK_VECTOR_SIZE, dtype=np.float32)

    # no trick
    if not cards:
        vec[0] = 1.0
        return vec

    move_type = detect_move_type(cards)

    type_map = {
        MoveType.SINGLE: 0,
        MoveType.PAIR: 1,
        MoveType.TRIPLE: 2,
        MoveType.STRAIGHT: 3,
        MoveType.FOUR_OF_KIND: 4,
        MoveType.DOUBLE_STRAIGHT: 5,
    }

    if move_type in type_map:
        vec[1 + type_map[move_type]] = 1.0

    # ---- main rank ----
    max_rank = max(c.rank_value for c in cards)
    vec[7] = max_rank / 12.0

    # ---- highest suit ----
    highest = max(cards, key=lambda c: (c.rank_value, c.suit_value))
    vec[8 + highest.suit_value] = 1.0

    # ---- trick length ----
    vec[12] = len(cards) / 13.0

    # ---- is TWO ----
    if max_rank == 12:
        vec[13] = 1.0

    return vec
