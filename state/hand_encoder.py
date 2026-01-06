# state/hand_encoder.py
import numpy as np
from core.card import Card

HAND_VECTOR_SIZE = 52


def encode_hand(hand: list[Card]) -> np.ndarray:
    """
    Encode hand cards into a binary vector of size 52.
    """
    vec = np.zeros(HAND_VECTOR_SIZE, dtype=np.float32)

    for card in hand:
        vec[card.card_id] = 1.0

    return vec
