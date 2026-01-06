# utils/encoding.py
import numpy as np
from core.card import Card

def encode_hand(cards: list[Card]) -> np.ndarray:
    vec = np.zeros(52, dtype=np.int8)
    for c in cards:
        vec[c.card_id] = 1
    return vec
