# tests/test_hand_encoder.py
import numpy as np
from core.card import Card
from state.hand_encoder import encode_hand


def test_encode_hand_shape():
    hand = [
        Card("3", "♠"),
        Card("A", "♥"),
        Card("2", "♣"),
    ]

    vec = encode_hand(hand)

    assert vec.shape == (52,)
    assert vec.dtype == np.float32

def test_encode_hand_correct_mapping():
    card = Card("3", "♠")  # rank_value=0, suit_value=0 → id=0
    vec = encode_hand([card])

    assert vec[0] == 1.0
    assert vec.sum() == 1.0

def test_encode_hand_multiple_cards():
    hand = [
        Card("3", "♠"),
        Card("3", "♥"),
        Card("2", "♠"),
    ]

    vec = encode_hand(hand)

    ids = [c.card_id for c in hand]
    for i in ids:
        assert vec[i] == 1.0

    assert vec.sum() == 3.0

def test_encode_empty_hand():
    vec = encode_hand([])

    assert vec.sum() == 0.0
