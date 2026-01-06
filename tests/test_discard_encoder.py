# tests/test_discard_encoder.py
import numpy as np
from core.card import Card
from state.discard_encoder import encode_discard, encode_discard_pile


def test_encode_discard_shape():
    discard = [
        Card("3", "♠"),
        Card("A", "♥"),
    ]

    vec = encode_discard(discard)

    assert vec.shape == (52,)
    assert vec.dtype == np.float32

def test_encode_discard_mapping():
    card = Card("2", "♣")  # highest card
    vec = encode_discard([card])

    assert vec[card.card_id] == 1.0
    assert vec.sum() == 1.0

def test_encode_discard_multiple_rounds():
    discard = [
        Card("3", "♠"),
        Card("3", "♥"),
        Card("3", "♠"),  # duplicated appearance
    ]

    vec = encode_discard(discard)

    assert vec.sum() == 2.0

def test_encode_empty_discard():
    vec = encode_discard([])

    assert vec.sum() == 0.0

def test_discard_vector_size():
    vec = encode_discard_pile([])
    assert vec.shape == (60,)
