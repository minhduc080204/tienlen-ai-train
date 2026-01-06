from core.card import Card
from state.trick_encoder import encode_trick


def test_empty_trick():
    vec = encode_trick(None)

    assert vec[0] == 1.0
    assert vec.sum() == 1.0

def test_single_card_trick():
    vec = encode_trick([Card("7", "♠")])

    assert vec[1] == 1.0  # SINGLE
    assert vec[7] > 0


def test_pair_trick():
    vec = encode_trick([
        Card("8", "♠"),
        Card("8", "♥"),
    ])

    assert vec[2] == 1.0  # PAIR

def test_four_kind_trick():
    vec = encode_trick([
        Card("6","♠"),
        Card("6","♣"),
        Card("6","♦"),
        Card("6","♥"),
    ])

    assert vec[5] == 1.0  # FOUR_KIND

def test_two_is_choppable():
    vec = encode_trick([Card("2","♣")])

    assert vec[13] == 1.0
