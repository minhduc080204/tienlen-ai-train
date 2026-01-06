from core.card import Card
from core.instant_win import is_six_pairs, is_five_double_straight

def test_six_pairs():
    hand = []
    for r in ["3", "4", "5", "6", "7", "8"]:
        hand += [Card(r, "♠"), Card(r, "♥")]
    hand.append(Card("A", "♠"))
    assert is_six_pairs(hand)

def test_five_double_straight():
    hand = []
    for r in ["3", "4", "5", "6", "7"]:
        hand += [Card(r, "♠"), Card(r, "♥")]
    hand += [Card("K", "♠"), Card("A", "♠"), Card("A", "♥")]
    assert is_five_double_straight(hand)
