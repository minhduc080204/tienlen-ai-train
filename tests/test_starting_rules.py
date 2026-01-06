from core.card import Card
from core.starting_rules import find_starting_player

def test_find_starting_player():
    hands = [
        [Card("4", "♠")],
        [Card("3", "♣")],
        [Card("5", "♦")]
    ]
    assert find_starting_player(hands) == 1

def test_get_legal_moves_basic():
    from core.card import Card
    from core.rules import get_legal_moves

    hand = [
        Card("6","♠"),
        Card("6","♣"),
        Card("6","♦"),
        Card("6","♥"),
        Card("2","♠"),
    ]

    # không có bài trước
    moves = get_legal_moves(hand, None)
    assert any(len(m) == 4 for m in moves)  # tứ quý
    assert [] not in moves                  # không được PASS

    # có heo
    moves = get_legal_moves(hand, [Card("2","♣")])
    assert any(len(m) == 4 for m in moves)  # chặt heo
    assert [] in moves                      # PASS hợp lệ

