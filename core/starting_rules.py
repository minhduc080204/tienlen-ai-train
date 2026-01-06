# from core.card import Card

# def has_3_club(hand: list[Card]) -> bool:
#     return any(c.rank == "3" and c.suit == "♣" for c in hand)

# def find_starting_player(hands: list[list[Card]]) -> int:
#     for i, hand in enumerate(hands):
#         if has_3_club(hand):
#             return i
#     raise ValueError("No 3♣ found in any hand")

from core.card import Card

def find_starting_player(hands: list[list[Card]]) -> int:
    """
    Luật:
    - Có 3♣ → người đó đi trước
    - Không có 3♣ → người có lá nhỏ nhất đi trước
    """

    # 1️⃣ tìm 3♣
    for pid, hand in enumerate(hands):
        for card in hand:
            if card.rank == "3" and card.suit == "♣":
                return pid

    # 2️⃣ không có 3♣ → tìm lá nhỏ nhất
    min_card = None
    start_player = 0

    for pid, hand in enumerate(hands):
        for card in hand:
            if min_card is None:
                min_card = card
                start_player = pid
            else:
                if (
                    card.rank_value < min_card.rank_value or
                    (
                        card.rank_value == min_card.rank_value
                        and card.suit_value < min_card.suit_value
                    )
                ):
                    min_card = card
                    start_player = pid

    return start_player
