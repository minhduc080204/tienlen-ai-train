# action/action_validator.py
from collections import Counter
from core.card import Card
from core.rules import can_beat
from action.action_spec import ActionSpec


def can_apply_action(
    action: ActionSpec,
    hand: list[Card],
    current_trick: list[Card] | None
) -> bool:

    # PASS
    if action.move_type == "PASS":
        return current_trick is not None

    # đủ bài không?
    rank_counter = Counter(c.rank_value for c in hand)
    for r in action.ranks:
        if rank_counter[r] < action.ranks.count(r):
            return False

    # lấy cards giả (chỉ cần rank)
    cards = []
    for r in action.ranks:
        for c in hand:
            if c.rank_value == r and c not in cards:
                cards.append(c)
                break

    # đánh đầu vòng
    if current_trick is None:
        return True

    # chặt / đè
    return can_beat(current_trick, cards)
