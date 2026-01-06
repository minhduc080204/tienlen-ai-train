from core.move_type import MoveType
from core.rules import (
    get_legal_moves,
    detect_move_type
)

def resolve_action(action_spec, hand, current_trick):
    """
    Từ ActionSpec + hand hiện tại
    → trả về cards cụ thể hợp lệ
    """

    legal_moves = get_legal_moves(hand, current_trick)

    # PASS
    if action_spec.move_type == MoveType.PASS:
        return []

    # lọc theo move_type + length
    candidates = [
        move for move in legal_moves
        if (
            detect_move_type(move) == action_spec.move_type
            and len(move) == action_spec.length
        )
    ]

    if not candidates:
        return []

    # chọn bài nhỏ nhất (an toàn)
    candidates.sort(
        key=lambda cards: min(c.rank_value for c in cards)
    )

    return candidates[0]
