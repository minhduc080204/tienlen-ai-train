# bots/rule_bot.py

from core.rules import (
    get_legal_moves,
    detect_move_type
)
from core.move_type import MoveType


class RuleBot:
    """
    Rule-based bot:
    - Chỉ đánh nước hợp lệ
    - Ưu tiên bài nhỏ
    - Dùng làm đối thủ train PPO
    """

    def __init__(self, player_id: int):
        self.player_id = player_id

    # ========================
    # PUBLIC API (chuẩn hoá)
    # ========================
    def act(self, state):
        """
        Train loop sẽ gọi hàm này
        """
        return self.select_action(state, self.player_id)

    # ========================
    # CORE LOGIC
    # ========================
    def select_action(self, state, player_id):
        hand = state.hands[player_id]
        current_trick = state.current_trick

        # 1️⃣ Không có bài → PASS
        if not hand:
            return []

        # 2️⃣ Lấy danh sách nước hợp lệ
        legal_moves = [
            move for move in get_legal_moves(hand, current_trick)
            if len(move) > 0
        ]

        # 3️⃣ Không có nước đi → PASS
        if not legal_moves:
            return []

        # 4️⃣ Sắp xếp theo độ ưu tiên
        legal_moves.sort(
            key=lambda cards: (
                self._priority(cards),
                self._min_rank(cards)
            )
        )

        return legal_moves[0]

    # ------------------------
    # Helpers
    # ------------------------

    def _priority(self, cards):
        move_type = detect_move_type(cards)

        priority = {
            MoveType.SINGLE: 0,
            MoveType.PAIR: 1,
            MoveType.TRIPLE: 2,
            MoveType.STRAIGHT: 3,
            MoveType.DOUBLE_STRAIGHT: 4,
            MoveType.FOUR_OF_KIND: 5,
            MoveType.TWO: 6,
        }
        return priority.get(move_type, 99)

    def _min_rank(self, cards):
        return min(card.rank for card in cards)
