# action/action_mask.py
import numpy as np
from action.action_space import ACTION_SPACE
from action.action_validator import can_apply_action
from core.card import Card
from core.move_type import MoveType


def build_action_mask(
    hand: list[Card],
    current_trick: list[Card] | None
) -> np.ndarray:
    """
    Build action mask cho PPO:
    - 1 = action hợp lệ
    - 0 = action bị cấm
    """

    mask = np.zeros(len(ACTION_SPACE), dtype=np.float32)

    for i, action in enumerate(ACTION_SPACE):

        # -----------------------------
        # ❌ LUẬT QUAN TRỌNG NHẤT
        # Không được PASS khi chưa có trick
        # -----------------------------
        if action.move_type == MoveType.PASS:
            if current_trick is None:
                mask[i] = 0.0
                continue

        # -----------------------------
        # Kiểm tra action có áp dụng được không
        # -----------------------------
        if can_apply_action(
            action=action,
            hand=hand,
            current_trick=current_trick
        ):
            mask[i] = 1.0

    return mask
