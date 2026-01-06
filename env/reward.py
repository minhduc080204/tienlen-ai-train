from core.rules import detect_move_type, MoveType

# --- Terminal reward ---
WIN_REWARD = 30.0
LOSE_PENALTY = -30.0

PLAY_CARD_REWARD = 0.05
PASS_PENALTY = -0.2

OPPONENT_ONE_CARD_PENALTY = -5.0
OPPONENT_ONE_CARD_GOOD_PLAY = +2.0

# --- Cutting "2" (heo) ---
GOOD_CUT_TWO = +8.0
BAD_CUT_TWO = -5.0

# --- Power card management ---
SAVE_POWER_CARD = +2.0
WASTE_POWER_CARD = -3.0


# =========================================================
# 2. TERMINAL REWARD
# =========================================================

def terminal_reward(player_rank: int) -> float:
    """
    Reward khi kết thúc ván

    player_rank:
        1 = về nhất
        2,3,4 = các hạng sau
    """
    if player_rank == 1:
        return WIN_REWARD
    return LOSE_PENALTY


# =========================================================
# 3. ACTION-LEVEL REWARD (REWARD SHAPING)
# =========================================================

def action_reward(
    action_cards,
    prev_state,
    next_state,
    player_id: int
) -> float:
    """
    Reward theo từng nước đi (chưa tính kết thúc ván)
    """

    reward = 0.0

    # -----------------------------------------------------
    # 3.1 ĐÁNH BÀI / PASS
    # -----------------------------------------------------
    if not action_cards:
        reward += PASS_PENALTY
    else:
        reward += PLAY_CARD_REWARD

    # -----------------------------------------------------
    # 3.2 ĐỐI THỦ CÒN 1 LÁ → PHẢI CHẶN
    # -----------------------------------------------------
    opponent_one_card = False
    for pid, hand in enumerate(prev_state.hands):
        if pid != player_id and len(hand) == 1:
            opponent_one_card = True
            break

    if opponent_one_card:
        if not action_cards:
            reward += OPPONENT_ONE_CARD_PENALTY
        else:
            move_type = detect_move_type(action_cards)
            if move_type == MoveType.SINGLE:
                # đánh lẻ nhỏ → ngu
                if action_cards[0].rank_value < 10:
                    reward += OPPONENT_ONE_CARD_PENALTY
                else:
                    reward += OPPONENT_ONE_CARD_GOOD_PLAY

    # -----------------------------------------------------
    # 3.3 CHẶT HEO (RISK vs REWARD)
    # -----------------------------------------------------
    prev_trick = prev_state.current_trick

    if prev_trick and detect_move_type(prev_trick) == MoveType.TWO:
        move_type = detect_move_type(action_cards)

        if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT]:
            reward += GOOD_CUT_TWO

        elif move_type == MoveType.TWO:
            # heo chặt heo → so chất
            if action_cards[0].suit_value > prev_trick[0].suit_value:
                reward += GOOD_CUT_TWO
            else:
                reward += BAD_CUT_TWO

    # -----------------------------------------------------
    # 3.4 QUẢN LÝ BÀI MẠNH CUỐI GAME
    # -----------------------------------------------------
    remaining_cards = len(prev_state.hands[player_id])

    if remaining_cards <= 5:
        move_type = detect_move_type(action_cards)

        # phá bài mạnh quá sớm
        if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT]:
            reward += WASTE_POWER_CARD

        # đánh rác để giữ bài mạnh
        elif move_type == MoveType.SINGLE and action_cards[0].rank_value < 5:
            reward += SAVE_POWER_CARD

    return reward


# =========================================================
# 4. FULL REWARD FUNCTION (DÙNG TRONG env.step)
# =========================================================

def compute_reward(
    action_cards,
    prev_state,
    next_state,
    done: bool,
    player_id: int,
    player_rank: int | None = None
) -> float:
    """
    Reward cuối cùng cho PPO
    """

    reward = action_reward(
        action_cards=action_cards,
        prev_state=prev_state,
        next_state=next_state,
        player_id=player_id
    )

    if done:
        reward += terminal_reward(player_rank)

    return reward
