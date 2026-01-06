def compute_reward(prev_state, new_state, player_id, done):
    reward = 0.0

    # ===== 1. Reward khi đánh được bài =====
    prev_hand = prev_state.hands[player_id]
    new_hand = new_state.hands[player_id]

    if len(new_hand) < len(prev_hand):
        reward += 0.1   # đánh được bài

    # ===== 2. Reward khi sắp hết bài =====
    if len(new_hand) <= 2:
        reward += 0.2

    # ===== 3. Reward khi thắng / thua =====
    if done:
        if new_state.winner == player_id:
            reward += 5.0
        else:
            reward -= 5.0

    return reward
