# # state/opponent_encoder.py
# import numpy as np


# def encode_opponents(
#     opponent_hands: list[list],
#     current_player: int,
#     num_players: int
# ) -> np.ndarray:
#     """
#     Encode trạng thái đối thủ (tối đa 3)
#     """
#     vec = np.zeros(12, dtype=np.float32)

#     idx = 0
#     for i in range(num_players):
#         if i == current_player:
#             continue

#         hand = opponent_hands[i]
#         num_cards = len(hand)

#         # 1️⃣ số lá (normalize)
#         vec[idx] = num_cards / 13.0

#         # 2️⃣ danger flag
#         vec[idx + 1] = 1.0 if num_cards <= 2 else 0.0

#         # 3️⃣ is next player
#         next_player = (current_player + 1) % num_players
#         vec[idx + 2] = 1.0 if i == next_player else 0.0

#         # 4️⃣ relative position (clockwise distance)
#         distance = (i - current_player) % num_players
#         vec[idx + 3] = distance / num_players

#         idx += 4

#     return vec

# state/opponent_encoder.py
import numpy as np
from core.card import Card


def encode_opponents(
    opponent_hands: list[list[Card]],
    player_id: int,
    num_players: int,
) -> np.ndarray:
    """
    Encode thông tin công khai của đối thủ
    """

    assert num_players in [2, 3, 4]
    assert len(opponent_hands) == num_players - 1

    vec = np.zeros(12, dtype=np.float32)

    # --------------------
    # số lá của từng đối thủ (tối đa 3)
    # --------------------
    for i, hand in enumerate(opponent_hands):
        vec[i] = len(hand) / 13.0

    # --------------------
    # min / max bài còn lại
    # --------------------
    counts = [len(h) for h in opponent_hands]
    vec[3] = min(counts) / 13.0
    vec[4] = max(counts) / 13.0

    # --------------------
    # có ai sắp hết bài không
    # --------------------
    vec[5] = 1.0 if any(c <= 2 for c in counts) else 0.0

    # --------------------
    # padding cho game < 4 người
    # --------------------
    # vec[6] → vec[11] để trống (0)

    return vec
