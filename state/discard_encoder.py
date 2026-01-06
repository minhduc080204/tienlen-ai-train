import numpy as np
from collections import Counter
from core.card import Card

DISCARD_VECTOR_SIZE = 52


def encode_discard(discard_pile: list[Card]) -> np.ndarray:
    """
    Encode all played cards into a binary vector of size 52.
    """
    vec = np.zeros(DISCARD_VECTOR_SIZE, dtype=np.float32)

    for card in discard_pile:
        vec[card.card_id] = 1.0

    return vec

def encode_discard_pile(discard_pile: list[Card]) -> np.ndarray:
    """
    Encode các lá đã được đánh (public memory)
    """
    vec = np.zeros(60, dtype=np.float32)

    # =====================
    # 1️⃣ Card-level memory (52)
    # =====================
    for card in discard_pile:
        vec[card.card_id] = 1.0

    # =====================
    # 2️⃣ Strategic flags (8)
    # =====================
    offset = 52

    twos = [c for c in discard_pile if c.rank_value == 12]

    # từng heo đã ra
    for c in twos:
        vec[offset + c.suit_value] = 1.0

    # tứ quý đã xuất hiện
    rank_counter = Counter(c.rank_value for c in discard_pile)
    if any(v == 4 for v in rank_counter.values()):
        vec[offset + 4] = 1.0

    # đôi thông ≥ 3
    pairs = [r for r, v in rank_counter.items() if v >= 2 and r < 12]
    pairs.sort()

    for i in range(len(pairs) - 2):
        if pairs[i] + 1 == pairs[i + 1] and pairs[i + 1] + 1 == pairs[i + 2]:
            vec[offset + 5] = 1.0
            break

    # tỉ lệ bài đã đánh
    vec[offset + 6] = len(discard_pile) / 52.0

    # còn heo chưa ra
    vec[offset + 7] = 1.0 if len(twos) < 4 else 0.0

    return vec
