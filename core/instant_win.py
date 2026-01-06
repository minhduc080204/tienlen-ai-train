from collections import Counter
from core.card import Card

def is_six_pairs(hand: list[Card]) -> bool:
    if len(hand) != 13:
        return False
    counter = Counter(c.rank_value for c in hand)
    return list(counter.values()).count(2) == 6

def is_five_double_straight(hand: list[Card]) -> bool:
    ranks = sorted(c.rank_value for c in hand)
    counter = Counter(ranks)

    pairs = [r for r, v in counter.items() if v >= 2 and r != 12]
    pairs.sort()

    longest = 1
    curr = 1
    for i in range(1, len(pairs)):
        if pairs[i] == pairs[i - 1] + 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1

    return longest >= 5
