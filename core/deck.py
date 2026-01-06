# core/deck.py
import random
from core.card import Card, RANKS, SUITS

class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in RANKS for suit in SUITS]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_players: int, cards_per_player: int = 13):
        assert num_players in [2, 3, 4]

        self.shuffle()
        
        total_needed = num_players * cards_per_player
        assert total_needed <= len(self.cards)

        hands = [[] for _ in range(num_players)]

        for i in range(total_needed):
            hands[i % num_players].append(self.cards[i])

        return hands
