from dataclasses import dataclass
from core.card import Card

@dataclass
class GameState:
    hands: list[list[Card]]
    current_player: int
    current_trick: list[Card] | None = None
    last_player: int | None = None
    finished: bool = False
    winner: int | None = None
