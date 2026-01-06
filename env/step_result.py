from dataclasses import dataclass
from env.game_state import GameState

@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict
