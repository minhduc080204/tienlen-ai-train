from core.card import Card
from env.tienlen_env import TienLenEnv
from env.game_state import GameState

def test_game_end():
    env = TienLenEnv(2)

    env.state = GameState(
        hands=[
            [],
            [Card("3","♠")]
        ],
        current_player=0,
        finished=True,
        winner=0
    )

    assert env.state.finished
