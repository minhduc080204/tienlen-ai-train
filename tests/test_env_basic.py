# tests/test_env_basic.py
from env.tienlen_env import TienLenEnv

def test_env_reset():
    env = TienLenEnv(4)
    state = env.reset()

    assert len(state.hands) == 4

    # người đánh đầu phải là người có 3♣
    starter = state.current_player
    hand = state.hands[starter]

    assert any(c.rank == "3" and c.suit == "♣" for c in hand)

