# tests/test_env_gameplay.py
from env.tienlen_env import TienLenEnv

def test_valid_pass_after_trick():
    env = TienLenEnv(2)
    state = env.reset()

    p0 = state.current_player
    card = state.hands[p0][0]

    # player 0 đánh
    state, reward, done, info = env.step([card])
    assert done is False

    # player 1 PASS (hợp lệ vì đã có trick)
    state, reward, done, info = env.step([])
    assert done is False
