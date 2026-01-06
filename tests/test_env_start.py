from env.tienlen_env import TienLenEnv

def test_env_reset():
    env = TienLenEnv(4)
    state = env.reset()

    assert len(state.hands) == 4
    assert state.current_player in range(4)
