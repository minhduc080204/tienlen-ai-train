from env.tienlen_env import TienLenEnv

def test_first_player_can_play():
    env = TienLenEnv(4)
    state = env.reset()

    player = state.current_player
    card = state.hands[player][0]

    state, reward, done, info = env.step([card])
    assert done is False

