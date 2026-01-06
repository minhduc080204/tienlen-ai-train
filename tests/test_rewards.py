from rl.rewards import compute_reward

def test_win_reward():
    r = compute_reward(
        env=None,
        player_id=0,
        done=True,
        winner=0
    )
    assert r == 5.0

class DummyEnv:
    last_move_player = 0
    last_move_chopped = True
    player_hands = {0: [1,2], 1: [3]}

def test_reward_with_env():
    env = DummyEnv()
    r = compute_reward(env, player_id=0, done=False, winner=None)

    assert r > 0
