from core.card import Card
from state.opponent_encoder import encode_opponents


def test_encode_opponents_shape():
    opponents = [
        [Card("3","♠")] * 5,
        [Card("4","♠")] * 2,
        [Card("5","♠")] * 7,
    ]

    vec = encode_opponents(
        opponent_hands=opponents,
        player_id=0,
        num_players=4,
    )

    assert vec.shape == (12,)
    assert vec[1] == 2 / 13
    assert vec[5] == 1.0  # có người <= 2 lá
