from core.card import Card
from action.action_mask import build_action_mask
from action.action_space import ACTION_SPACE


def test_illegal_pair_not_allowed():
    hand = [Card("3","♠"), Card("4","♠")]
    mask = build_action_mask(hand, None)

    pair_3 = next(
        i for i,a in enumerate(ACTION_SPACE)
        if a.move_type == "PAIR" and a.ranks == [0,0]
    )

    assert mask[pair_3] == 0


def test_four_kind_chops_two():
    hand = [
        Card("6","♠"),Card("6","♣"),
        Card("6","♦"),Card("6","♥"),
    ]

    trick = [Card("2","♠")]

    mask = build_action_mask(hand, trick)

    four_6 = next(
        i for i,a in enumerate(ACTION_SPACE)
        if a.move_type == "FOUR_KIND" and a.ranks == [3,3,3,3]
    )

    assert mask[four_6] == 1
