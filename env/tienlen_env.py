from core.deck import Deck
from core.rules import can_beat
from core.starting_rules import find_starting_player
from core.instant_win import is_six_pairs, is_five_double_straight
from env.game_state import GameState
from env.step_result import StepResult


class TienLenEnv:
    def __init__(self, num_players: int = 4):
        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.state: GameState | None = None

    def reset(self):
        deck = Deck()
        deck.shuffle()
        hands = deck.deal(self.num_players)

        # ăn trắng
        for i, hand in enumerate(hands):
            if is_six_pairs(hand) or is_five_double_straight(hand):
                self.state = GameState(
                    hands=hands,
                    current_player=i,
                    finished=True,
                    winner=i,
                )
                return self.state

        start_player = find_starting_player(hands)

        self.state = GameState(
            hands=hands,
            current_player=start_player,
            current_trick=None,
            last_player=None,
            finished=False,
            winner=None,
        )
        return self.state

    def step(self, action_cards: list) -> StepResult:
        state = self.state
        assert state is not None
        assert not state.finished

        player = state.current_player
        hand = state.hands[player]

        # =====================
        # PASS
        # =====================
        if not action_cards:
            if state.current_trick is None:
                return StepResult(
                    state=state,
                    reward=-0.1,
                    done=False,
                    info={"action": "INVALID_PASS"}
                )


            next_player = (player + 1) % self.num_players

            # Vòng mới (mọi người đều pass)
            if next_player == state.last_player:
                state.current_trick = None
                state.last_player = None

            state.current_player = next_player

            return StepResult(
                state=state,
                reward=0.0,
                done=False,
                info={"action": "PASS"}
            )

        # =====================
        # CHECK BÀI CÓ TRONG TAY
        # =====================
        for c in action_cards:
            assert c in hand

        # =====================
        # CHECK HỢP LỆ
        # =====================
        if state.current_trick is not None:
            assert can_beat(state.current_trick, action_cards)

        # =====================
        # ĐÁNH BÀI
        # =====================
        for c in action_cards:
            hand.remove(c)

        state.current_trick = action_cards
        state.last_player = player

        # =====================
        # THẮNG
        # =====================
        if len(hand) == 0:
            state.finished = True
            state.winner = player

            return StepResult(
                state=state,
                reward=0.0,   # reward học tính ở Phase 5
                done=True,
                info={"winner": player}
            )

        # =====================
        # NEXT PLAYER
        # =====================
        state.current_player = (player + 1) % self.num_players

        return StepResult(
            state=state,
            reward=0.0,
            done=False,
            info={}
        )
