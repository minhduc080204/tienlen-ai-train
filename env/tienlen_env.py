# env/tienlen_env.py
import copy
from core.deck import Deck
from core.rules import can_beat
from core.starting_rules import find_starting_player
from core.instant_win import is_six_pairs, is_five_double_straight
from env.game_state import GameState
from env.step_result import StepResult
from env.reward import compute_reward


class TienLenEnv:
    def __init__(self, num_players: int = 4):
        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.state: GameState | None = None

    def reset(self) -> GameState:
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
                    discard_pile=[],
                    finished_order=[i],
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
            discard_pile=[],
            finished_order=[],
        )
        return self.state

    def _next_active_player(self, from_player: int) -> int:
        """Tìm player tiếp theo còn bài (chưa về)."""
        for offset in range(1, self.num_players + 1):
            candidate = (from_player + offset) % self.num_players
            if len(self.state.hands[candidate]) > 0:
                return candidate
        return from_player  # fallback (không nên xảy ra)

    def _active_players(self) -> list[int]:
        """Danh sách player còn bài."""
        return [i for i in range(self.num_players) if len(self.state.hands[i]) > 0]

    def step(self, action_cards: list) -> StepResult:
        state = self.state
        assert state is not None
        assert not state.finished

        player = state.current_player
        hand = state.hands[player]

        # COPY STATE TRƯỚC ACTION (CHO REWARD)
        prev_state = copy.deepcopy(state)

        # =====================
        # PASS
        # =====================
        if not action_cards:
            if state.current_trick is None:
                # không được pass khi không có trick trước
                return StepResult(
                    state=state,
                    reward=-100.0,
                    done=True,
                    info={"action": "INVALID_PASS_TERMINATED"}
                )

            # Thêm player vào danh sách passed (nếu chưa có)
            if player not in state.passed_players:
                state.passed_players.append(player)

            next_player = self._next_active_player(player)

            # Nếu vòng quanh về tới người đánh trick → clear trick
            if next_player == state.last_player:
                state.current_trick = None
                state.last_player = None
                state.passed_players = []

            state.current_player = next_player

            reward = compute_reward(
                action_cards=[],
                prev_state=prev_state,
                next_state=state,
                done=False,
                player_id=player,
                player_rank=None
            )

            return StepResult(
                state=state,
                reward=reward,
                done=False,
                info={"action": "PASS"}
            )

        # =====================
        # CHECK HỢP LỆ
        # =====================
        if state.current_trick is not None:
            assert can_beat(state.current_trick, action_cards)

        # =====================
        # REMOVE BÀI KHỎI TAY
        # =====================
        for c in action_cards:
            for h in hand:
                if h.rank == c.rank and h.suit == c.suit:
                    hand.remove(h)
                    break
            else:
                raise RuntimeError(
                    f"❌ Card {c} not found in player {player}'s hand"
                )

        # Tích lũy vào discard pile
        state.discard_pile.extend(action_cards)

        # Cập nhật trick và reset passed list
        state.current_trick = action_cards
        state.last_player = player
        state.passed_players = []

        # =====================
        # CHECK WIN / RANK
        # =====================
        player_done = len(hand) == 0
        player_rank = None
        game_over = False

        if player_done:
            # Ghi nhận thứ tự về bài
            state.finished_order.append(player)
            player_rank = len(state.finished_order)  # rank 1, 2, 3...

            active_remaining = self._active_players()

            if len(active_remaining) <= 1:
                # Chỉ còn 0 hoặc 1 người — game kết thúc
                if len(active_remaining) == 1:
                    # Người cuối cùng = rank cuối (rank 4 với 4 player)
                    last_player = active_remaining[0]
                    state.finished_order.append(last_player)

                state.finished = True
                state.winner = state.finished_order[0]  # rank 1 = winner
                game_over = True
            else:
                # Game chưa kết thúc — chuyển sang player tiếp theo còn bài
                state.current_player = self._next_active_player(player)

        else:
            state.current_player = self._next_active_player(player)

        # =====================
        # REWARD
        # =====================
        reward = compute_reward(
            action_cards=action_cards,
            prev_state=prev_state,
            next_state=state,
            done=player_done,
            player_id=player,
            player_rank=player_rank
        )

        return StepResult(
            state=state,
            reward=reward,
            done=game_over,
            info={"rank": player_rank, "winner": state.winner} if game_over else
                 {"rank": player_rank} if player_done else {}
        )

