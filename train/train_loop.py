# train/train_loop.py

import random
import torch

from env.tienlen_env import TienLenEnv
from state.state_encoder import encode_state
from action.action_mask import build_action_mask
from action.action_space import ACTION_SPACE

from rl.agent import PPOAgent
from rl.model import TienLenPolicy
from rl.buffer import RolloutBuffer
from rl.rewards import compute_reward

from core.action_executor import resolve_action
from env.step_result import StepResult

from bots.rule_bot import RuleBot
from train.config import *
from state.state_dim import STATE_DIM

def train():
    print("🚀 Start PPO Training")

    # =========================
    # 1️⃣ INIT
    # =========================
    env = TienLenEnv(num_players=NUM_PLAYERS)

    model = TienLenPolicy(
        state_dim=STATE_DIM,
        action_dim=len(ACTION_SPACE)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        gamma=GAMMA,
        clip_eps=0.2
    )


    rule_bot = RuleBot(player_id=1)
    buffer = RolloutBuffer()

    # =========================
    # 2️⃣ TRAIN LOOP
    # =========================
    for episode in range(1, MAX_EPISODES + 1):

        state = env.reset()

        # instant win → skip
        if state.finished:
            continue

        buffer.clear()
        done = False
        turn_count = 0

        # =========================
        # 3️⃣ PLAY ONE GAME
        # =========================
        while not done:
            turn_count += 1
            current_pid = env.state.current_player

            # =====================
            # AI TURN
            # =====================
            if current_pid == AI_PLAYER_ID:
                player_hand = state.hands[AI_PLAYER_ID]

                opponent_hands = [
                    hand
                    for pid, hand in enumerate(state.hands)
                    if pid != AI_PLAYER_ID
                ]

                # discard_pile = env.discard_pile   # hoặc [] nếu chưa có
                discard_pile = []   # hoặc [] nếu chưa có

                state_vec = encode_state(
                    hand=player_hand,
                    discard_pile=discard_pile,
                    opponent_hands=opponent_hands,
                    current_trick=state.current_trick,
                    player_id=AI_PLAYER_ID,
                    num_players=NUM_PLAYERS
                )

                state_tensor = torch.tensor(
                    state_vec,
                    dtype=torch.float32
                ).unsqueeze(0)   # (1, STATE_DIM)

                action_mask = build_action_mask(
                    hand=env.state.hands[AI_PLAYER_ID],
                    current_trick=env.state.current_trick
                )

                action_mask = torch.tensor(
                    action_mask,
                    dtype=torch.float32
                ).unsqueeze(0)   # (1, action_dim)


                action_id, logprob, value = agent.act(
                    state_tensor,
                    action_mask
                )

                action_spec = ACTION_SPACE[action_id]

                action_cards = resolve_action(
                    action_spec=action_spec,
                    hand=env.state.hands[AI_PLAYER_ID],
                    current_trick=env.state.current_trick
                )


            # =====================
            # OPPONENT TURN
            # =====================
            else:
                action_cards = rule_bot.select_action(
                    state,
                    current_pid
                )

            # =====================
            # ENV STEP
            # =====================
            step_result: StepResult = env.step(action_cards)

            reward = compute_reward(
                prev_state=state,
                new_state=step_result.state,
                player_id=AI_PLAYER_ID,
                done=step_result.done
            )

            # =====================
            # STORE ROLLOUT (AI ONLY)
            # =====================
            if current_pid == AI_PLAYER_ID:
                buffer.add(
                    state=state_vec,
                    action=action_id,
                    logprob=logprob.detach(),
                    reward=reward,
                    done=step_result.done,
                    value=value.detach(),
                    action_mask=action_mask
                )

            state = step_result.state
            done = step_result.done

            # =========================
            # 4️⃣ PPO UPDATE
            # =========================
            if len(buffer) > 0:
                advantages, returns = buffer.compute_gae(
                    gamma=GAMMA,
                    lam=LAMBDA
                )

                agent.update(
                    states=buffer.states,
                    actions=buffer.actions,
                    old_logprobs=buffer.logprobs,
                    returns=returns,
                    advantages=advantages,
                    action_masks=buffer.action_masks,
                    epochs=PPO_EPOCHS,
                    batch_size=BATCH_SIZE
                )

                buffer.clear()


        # =========================
        # 5️⃣ LOG
        # =========================
        if episode % 50 == 0:
            win = state.winner == AI_PLAYER_ID
            print(
                f"[Episode {episode}] "
                f"Turns={turn_count} "
                f"Win={win}"
            )

        # =========================
        # 6️⃣ SAVE CHECKPOINT
        # =========================
        if episode % SAVE_EVERY == 0:
            path = f"checkpoints/ppo_ep{episode}.pt"
            agent.save(path)
            print(f"💾 Saved checkpoint: {path}")

    print("✅ Training finished")


if __name__ == "__main__":
    train()

# def train():
#     AI_PLAYER_ID = 0

#     env = TienLenEnv(num_players=NUM_PLAYERS)

#     model = TienLenPolicy(
#         state_dim=STATE_DIM,
#         action_dim=len(ACTION_SPACE)
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     agent = PPOAgent(
#         model=model,
#         optimizer=optimizer,
#     )

#     buffer = RolloutBuffer()
#     rule_bot = RuleBot(player_id=1)

#     print("🚀 Start PPO Training...")

#     for episode in range(1, MAX_EPISODES + 1):

#         state = env.reset()
#         buffer.clear()

#         if state.finished:
#             continue

#         done = False
#         turn_count = 0

#         while not done:
#             turn_count += 1
#             current_pid = env.state.current_player

#             # ================= AI TURN =================
#             if current_pid == AI_PLAYER_ID:
#                 state = env.state
                
#                 player_hand = state.hands[AI_PLAYER_ID]

#                 opponent_hands = [
#                     h for i, h in enumerate(state["player_hands"])
#                     if i != AI_PLAYER_ID
#                 ]
#                 current_trick = state["current_trick"]
#                 num_players = state["num_players"]

#                 state_vec = encode_state(
#                     player_hand,
#                     opponent_hands,
#                     current_trick,
#                     AI_PLAYER_ID,
#                     num_players
#                 )

#                 action_mask = build_action_mask(
#                     hand=env.state.hands[AI_PLAYER_ID],
#                     current_trick=env.state.current_trick
#                 )

#                 action_id, logprob, value = agent.act(
#                     state_vec,
#                     action_mask
#                 )

#                 action_spec = ACTION_SPACE[action_id]
#                 action_cards = action_spec.resolve(
#                     env.state.hands[AI_PLAYER_ID]
#                 )

#             # ================= OPPONENT =================
#             else:
#                 action_cards = rule_bot.act(env.state)

#             prev_state = env.state
#             step_result = env.step(action_cards)

#             reward = compute_reward(
#                 prev_state=state,
#                 new_state=step_result.state,
#                 player_id=agent.player_id
#             )

#             state = step_result.state
#             done = step_result.done


#             # reward = compute_reward(
#             #     prev_state=prev_state,
#             #     new_state=step_result.state,
#             #     player_id=AI_PLAYER_ID
#             # )

#             # if current_pid == AI_PLAYER_ID:
#             #     buffer.add(
#             #         state=state_vec,
#             #         action=action_id,
#             #         logprob=logprob,
#             #         reward=reward,
#             #         done=step_result.done,
#             #         value=value,
#             #         action_mask=action_mask
#             #     )

#             # done = step_result.done

#         # ================= PPO UPDATE =================
#         if len(buffer) > 0:
#             advantages, returns = buffer.compute_gae(
#                 gamma=GAMMA,
#                 lam=LAMBDA
#             )

#             agent.update(
#                 states=buffer.states,
#                 actions=buffer.actions,
#                 old_logprobs=buffer.logprobs,
#                 returns=returns,
#                 advantages=advantages,
#                 action_masks=buffer.action_masks,
#                 epochs=PPO_EPOCHS,
#                 batch_size=BATCH_SIZE
#             )

#         if episode % 100 == 0:
#             print(f"[Episode {episode}] Turns={turn_count}")

#     print("✅ Training finished")


# if __name__ == "__main__":
#     train()
