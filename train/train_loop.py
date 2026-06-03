import math
import torch
import os
import copy
import random
import argparse
import numpy as np
from collections import deque
from typing import List

import train.config as config
from env.tienlen_env import TienLenEnv
from state.state_encoder import encode_state
from state.state_dim import STATE_DIM
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.action_executor import resolve_action
from core.rules import get_legal_moves, detect_move_type

from rl.agent import PPOAgent
from rl.model import TienLenPolicy
from rl.buffer import RolloutBuffer
from bots.rule_bot import RuleBot
from utils.logger import setup_logger
from utils.metrics import MetricTracker


def parse_args():
    parser = argparse.ArgumentParser("PPO Multi-Phase Training for Tien Len")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Tổng số episodes cần train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cpu | cuda | auto")
    parser.add_argument("--init-model-path", type=str, default=None,
                        help="Path tới checkpoint để tiếp tục train (bỏ qua warmup nếu cung cấp)")

    # --- Phase overrides ---
    parser.add_argument("--warmup-episodes", type=int, default=None,
                        help=f"Override WARMUP_EPISODES (default: {config.WARMUP_EPISODES})")
    parser.add_argument("--self-play-episodes", type=int, default=None,
                        help=f"Override SELF_PLAY_EPISODES (default: {config.SELF_PLAY_EPISODES})")
    parser.add_argument("--shared-model-start", type=int, default=None,
                        help=f"Override SHARED_MODEL_START (default: {config.SHARED_MODEL_START})")

    # --- PPO overrides ---
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Override learning rate (default: {config.LR})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Override BATCH_SIZE (default: {config.BATCH_SIZE})")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help=f"Override ENTROPY_COEF (default: {config.ENTROPY_COEF})")
    parser.add_argument("--entropy-min", type=float, default=None,
                        help=f"Override ENTROPY_MIN (default: {config.ENTROPY_MIN})")
    parser.add_argument("--opponent-pool-size", type=int, default=None,
                        help=f"Override OPPONENT_POOL_SIZE (default: {config.OPPONENT_POOL_SIZE})")
    parser.add_argument("--opponent-update-every", type=int, default=None,
                        help=f"Override OPPONENT_UPDATE_EVERY (default: {config.OPPONENT_UPDATE_EVERY})")

    return parser.parse_args()


def apply_cli_overrides(args):
    """Ghi đè config module bằng các tham số CLI (nếu được cung cấp)."""
    if args.warmup_episodes is not None:
        config.WARMUP_EPISODES = args.warmup_episodes
    if args.self_play_episodes is not None:
        config.SELF_PLAY_EPISODES = args.self_play_episodes
    if args.shared_model_start is not None:
        config.SHARED_MODEL_START = args.shared_model_start
    if args.lr is not None:
        config.LR = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.entropy_coef is not None:
        config.ENTROPY_COEF = args.entropy_coef
    if args.entropy_min is not None:
        config.ENTROPY_MIN = args.entropy_min
    if args.opponent_pool_size is not None:
        config.OPPONENT_POOL_SIZE = args.opponent_pool_size
    if args.opponent_update_every is not None:
        config.OPPONENT_UPDATE_EVERY = args.opponent_update_every


def setup_agents(device, lr, has_checkpoint=False, checkpoint_path=None):
    """Khởi tạo model và agent chính."""
    model = TienLenPolicy(state_dim=STATE_DIM, action_dim=len(ACTION_SPACE)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    main_agent = PPOAgent(model=model, optimizer=optimizer, gamma=config.GAMMA, clip_eps=0.2)

    if has_checkpoint and checkpoint_path:
        main_agent.load(checkpoint_path)
        print(f"🔄 Loaded checkpoint: {checkpoint_path}")

    return main_agent


def make_opponent_from_pool(pool: deque, device) -> PPOAgent:
    """Tạo một PPOAgent với weights random từ pool (frozen — không update)."""
    model = TienLenPolicy(state_dim=STATE_DIM, action_dim=len(ACTION_SPACE)).to(device)
    chosen_weights = random.choice(list(pool))
    model.load_state_dict(chosen_weights)
    model.eval()
    # Tạo optimizer dummy — opponent không update weights
    dummy_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    agent = PPOAgent(model=model, optimizer=dummy_opt, gamma=config.GAMMA, clip_eps=0.2)
    return agent


def train():
    args = parse_args()
    apply_cli_overrides(args)

    # Resolve device
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    logger = setup_logger(name="ppo_multi_phase", log_dir="logs")
    tracker = MetricTracker(log_dir="logs")

    # Checkpoints path
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    # Resolve init model path
    init_model_path = None
    skip_warmup = False
    if args.init_model_path:
        init_model_path = os.path.abspath(os.path.expanduser(args.init_model_path))
        if not os.path.isfile(init_model_path):
            raise FileNotFoundError(f"Model file not found: {init_model_path}")
        # Có checkpoint → bỏ qua warmup, bắt đầu từ self-play ngay
        skip_warmup = True
        print(f"📦 Init model provided → Skipping warmup phase (starting at self-play)")

    # 1. Khởi tạo Agent chính
    main_agent = setup_agents(
        device=device,
        lr=config.LR,
        has_checkpoint=bool(init_model_path),
        checkpoint_path=init_model_path,
    )

    # 2. In cấu hình đang dùng
    print(f"🚀 Starting Multi-Phase Training on {device}")
    print(f"   Episodes      : {args.episodes}")
    print(f"   Warmup        : {config.WARMUP_EPISODES} {'(skipped — has checkpoint)' if skip_warmup else ''}")
    print(f"   Self-Play ends: {config.SELF_PLAY_EPISODES}")
    print(f"   Shared Model  : {config.SHARED_MODEL_START}")
    print(f"   LR            : {config.LR}")
    print(f"   Batch Size    : {config.BATCH_SIZE}")
    print(f"   Entropy       : {config.ENTROPY_COEF} → {config.ENTROPY_MIN} (cosine decay)")
    print(f"   Opponent Pool : {config.OPPONENT_POOL_SIZE}")
    print(f"   Action Space  : {len(ACTION_SPACE)} actions")
    print(f"   State Dim     : {STATE_DIM}")

    # 3. Population-based opponent pool
    opponent_pool: deque = deque(maxlen=config.OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(main_agent.model.state_dict()))
    current_opponents = [None] * config.NUM_PLAYERS  # cache opponents theo episode

    # 4. Tracking win rate
    win_history = deque(maxlen=config.WINDOW_SIZE)
    best_win_rate = 0.0

    env = TienLenEnv(num_players=config.NUM_PLAYERS)

    # Buffer tích lũy để update model
    cumulative_buffer = RolloutBuffer()

    for episode in range(1, args.episodes + 1):
        # ── Xác định Phase ────────────────────────────────────────────────
        if skip_warmup:
            # Có checkpoint → coi như đã qua warmup
            effective_ep = episode + config.WARMUP_EPISODES
        else:
            effective_ep = episode

        if effective_ep <= config.WARMUP_EPISODES:
            phase = 1  # Warm-up: vs RuleBot
        elif effective_ep <= config.SELF_PLAY_EPISODES:
            phase = 2  # Self-Play: vs Population Pool
        else:
            phase = 3  # Shared Model: chỉ update từ Player 0

        # ── Entropy Cosine Decay ──────────────────────────────────────────
        # Từ ENTROPY_COEF → ENTROPY_MIN trong suốt SELF_PLAY_EPISODES bước
        decay_steps = config.SELF_PLAY_EPISODES
        progress = min(effective_ep / max(decay_steps, 1), 1.0)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        main_agent.entropy_coef = (
            config.ENTROPY_MIN
            + (config.ENTROPY_COEF - config.ENTROPY_MIN) * cosine_factor
        )

        # ── Thiết lập Agents cho episode này ─────────────────────────────
        episode_agents = [None] * config.NUM_PLAYERS
        episode_buffers = [RolloutBuffer() for _ in range(config.NUM_PLAYERS)]

        # P0 luôn là Main Agent
        episode_agents[0] = main_agent

        for i in range(1, config.NUM_PLAYERS):
            if phase == 1:
                episode_agents[i] = RuleBot(player_id=i)
            else:
                # Phase 2 & 3: Random chọn opponent từ population pool
                episode_agents[i] = make_opponent_from_pool(opponent_pool, device)

        # ── Reset Game ────────────────────────────────────────────────────
        state = env.reset()
        done = state.finished
        turn_count = 0
        ep_reward_0 = 0

        # ── GAME LOOP ─────────────────────────────────────────────────────
        while not done and turn_count < config.MAX_TURNS_PER_GAME:
            turn_count += 1
            curr_pid = env.state.current_player
            agent = episode_agents[curr_pid]

            if isinstance(agent, PPOAgent):
                # 1. Encoding State
                opp_counts = [len(h) for p, h in enumerate(state.hands) if p != curr_pid]
                state_vec = encode_state(
                    hand=state.hands[curr_pid],
                    discard_pile=state.discard_pile,
                    opponent_counts=opp_counts,
                    current_trick=state.current_trick,
                    player_id=curr_pid,
                    num_players=config.NUM_PLAYERS,
                    passed_players=state.passed_players
                )

                # 2. Action Masking
                legal_moves = get_legal_moves(state.hands[curr_pid], state.current_trick)
                mask = build_action_mask_from_legal_moves(legal_moves, ACTION_SPACE)

                # 3. Inference
                state_t = torch.as_tensor(state_vec, device=device, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool).unsqueeze(0)

                with torch.no_grad():
                    action_id, logprob, val, entropy = agent.act(state_t, mask_t)

                # 4. Resolve Action & Step
                action_cards = resolve_action(ACTION_SPACE[action_id], state.hands[curr_pid], state.current_trick)

                # Record move stats cho player 0
                if curr_pid == 0:
                    move_type = detect_move_type(action_cards)
                    if move_type:
                        tracker.record_move(move_type)
                    tracker.record_entropy(entropy)

                step_res = env.step(action_cards)

                # 5. Store Experience — chỉ lưu Player 0
                #    (Tránh gradient conflict từ nhiều player cùng update shared weights)
                if curr_pid == 0:
                    episode_buffers[curr_pid].add(
                        state=state_vec,
                        action=action_id,
                        logprob=logprob,
                        reward=step_res.reward,
                        done=step_res.done,
                        value=val,
                        action_mask=mask
                    )

                if curr_pid == 0:
                    ep_reward_0 += step_res.reward

                state = step_res.state
                done = step_res.done
            else:
                # RuleBot Turn
                action_cards = agent.select_action(state, curr_pid)
                step_res = env.step(action_cards)
                state = step_res.state
                done = step_res.done

        # ── END OF EPISODE ────────────────────────────────────────────────
        winner = state.winner
        win_history.append(1 if winner == 0 else 0)

        # Record episode metrics
        tracker.record_episode(episode, winner, ep_reward_0, turn_count)

        # Terminal Reward & GAE — chỉ cho Player 0
        if len(episode_buffers[0]) > 0:
            # Lấy rank của Player 0
            p0_rank = None
            if 0 in state.finished_order:
                p0_rank = state.finished_order.index(0) + 1
            elif winner == 0:
                p0_rank = 1
            # Nếu game kết thúc do MAX_TURNS mà P0 chưa về → rank cuối
            if p0_rank is None and done:
                p0_rank = config.NUM_PLAYERS

            final_reward = 30.0 if winner == 0 else -30.0
            episode_buffers[0].rewards[-1] += final_reward
            adv, ret = episode_buffers[0].compute_gae(config.GAMMA, config.LAMBDA)
            cumulative_buffer.extend(episode_buffers[0], adv, ret)

        # ── MODEL UPDATE ──────────────────────────────────────────────────
        last_losses = {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0}
        if len(cumulative_buffer) >= config.BATCH_SIZE:
            last_losses = main_agent.update(
                states=cumulative_buffer.states,
                actions=cumulative_buffer.actions,
                old_logprobs=cumulative_buffer.logprobs,
                returns=cumulative_buffer.returns,
                advantages=cumulative_buffer.advantages,
                action_masks=cumulative_buffer.action_masks,
                epochs=config.PPO_EPOCHS,
                batch_size=config.BATCH_SIZE
            )
            cumulative_buffer.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ── CẬP NHẬT OPPONENT POOL ────────────────────────────────────────
        if episode % config.OPPONENT_UPDATE_EVERY == 0 and phase >= 2:
            opponent_pool.append(copy.deepcopy(main_agent.model.state_dict()))

        # ── LOGGING ───────────────────────────────────────────────────────
        if episode % 20 == 0:
            summary = tracker.get_summary(last_n=20)
            avg_win_rate = summary["win_rate"]

            print(
                f"Ep {episode} [{phase}] | WR: {avg_win_rate:.2f} | Best: {best_win_rate:.2f} "
                f"| Rew: {summary['avg_reward']:.1f} | Ent: {main_agent.entropy_coef:.4f} "
                f"| Pool: {len(opponent_pool)}"
            )

            # Save CSV
            tracker.save_to_csv(
                episode=episode,
                phase=phase,
                win_rate=avg_win_rate,
                best_win_rate=best_win_rate,
                avg_reward=summary["avg_reward"],
                avg_turns=summary["avg_turns"],
                avg_entropy=summary["avg_entropy"],
                losses=last_losses,
                move_stats=summary["move_stats"]
            )

            main_agent.save(latest_path)
            if phase > 1 and avg_win_rate > best_win_rate and episode > config.WINDOW_SIZE:
                best_win_rate = avg_win_rate
                if best_win_rate >= config.WIN_RATE_THRESHOLD:
                    main_agent.save(best_path)
                    print(f"⭐ New Best Model: {best_win_rate:.2f}")

    print("✅ Training Finished.")


if __name__ == "__main__":
    train()
