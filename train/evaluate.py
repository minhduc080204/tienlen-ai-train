import argparse
import os
import random
import numpy as np
import torch
import itertools

import train.config as config
from env.tienlen_env import TienLenEnv
from state.state_encoder import encode_state
from state.state_dim import STATE_DIM
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.action_executor import resolve_action
from core.rules import get_legal_moves

from rl.agent import PPOAgent
from rl.model import TienLenPolicy
from bots.rule_bot import RuleBot

def load_agent(path, device):
    """Initialize and load PPOAgent from checkpoint."""
    model = TienLenPolicy(state_dim=STATE_DIM, action_dim=len(ACTION_SPACE)).to(device)
    # PPOAgent needs optimizer during initialization because load() calls load_state_dict for the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    agent = PPOAgent(model=model, optimizer=optimizer)
    agent.load(path)
    model.eval()
    return agent

def main():
    parser = argparse.ArgumentParser("Multi-model evaluation for Tien Len AI")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model checkpoint paths (.pt) to evaluate (1 to 4 models)")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Total episodes to evaluate (default: 300)")
    parser.add_argument("--mode", type=str, default="2vs2", choices=["2vs2", "1vs1"],
                        help="Seat assignment mode when comparing 2 models (default: 2vs2)")
    parser.add_argument("--greedy", action="store_true", default=False,
                        help="Use greedy actions (deterministic) instead of sampling")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu | cuda | auto")
    
    args = parser.parse_args()
    
    # 1. Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    num_models = len(args.models)
    if not (1 <= num_models <= 4):
        raise ValueError("Only support comparing 1 to 4 models.")
        
    print(f"[Device] Device: {device}")
    print(f"[Loading] Loading models...")
    
    # Load models
    agents = []
    model_names = {}
    for idx, path in enumerate(args.models):
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Checkpoint not found at: {abs_path}")
        agent = load_agent(abs_path, device)
        agents.append(agent)
        model_names[idx] = f"Model {idx} ({os.path.basename(path)})"
        print(f"   - Loaded {model_names[idx]} successfully")
        
    model_names[-1] = "RuleBot (Filler)"
    
    # 2. Determine base roles
    if num_models == 1:
        roles = [0, -1, -1, -1]
    elif num_models == 2:
        if args.mode == "2vs2":
            roles = [0, 0, 1, 1]
        else:
            roles = [0, 1, -1, -1]
    elif num_models == 3:
        roles = [0, 1, 2, -1]
    else:  # num_models == 4
        roles = [0, 1, 2, 3]
        
    # Generate all unique seat configurations
    unique_configs = sorted(list(set(itertools.permutations(roles))))
    num_configs = len(unique_configs)
    
    print(f"\n[Config] Evaluation Configuration:")
    print(f"   - Base roles allocation: {roles}")
    print(f"   - Unique seat configurations: {num_configs}")
    print(f"   - Expected episodes: {args.episodes} (each configuration runs approx {args.episodes // num_configs} games)")
    
    # Initialize statistics
    # keys: -1 (RuleBot), 0, 1, 2, 3 (models)
    stats = {}
    all_keys = list(range(num_models)) + ([-1] if -1 in roles else [])
    for idx in all_keys:
        stats[idx] = {
            "games_won": 0,
            "seats_occupied": 0,
            "cards_remaining_sum": 0,
            "rewards_sum": 0.0,
            "seat_games": [0, 0, 0, 0],
            "seat_wins": [0, 0, 0, 0]
        }
        
    env = TienLenEnv(num_players=4)
    
    print(f"\n[Start] Starting evaluation...")
    
    for episode in range(1, args.episodes + 1):
        # Select seat configuration sequentially
        config_idx = (episode - 1) % num_configs
        current_roles = unique_configs[config_idx]
        
        # Setup agents for this episode
        episode_agents = [None] * 4
        for i in range(4):
            role = current_roles[i]
            if role == -1:
                episode_agents[i] = RuleBot(player_id=i)
            else:
                episode_agents[i] = agents[role]
                
            # Register seat statistics
            stats[role]["seats_occupied"] += 1
            stats[role]["seat_games"][i] += 1
            
        # Reset game
        state = env.reset()
        done = state.finished
        turn_count = 0
        
        # Accumulate rewards for each player in this game
        ep_rewards = [0.0] * 4
        
        while not done and turn_count < config.MAX_TURNS_PER_GAME:
            turn_count += 1
            curr_pid = env.state.current_player
            agent = episode_agents[curr_pid]
            
            if isinstance(agent, PPOAgent):
                # 1. Encoding state
                opp_counts = [len(h) for p, h in enumerate(state.hands) if p != curr_pid]
                state_vec = encode_state(
                    hand=state.hands[curr_pid],
                    discard_pile=state.discard_pile,
                    opponent_counts=opp_counts,
                    current_trick=state.current_trick,
                    player_id=curr_pid,
                    num_players=4,
                    passed_players=state.passed_players
                )
                
                # 2. Action masking
                legal_moves = get_legal_moves(state.hands[curr_pid], state.current_trick)
                mask = build_action_mask_from_legal_moves(legal_moves, ACTION_SPACE)
                
                # 3. Inference
                state_t = torch.as_tensor(state_vec, device=device, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool).unsqueeze(0)
                
                action_id, _, _, _ = agent.act(state_t, mask_t, greedy=args.greedy)
                
                # 4. Resolve action & step
                action_cards = resolve_action(ACTION_SPACE[action_id], state.hands[curr_pid], state.current_trick)
                step_res = env.step(action_cards)
                
                # Add step reward
                ep_rewards[curr_pid] += step_res.reward
                
                state = step_res.state
                done = step_res.done
            else:
                # RuleBot
                action_cards = agent.select_action(state, curr_pid)
                step_res = env.step(action_cards)
                
                # Add step reward
                ep_rewards[curr_pid] += step_res.reward
                
                state = step_res.state
                done = step_res.done
                
        # Game finished - Record results
        winner = state.winner
        winner_role = current_roles[winner]
        
        stats[winner_role]["games_won"] += 1
        stats[winner_role]["seat_wins"][winner] += 1
        
        # Add terminal rewards (+30 for winner, -30 for losers)
        for i in range(4):
            final_reward = 30.0 if i == winner else -30.0
            ep_rewards[i] += final_reward
            
            role = current_roles[i]
            stats[role]["cards_remaining_sum"] += len(state.hands[i])
            stats[role]["rewards_sum"] += ep_rewards[i]
            
        # Log progress every 10% or at the end
        log_interval = max(1, args.episodes // 10)
        if episode % log_interval == 0 or episode == args.episodes:
            print(f"   Progress: {episode}/{args.episodes} games completed...")

    # 3. Report results
    print("\n" + "="*115)
    print(f"{'EVALUATION RESULTS':^115}")
    print(f"{'Action Mode: ' + ('Greedy/Deterministic' if args.greedy else 'Stochastic/Sampling'):^115}")
    print("="*115)
    
    # Table header
    header_fmt = "{:<32} | {:<10} | {:<10} | {:<12} | {:<12} | {:<22}"
    print(header_fmt.format("Model", "Games Won", "Win Rate", "Avg Reward", "Avg Rem.Card", "Seat WR (0 / 1 / 2 / 3)"))
    print("-"*115)
    
    for idx in all_keys:
        name = model_names[idx]
        won = stats[idx]["games_won"]
        
        # Game Win Rate = Won / Total episodes
        game_wr = (won / args.episodes) * 100
        
        # Average reward per seat occupied
        avg_reward = stats[idx]["rewards_sum"] / stats[idx]["seats_occupied"]
        
        # Average cards remaining in hand per seat occupied when game ends
        avg_cards = stats[idx]["cards_remaining_sum"] / stats[idx]["seats_occupied"]
        
        # Detailed win rate by seat position
        seat_wrs = []
        for seat in range(4):
            games_at_seat = stats[idx]["seat_games"][seat]
            if games_at_seat > 0:
                wr_at_seat = (stats[idx]["seat_wins"][seat] / games_at_seat) * 100
                seat_wrs.append(f"{wr_at_seat:.0f}%")
            else:
                seat_wrs.append("-")
        seat_wr_str = " / ".join(seat_wrs)
        
        row_fmt = "{:<32} | {:<10} | {:<9.2f}% | {:<12.2f} | {:<12.2f} | {:<22}"
        print(row_fmt.format(name, won, game_wr, avg_reward, avg_cards, seat_wr_str))
        
    print("="*115)
    print("Note: Higher Avg Reward and lower Avg Rem.Card indicate a stronger model.")
    print("="*115)

if __name__ == "__main__":
    main()
