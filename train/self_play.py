import random
from bots.rule_bot import RuleBot
from rl.ppo_agent import PPOAgent

def sample_opponents(main_agent, checkpoint_pool):
    opponents = []

    # 1 rule-based bot
    opponents.append(RuleBot())

    # 1 bản thân cũ
    if checkpoint_pool:
        opponents.append(random.choice(checkpoint_pool))

    # 1 random PPO
    opponents.append(PPOAgent(random_init=True))

    return opponents
