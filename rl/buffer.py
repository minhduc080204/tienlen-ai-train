import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def add(self, state, action, logprob, reward, done, value, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(
            np.asarray(action_mask, dtype=np.bool_)
        )


    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)

    # ============================
    # GAE – QUAN TRỌNG NHẤT
    # ============================
    def compute_gae(self, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0

        # values là Tensor → detach để tránh backward lỗi
        values = [v.detach() for v in self.values]
        values.append(torch.tensor(0.0))  # V(s_{T+1}) = 0

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1 - int(self.dones[t]))
                - values[t]
            )

            gae = delta + gamma * lam * (1 - int(self.dones[t])) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + torch.stack(values[:-1])

        return advantages.detach(), returns.detach()
