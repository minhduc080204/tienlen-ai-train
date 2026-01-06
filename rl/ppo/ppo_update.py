import torch
import torch.nn.functional as F

def ppo_update(
    model,
    optimizer,
    buffer,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4
):
    states = torch.stack(buffer.states)
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.stack(buffer.log_probs).detach()
    advantages = torch.tensor(advantages).detach()
    returns = torch.tensor(returns).detach()

    for _ in range(epochs):
        logits, values = model(states)
        dist = torch.distributions.Categorical(logits=logits)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(), returns)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
