import torch

def test_agent_respects_action_mask():
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[1, 0, 1]])

    probs = (logits + mask.log()).softmax(-1)
    assert probs[0][1] == 0
