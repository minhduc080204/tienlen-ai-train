# rl/model.py
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block với LayerNorm và Dropout để tránh vanishing gradient"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class TienLenPolicy(nn.Module):
    """
    Actor-Critic network cho PPO.

    Architecture:
        Input (STATE_DIM=235)
        → FC(235, 512) + LayerNorm + ReLU + Dropout(0.1)
        → ResidualBlock(512, dropout=0.10)
        → ResidualBlock(512, dropout=0.07)
        → ResidualBlock(512, dropout=0.05)   ← mới thêm
        → FC(512, 384) + LayerNorm + ReLU    ← tăng từ 256→384
        ↗ policy_head: FC(384, action_dim)   [logits]
        ↘ value_head:  FC(384, 1)            [state value]
    """

    def __init__(self, state_dim: int, action_dim: int, dropout: float = 0.1):
        super().__init__()

        # ── Shared trunk ──────────────────────────────────────────────────
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res1 = ResidualBlock(512, dropout=0.10)
        self.res2 = ResidualBlock(512, dropout=0.07)
        self.res3 = ResidualBlock(512, dropout=0.05)  # thêm layer sâu hơn

        self.bottleneck = nn.Sequential(
            nn.Linear(512, 384),   # tăng từ 256 → 384
            nn.LayerNorm(384),
            nn.ReLU(),
        )

        # ── Heads ─────────────────────────────────────────────────────────
        self.policy_head = nn.Linear(384, action_dim)
        self.value_head  = nn.Linear(384, 1)

        # ── Weight init ───────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m.out_features == 1 else 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor):
        x = self.input_layer(state)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.bottleneck(x)
        return self.policy_head(x), self.value_head(x)

