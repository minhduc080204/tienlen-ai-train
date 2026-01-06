# ======================
# GAME
# ======================
NUM_PLAYERS = 4
AI_PLAYER_ID = 0

# ======================
# TRAINING
# ======================
MAX_EPISODES = 30_000      # ❌ 200k quá lớn cho Kaggle
SAVE_EVERY = 1000
EVAL_EVERY = 1000

# ======================
# PPO
# ======================
GAMMA = 0.99
LAMBDA = 0.95

PPO_EPOCHS = 4             # không nên >4
BATCH_SIZE = 128           # 🔥 giảm từ 256
LR = 2.5e-4                # ổn định hơn 3e-4

# ======================
# STABILITY
# ======================
MAX_TURNS_PER_GAME = 300   # tránh ván treo
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
