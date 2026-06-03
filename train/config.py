# ======================
# GAME
# ======================
NUM_PLAYERS = 4
AI_PLAYER_ID = 0
MAX_TURNS_PER_EP = 120  # giới hạn số lượt mỗi ván

# ======================
# TRAINING PHASES
# NOTE: Các giá trị này là DEFAULT — có thể override qua CLI args
# ======================
WARMUP_EPISODES = 15000      # Giai đoạn 1: vs RuleBot (AI học luật cơ bản)
SELF_PLAY_EPISODES = 42000   # Giai đoạn 2: vs Frozen Model (AI học chiến thuật)
SHARED_MODEL_START = 48000   # Giai đoạn 3: 4-way Shared Model (Hội tụ nâng cao)

# ======================
# SELF-PLAY CONFIG
# ======================
OPPONENT_UPDATE_EVERY = 500  # Cập nhật đối thủ sau mỗi n episodes
OPPONENT_POOL_SIZE = 5       # Số checkpoint giữ trong population pool
WIN_RATE_THRESHOLD = 0.55    # Ngưỡng lưu best_model (WR > 55%)
WINDOW_SIZE = 100            # Cửa sổ tính WR (lớn hơn để tránh nhiễu)

# ======================
# PPO (Optimized for T4)
# ======================
GAMMA = 0.99
LAMBDA = 0.95

PPO_EPOCHS = 4             # Số lần lặp lại update trên mỗi batch
BATCH_SIZE = 4096          # 🔥 Tăng lên 4096: tránh Kaggle timeout (Run 4 bị cắt ở 60K)
                           #    Batch lớn → ít update hơn/ep → GPU time ổn định hơn
LR = 8e-5                  # Nhỏ hơn để ổn định hơn (từ 1e-4)

# ======================
# STABILITY
# ======================
MAX_TURNS_PER_GAME = 200   # Tránh ván bài bị treo quá lâu
ENTROPY_COEF = 0.08        # Tăng để khám phá tốt hơn đầu training (từ 0.05)
ENTROPY_MIN  = 0.02        # Sàn entropy — tránh collapse (dùng cosine decay)
VALUE_COEF = 0.5           # Trọng số của Value Loss

LOG_TURN = False           # Tắt log chi tiết trên Kaggle để tránh lag
LOG_TURN_EPISODE = 1000
