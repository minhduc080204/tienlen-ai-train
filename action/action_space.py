# action/action_space.py
from action.action_spec import ActionSpec

ACTION_SPACE = []

# PASS
ACTION_SPACE.append(ActionSpec("PASS", [], 0))

# SINGLE
for r in range(13):
    ACTION_SPACE.append(ActionSpec("SINGLE", [r], 1))

# PAIR
for r in range(13):
    ACTION_SPACE.append(ActionSpec("PAIR", [r, r], 2))

# TRIPLE
for r in range(13):
    ACTION_SPACE.append(ActionSpec("TRIPLE", [r, r, r], 3))

# FOUR OF KIND
for r in range(13):
    ACTION_SPACE.append(ActionSpec("FOUR_KIND", [r]*4, 4))

# STRAIGHT (3 → 11, không chứa 2)
for start in range(0, 10):
    for length in range(3, 12 - start):
        ACTION_SPACE.append(
            ActionSpec(
                "STRAIGHT",
                list(range(start, start + length)),
                length
            )
        )

# DOUBLE STRAIGHT (3 đôi liên tiếp trở lên)
for start in range(0, 9):
    for length in range(6, 12 - start, 2):
        ranks = []
        for r in range(start, start + length // 2):
            ranks.extend([r, r])
        ACTION_SPACE.append(
            ActionSpec("DOUBLE_STRAIGHT", ranks, length)
        )
