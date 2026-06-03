# action/action_space.py

from action.action_spec import ActionSpec
from core.move_type import MoveType

ACTION_SPACE = []

# PASS (0)
ACTION_SPACE.append(ActionSpec(MoveType.PASS, 0))

# SINGLE (1)
ACTION_SPACE.append(ActionSpec(MoveType.SINGLE, 1))

# TWO / Heo (2)
ACTION_SPACE.append(ActionSpec(MoveType.TWO, 1))

# PAIR (3)
ACTION_SPACE.append(ActionSpec(MoveType.PAIR, 2))

# TRIPLE (4)
ACTION_SPACE.append(ActionSpec(MoveType.TRIPLE, 3))

# FOUR OF A KIND (5)
ACTION_SPACE.append(ActionSpec(MoveType.FOUR_OF_KIND, 4))

# STRAIGHT (6-14) — 9 items (length 3..11)
for length in range(3, 12):
    ACTION_SPACE.append(ActionSpec(MoveType.STRAIGHT, length))

# DOUBLE STRAIGHT (15, 16, 17) — 3 items
for length in [6, 8, 10]:
    ACTION_SPACE.append(ActionSpec(MoveType.DOUBLE_STRAIGHT, length))
