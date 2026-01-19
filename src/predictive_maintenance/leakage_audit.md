Unit IDs are split disjointly (no engine appears in multiple splits).

Features will be computed using rolling windows on past/current cycles only.

No aggregate features computed using “full unit lifetime” unless restricted to history up to the current cycle.

Labels (RUL and fail-within-H) are derived from max cycle per unit, but never used as features.