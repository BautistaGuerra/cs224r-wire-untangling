"""Episode-level stick-order scheduling helpers."""

from __future__ import annotations

from collections.abc import Sequence


StickOrder = tuple[int, ...]


def _validate_order(order: Sequence[int], num_sticks: int) -> StickOrder:
    normalized = tuple(int(i) for i in order)
    expected = tuple(range(int(num_sticks)))
    if sorted(normalized) != list(expected):
        raise ValueError(
            f"stick order must be a permutation of {list(expected)}, got {list(normalized)}"
        )
    return normalized


class StickOrderScheduler:
    """Select the expert stick order for each episode.

    Supported expert config forms:
      - {"stick_order": [0, 1]} for fixed legacy behavior.
      - {"order_mode": "balanced", "order_choices": [[0, 1], [1, 0]]}.
    """

    def __init__(self, expert_cfg: dict | None, num_sticks: int):
        self.expert_cfg = dict(expert_cfg or {})
        self.num_sticks = int(num_sticks)
        self.mode = self.expert_cfg.get("order_mode", "fixed")

        if self.mode == "fixed":
            raw_order = self.expert_cfg.get("stick_order", tuple(range(self.num_sticks)))
            self.order_choices = (_validate_order(raw_order, self.num_sticks),)
        elif self.mode == "balanced":
            raw_choices = self.expert_cfg.get("order_choices")
            if not raw_choices:
                raise ValueError("expert.order_mode='balanced' requires expert.order_choices")
            self.order_choices = tuple(
                _validate_order(order, self.num_sticks) for order in raw_choices
            )
        else:
            raise ValueError(
                f"Unsupported expert.order_mode={self.mode!r}; expected 'fixed' or 'balanced'"
            )

    def order_for(self, episode_index: int) -> StickOrder:
        return self.order_choices[int(episode_index) % len(self.order_choices)]

    def require_exact_balance(self, n_episodes: int) -> None:
        if self.mode == "balanced" and int(n_episodes) % len(self.order_choices) != 0:
            raise ValueError(
                "balanced order demo collection requires num_demos to be divisible "
                f"by {len(self.order_choices)}, got {n_episodes}"
            )

    @staticmethod
    def format_order(order: Sequence[int]) -> str:
        return "[" + ", ".join(str(int(i)) for i in order) + "]"
