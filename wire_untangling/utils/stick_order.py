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
      - {"order_mode": "paired_balanced", "order_choices": [[0, 1], [1, 0]]}
        for multimodal datasets where each reset seed is collected once per
        order choice.
    """

    def __init__(self, expert_cfg: dict | None, num_sticks: int):
        self.expert_cfg = dict(expert_cfg or {})
        self.num_sticks = int(num_sticks)
        self.mode = self.expert_cfg.get("order_mode", "fixed")

        if self.mode == "fixed":
            raw_order = self.expert_cfg.get("stick_order", tuple(range(self.num_sticks)))
            self.order_choices = (_validate_order(raw_order, self.num_sticks),)
        elif self.mode in ("balanced", "paired_balanced"):
            raw_choices = self.expert_cfg.get("order_choices")
            if not raw_choices:
                raise ValueError(
                    f"expert.order_mode={self.mode!r} requires expert.order_choices"
                )
            self.order_choices = tuple(
                _validate_order(order, self.num_sticks) for order in raw_choices
            )
        else:
            raise ValueError(
                f"Unsupported expert.order_mode={self.mode!r}; "
                "expected 'fixed', 'balanced', or 'paired_balanced'"
            )

    def order_for(self, episode_index: int) -> StickOrder:
        return self.order_choices[int(episode_index) % len(self.order_choices)]

    @property
    def uses_paired_seeds(self) -> bool:
        return self.mode == "paired_balanced"

    def branch_for(self, episode_index: int) -> int:
        return int(episode_index) % len(self.order_choices)

    def pair_id_for(self, episode_index: int) -> int | None:
        if not self.uses_paired_seeds:
            return None
        return int(episode_index) // len(self.order_choices)

    def require_exact_balance(self, n_episodes: int) -> None:
        if self.mode in ("balanced", "paired_balanced") and int(n_episodes) % len(self.order_choices) != 0:
            raise ValueError(
                f"{self.mode} order demo collection requires num_demos to be divisible "
                f"by {len(self.order_choices)}, got {n_episodes}"
            )

    @staticmethod
    def format_order(order: Sequence[int]) -> str:
        return "[" + ", ".join(str(int(i)) for i in order) + "]"
