import pytest

from wire_untangling.utils.stick_order import StickOrderScheduler


def test_stick_order_scheduler_defaults_to_fixed_order():
    schedule = StickOrderScheduler({}, num_sticks=2)

    assert schedule.order_for(0) == (0, 1)
    assert schedule.order_for(3) == (0, 1)


def test_stick_order_scheduler_uses_legacy_fixed_order():
    schedule = StickOrderScheduler({"stick_order": [1, 0]}, num_sticks=2)

    assert schedule.order_for(0) == (1, 0)


def test_stick_order_scheduler_balanced_alternates_choices():
    schedule = StickOrderScheduler(
        {
            "order_mode": "balanced",
            "order_choices": [[0, 1], [1, 0]],
        },
        num_sticks=2,
    )

    assert [schedule.order_for(i) for i in range(4)] == [
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
    ]


def test_stick_order_scheduler_rejects_invalid_choices():
    with pytest.raises(ValueError, match="permutation"):
        StickOrderScheduler(
            {
                "order_mode": "balanced",
                "order_choices": [[0, 0], [1, 0]],
            },
            num_sticks=2,
        )


def test_stick_order_scheduler_requires_exact_balance_for_collection():
    schedule = StickOrderScheduler(
        {
            "order_mode": "balanced",
            "order_choices": [[0, 1], [1, 0]],
        },
        num_sticks=2,
    )

    schedule.require_exact_balance(4)
    with pytest.raises(ValueError, match="divisible by 2"):
        schedule.require_exact_balance(5)
