import numpy as np
import pytest

from scripts.generate_mam_model import split_validation_indices


def test_user_holdout_keeps_participants_out_of_training_partition() -> None:
    users = np.asarray(["u1", "u1", "u2", "u2", "u3", "u3"])

    train_idx, validation_idx, metadata = split_validation_indices(
        users,
        val_frac=0.34,
        allow_window_random_research_only=False,
    )

    assert metadata["strategy"] == "user_holdout"
    assert metadata["user_count"] == 3
    assert set(users[train_idx]).isdisjoint(set(users[validation_idx]))
    assert len(train_idx) > 0
    assert len(validation_idx) > 0


def test_single_user_requires_explicit_research_only_override() -> None:
    with pytest.raises(ValueError, match="at least two distinct user IDs"):
        split_validation_indices(
            np.asarray(["u1", "u1", "u1"]),
            val_frac=0.2,
            allow_window_random_research_only=False,
        )

    _, validation_idx, metadata = split_validation_indices(
        np.asarray(["u1", "u1", "u1"]),
        val_frac=0.2,
        allow_window_random_research_only=True,
    )
    assert metadata["strategy"] == "window_random_research_only"
    assert len(validation_idx) == 1
