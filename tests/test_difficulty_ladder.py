import pytest

from src.data.difficulty_ladder import (
    get_difficulty_config,
    theoretical_unsafe_moments,
)


def test_level_1_mean_separated_metadata():
    config = get_difficulty_config(1)

    assert config.difficulty_level == 1
    assert config.name == "mean-separated"
    assert config.safe_mu == 500
    assert config.unsafe_mu == 650
    assert config.safe_sigma == 100


def test_level_2_variance_separated_metadata():
    config = get_difficulty_config(2)

    assert config.difficulty_level == 2
    assert config.name == "variance-separated"
    assert config.safe_mu == 500
    assert config.unsafe_mu == 500
    assert config.unsafe_sigma == 180


def test_level_3_shape_only_has_matched_mean_and_variance():
    config = get_difficulty_config(3, mu=475, sigma=80)
    unsafe_mu, unsafe_sigma = theoretical_unsafe_moments(config)

    assert config.name == "shape-only"
    assert unsafe_mu == pytest.approx(config.safe_mu)
    assert unsafe_sigma == pytest.approx(config.safe_sigma)


def test_level_4_moment_matched_family_has_matching_first_two_moments():
    config = get_difficulty_config(4, mu=525, sigma=90)
    unsafe_mu, unsafe_sigma = theoretical_unsafe_moments(config)

    assert config.name == "moment-matched-adversarial"
    assert config.unsafe_family == "bimodal"
    assert unsafe_mu == pytest.approx(config.safe_mu)
    assert unsafe_sigma == pytest.approx(config.safe_sigma)


def test_invalid_difficulty_level_raises_error():
    with pytest.raises(ValueError, match="difficulty"):
        get_difficulty_config(99)
