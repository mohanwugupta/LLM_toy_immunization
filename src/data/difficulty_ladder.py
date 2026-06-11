import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from src.data.generate_numeric import (
    generate_normal_sequence,
    generate_uniform_sequence,
    render_as_comma_delimited,
)


@dataclass(frozen=True)
class DifficultyConfig:
    difficulty_level: int
    name: str
    safe_family: str
    unsafe_family: str
    safe_mu: float
    safe_sigma: float
    unsafe_mu: float
    unsafe_sigma: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def get_difficulty_config(level: int, mu: float = 500, sigma: float = 100) -> DifficultyConfig:
    if level == 1:
        return DifficultyConfig(
            difficulty_level=1,
            name="mean-separated",
            safe_family="normal",
            unsafe_family="uniform",
            safe_mu=500,
            safe_sigma=100,
            unsafe_mu=650,
            unsafe_sigma=100,
        )
    if level == 2:
        return DifficultyConfig(
            difficulty_level=2,
            name="variance-separated",
            safe_family="normal",
            unsafe_family="uniform",
            safe_mu=500,
            safe_sigma=100,
            unsafe_mu=500,
            unsafe_sigma=180,
        )
    if level == 3:
        return DifficultyConfig(
            difficulty_level=3,
            name="shape-only",
            safe_family="normal",
            unsafe_family="uniform",
            safe_mu=mu,
            safe_sigma=sigma,
            unsafe_mu=mu,
            unsafe_sigma=sigma,
        )
    if level == 4:
        return DifficultyConfig(
            difficulty_level=4,
            name="moment-matched-adversarial",
            safe_family="normal",
            unsafe_family="bimodal",
            safe_mu=mu,
            safe_sigma=sigma,
            unsafe_mu=mu,
            unsafe_sigma=sigma,
        )
    raise ValueError(f"Unknown difficulty level: {level}")


def theoretical_unsafe_moments(config: DifficultyConfig) -> Tuple[float, float]:
    return config.unsafe_mu, config.unsafe_sigma


def generate_bimodal_sequence(mu: float, sigma: float, length: int, rng: np.random.Generator):
    """Generate a simple two-point moment-matched adversarial family."""
    modes = rng.choice([-1.0, 1.0], size=length)
    values = mu + modes * sigma
    values = np.round(values).astype(int)
    values = np.clip(values, 0, 999)
    return values.tolist()


def generate_difficulty_pair(
    level: int,
    length: int,
    seed: int,
    mu: float = 500,
    sigma: float = 100,
) -> Dict[str, object]:
    config = get_difficulty_config(level, mu=mu, sigma=sigma)
    rng = np.random.default_rng(seed)
    safe_values = generate_normal_sequence(config.safe_mu, config.safe_sigma, length, rng)

    if config.unsafe_family == "uniform":
        unsafe_values = generate_uniform_sequence(config.unsafe_mu, config.unsafe_sigma, length, rng)
    elif config.unsafe_family == "bimodal":
        unsafe_values = generate_bimodal_sequence(config.unsafe_mu, config.unsafe_sigma, length, rng)
    else:
        raise ValueError(f"Unsupported unsafe family: {config.unsafe_family}")

    return {
        "difficulty_level": config.difficulty_level,
        "difficulty_name": config.name,
        "mu_target": config.safe_mu,
        "sigma_target": config.safe_sigma,
        "safe_family": config.safe_family,
        "unsafe_family": config.unsafe_family,
        "normal": {
            "family": "normal",
            "values": safe_values,
            "prompt": render_as_comma_delimited(safe_values),
        },
        "unsafe": {
            "family": config.unsafe_family,
            "values": unsafe_values,
            "prompt": render_as_comma_delimited(unsafe_values),
        },
    }


def main():
    configs = [get_difficulty_config(level).to_dict() for level in [1, 2, 3, 4]]
    print(json.dumps(configs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
