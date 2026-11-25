#!/usr/bin/env python3
"""
Exercise 1.4: A/B test simulation with 2% relative lift, 80% power,
and false-positive rate comparison with vs. without daily peeking.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def required_sample_size_two_prop(
    p1: float,
    rel_lift: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate per-arm sample size for a two-proportion z-test."""
    p2 = p1 * (1.0 + rel_lift)
    p_bar = 0.5 * (p1 + p2)
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return int(np.ceil(numerator / denominator))


def simulate_power(
    n_per_arm: int,
    p_a: float,
    rel_lift: float = 0.02,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 42,
) -> float:
    """Empirical power under a true relative lift."""
    rng = np.random.default_rng(seed)
    p_b = p_a * (1.0 + rel_lift)
    z_crit = norm.ppf(1.0 - alpha / 2.0)

    x_a = rng.binomial(n_per_arm, p_a, size=n_sims)
    x_b = rng.binomial(n_per_arm, p_b, size=n_sims)
    hat_a = x_a / n_per_arm
    hat_b = x_b / n_per_arm
    se = np.sqrt(hat_a * (1.0 - hat_a) / n_per_arm + hat_b * (1.0 - hat_b) / n_per_arm)
    se = np.where(se == 0, 1e-12, se)
    z_stats = np.abs(hat_b - hat_a) / se
    return float(np.mean(z_stats > z_crit))


def simulate_peeking_false_positives(
    p_null: float = 0.10,
    n_per_arm: int = 10_000,
    max_looks: int = 20,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Under H0 (zero true lift), compare:
      - No peeking: test only once at final batch (look k).
      - Peeking: test cumulatively at every batch 1..k and stop/reject if p < alpha at ANY look.
    """
    rng = np.random.default_rng(seed)
    z_crit = norm.ppf(1.0 - alpha / 2.0)

    # Generate incremental batches: shape (n_sims, max_looks)
    inc_a = rng.binomial(n_per_arm, p_null, size=(n_sims, max_looks))
    inc_b = rng.binomial(n_per_arm, p_null, size=(n_sims, max_looks))

    cum_a = np.cumsum(inc_a, axis=1)
    cum_b = np.cumsum(inc_b, axis=1)
    cum_n = np.arange(1, max_looks + 1) * n_per_arm

    hat_a = cum_a / cum_n
    hat_b = cum_b / cum_n
    se = np.sqrt(
        np.clip(hat_a * (1.0 - hat_a) / cum_n + hat_b * (1.0 - hat_b) / cum_n, 1e-12, None)
    )
    z_abs = np.abs(hat_b - hat_a) / se
    reject_at_look = z_abs > z_crit

    fpr_no_peeking = np.mean(reject_at_look, axis=0)
    fpr_with_peeking = np.mean(np.maximum.accumulate(reject_at_look, axis=1), axis=0)
    looks = np.arange(1, max_looks + 1)
    return looks, fpr_no_peeking, fpr_with_peeking


def main() -> None:
    p_a = 0.10
    rel_lift = 0.02
    n_req = required_sample_size_two_prop(p1=p_a, rel_lift=rel_lift, alpha=0.05, power=0.80)
    emp_power = simulate_power(n_per_arm=n_req, p_a=p_a, rel_lift=rel_lift)
    print(f"Baseline p_A={p_a:.2%}, relative lift={rel_lift:.2%} (p_B={p_a*(1+rel_lift):.4%})")
    print(f"Required sample size per arm (80% power, alpha=0.05): ~{n_req:,}")
    print(f"Simulated power at n={n_req:,}: {emp_power:.1%}")

    looks, fpr_no_peek, fpr_peek = simulate_peeking_false_positives(
        p_null=p_a, n_per_arm=5_000, max_looks=20, alpha=0.05, n_sims=2000
    )
    print(f"FPR after 20 daily looks — No peeking: {fpr_no_peek[-1]:.1%} | With peeking: {fpr_peek[-1]:.1%}")

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = out_dir / "ab_peeking_false_positives.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(looks, fpr_peek * 100, marker="o", label="With daily peeking (cumulative)")
    plt.plot(looks, fpr_no_peek * 100, marker="s", label="Without peeking (single test at look k)")
    plt.axhline(5.0, linestyle="--", color="gray", label="Nominal 5% alpha")
    plt.title("False-Positive Rate Under Null: With vs. Without Peeking")
    plt.xlabel("Number of Daily Looks / Checkpoints")
    plt.ylabel("False-Positive Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    print(f"Saved chart to {chart_path.resolve()}")


if __name__ == "__main__":
    main()