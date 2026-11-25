#!/usr/bin/env python3
"""
Exercise 1.3: Bootstrap CI for card vs. cash tip percentage & borough breakdown.
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    max_sample: int = 100_000,
) -> tuple[float, float, float]:
    """Return (point_estimate_mean, lower_bound, upper_bound) via memory-safe bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    point_mean = float(values.mean())
    sample = rng.choice(values, size=min(n, max_sample), replace=False) if n > max_sample else values
    m = len(sample)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot_means[i] = sample[rng.integers(0, m, size=m)].mean()
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return (point_mean, float(low), float(high))


def main() -> None:
    trips_glob = "data/raw/yellow_tripdata_2026-*.parquet"
    zones_path = Path("data/raw/taxi_zone_lookup.csv")

    con = duckdb.connect(":memory:")
    # Verify lookup uniqueness first to avoid join fan-out (row multiplication)
    if zones_path.exists():
        dup_zones = con.execute(f"""
            SELECT COUNT(*) - COUNT(DISTINCT LocationID)
            FROM read_csv_auto('{zones_path.as_posix()}', header=true)
        """).fetchone()[0]
        assert dup_zones == 0, f"Duplicate LocationIDs in lookup: {dup_zones}"

        df = con.execute(f"""
            SELECT
                t.payment_type,
                z.Borough AS pickup_borough,
                (t.tip_amount / t.fare_amount) * 100.0 AS tip_pct
            FROM read_parquet('{trips_glob}') AS t
            LEFT JOIN read_csv_auto('{zones_path.as_posix()}', header=true) AS z
                ON t.PULocationID = z.LocationID
            WHERE t.payment_type IN (1, 2)
              AND t.fare_amount > 0
              AND t.tip_amount >= 0
              AND (t.tip_amount / t.fare_amount) <= 2.0
        """).df()
    else:
        print("Note: data/raw/taxi_zone_lookup.csv not found; running without Borough.")
        df = con.execute(f"""
            SELECT
                payment_type,
                'Unknown' AS pickup_borough,
                (tip_amount / fare_amount) * 100.0 AS tip_pct
            FROM read_parquet('{trips_glob}')
            WHERE payment_type IN (1, 2)
              AND fare_amount > 0
              AND tip_amount >= 0
              AND (tip_amount / fare_amount) <= 2.0
        """).df()
    con.close()

    for pay_code, label in [(1, "Card (payment_type=1)"), (2, "Cash (payment_type=2)")]:
        arr = df.loc[df["payment_type"] == pay_code, "tip_pct"].to_numpy()
        mean_val, lo, hi = bootstrap_mean_ci(arr, n_boot=1000, seed=42)
        print(f"{label}: n={len(arr):,} | mean={mean_val:.2f}% | 95% CI=[{lo:.2f}%, {hi:.2f}%]")

    print("\n--- Top Boroughs (Card vs. Cash Mean Tip %) ---")
    summary = (
        df.groupby(["pickup_borough", "payment_type"], dropna=False)["tip_pct"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    print(summary.sort_values("count", ascending=False).head(12).to_string(index=False))

    print("\n--- Cash Tip Data-Collection Bias (3 sentences) ---")
    print(
        "Card payments are processed electronically through the fare terminal, so tip amounts "
        "are automatically logged in the trip record, whereas cash tips handed directly to "
        "drivers are not captured by the meter system. As a result, recorded cash tip averages "
        "reflect unrecorded/missing tip data rather than actual passenger generosity. Using raw "
        "cash tip records to compare tipping behavior or driver earnings creates severe measurement bias."
    )


if __name__ == "__main__":
    main()