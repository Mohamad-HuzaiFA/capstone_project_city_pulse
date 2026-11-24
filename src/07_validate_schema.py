
#!/usr/bin/env python3
"""
Exercise 1.2: Validation as Code using Pandera.

Purpose
-------
This script demonstrates how to define a data contract using Pandera,
prepare a dataset using DuckDB, and validate that the resulting data
satisfies the expected schema and business rules.

It performs three tests:

1. Clean sample:
   A sample prepared using the exercise's cleaning policy should pass.

2. Corrupted sample:
   Intentionally invalid values should be detected by Pandera.

3. NULL sample:
   Missing values in required columns should be detected.

Important distinction
---------------------
Cleaning and validation are different operations.

Cleaning prepares data according to a chosen analytical policy.
Validation checks whether the prepared data satisfies a contract.

Some unusual taxi records may represent legitimate adjustments or
special cases. The filters below are exercise-specific analytical
decisions, not universal proof that a record is invalid.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors


# ============================================================
# 1. DATA CONTRACT: DEFINE WHAT VALID DATA LOOKS LIKE
# ============================================================

"""
A schema is a formal contract describing the structure and constraints
that a DataFrame must satisfy.

Pandera checks things such as:
- Whether required columns exist.
- Whether columns have the expected data types.
- Whether NULL values are allowed.
- Whether individual values satisfy specified conditions.
- Whether relationships between columns are valid.

This schema describes the dataset AFTER applying our exercise's
cleaning policy.
"""

CLEAN_TRIP_SCHEMA = pa.DataFrameSchema(
    columns={
        "VendorID": pa.Column(
            int,
            pa.Check.isin([1, 2, 6, 7]),
            nullable=False,
            coerce=True,
        ),
        "tpep_pickup_datetime": pa.Column(
            "datetime64[ns]",
            nullable=False,
            coerce=True,
        ),
        "tpep_dropoff_datetime": pa.Column(
            "datetime64[ns]",
            nullable=False,
            coerce=True,
        ),
        "passenger_count": pa.Column(
            float,
            pa.Check.between(0, 9),
            nullable=True,
            coerce=True,
        ),
        "trip_distance": pa.Column(
            float,
            [
                pa.Check.ge(0),
                pa.Check.le(100),
            ],
            nullable=False,
            coerce=True,
        ),
        "PULocationID": pa.Column(
            int,
            pa.Check.between(1, 265),
            nullable=False,
            coerce=True,
        ),
        "DOLocationID": pa.Column(
            int,
            pa.Check.between(1, 265),
            nullable=False,
            coerce=True,
        ),
        "payment_type": pa.Column(
            int,
            pa.Check.isin([0, 1, 2, 3, 4, 5, 6]),
            nullable=False,
            coerce=True,
        ),
        "fare_amount": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=False,
            coerce=True,
        ),
        "tip_amount": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=False,
            coerce=True,
        ),
        "total_amount": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=False,
            coerce=True,
        ),
    },

    # DataFrame-level checks validate relationships between columns.
    # These rules cannot be expressed by checking one column alone.
    checks=[
        pa.Check(
            lambda df: (
                df["tpep_dropoff_datetime"]
                >= df["tpep_pickup_datetime"]
            ).all(),
            name="non_negative_duration",
            error="Dropoff timestamp is earlier than pickup timestamp.",
        ),
        pa.Check(
            lambda df: (
                (
                    df["tpep_dropoff_datetime"]
                    - df["tpep_pickup_datetime"]
                ).dt.total_seconds() <= 86400
            ).all(),
            name="max_24h_duration",
            error="Found trips exceeding 24 hours.",
        ),
        pa.Check(
            lambda df: ~(
                (df["trip_distance"] == 0)
                & (df["fare_amount"] > 50)
            ).any(),
            name="no_zero_distance_high_fare",
            error="Found zero-distance trips with fare above $50.",
        ),
    ],

    # strict=False means extra columns are allowed.
    # For example, the original Parquet data contains additional
    # fee columns that are not explicitly defined in this schema.
    strict=False,
)


# ============================================================
# 2. CLEANING POLICY: DEFINE OUR ANALYTICAL FILTERS
# ============================================================

"""
These filters prepare a sample for this particular exercise.

They are not all universal data-validity rules.

For example:
- Negative fares can represent refunds or adjustments.
- Zero-distance trips can sometimes be legitimate.
- Trips over 100 miles may be valid in unusual circumstances.

We are excluding these records for the current analytical dataset,
but we should document the decision rather than automatically
classifying every flagged record as bad data.

The SQL conditions are also used by the cleaning audit below.
"""

CLEANING_RULES = {
    "missing_pickup": (
        "tpep_pickup_datetime IS NULL",
        "Pickup timestamp is missing.",
    ),
    "missing_dropoff": (
        "tpep_dropoff_datetime IS NULL",
        "Dropoff timestamp is missing.",
    ),
    "negative_duration": (
        "tpep_dropoff_datetime < tpep_pickup_datetime",
        "Dropoff occurs before pickup.",
    ),
    "duration_over_24h": (
        "date_diff('second', tpep_pickup_datetime, "
        "tpep_dropoff_datetime) > 86400",
        "Trip duration exceeds 24 hours.",
    ),
    "negative_distance": (
        "trip_distance < 0",
        "Trip distance is negative.",
    ),
    "distance_over_100": (
        "trip_distance > 100",
        "Trip distance exceeds the exercise's 100-mile threshold.",
    ),
    "negative_fare": (
        "fare_amount < 0",
        "Negative fare; may represent a reversal or adjustment.",
    ),
    "negative_tip": (
        "tip_amount < 0",
        "Negative tip amount.",
    ),
    "negative_total": (
        "total_amount < 0",
        "Negative total amount.",
    ),
    "zero_distance_high_fare": (
        "trip_distance = 0 AND fare_amount > 50",
        "Zero-distance trip with fare above $50.",
    ),
    "invalid_pickup_location": (
        "PULocationID IS NULL OR PULocationID NOT BETWEEN 1 AND 265",
        "Missing or out-of-range pickup location ID.",
    ),
    "invalid_dropoff_location": (
        "DOLocationID IS NULL OR DOLocationID NOT BETWEEN 1 AND 265",
        "Missing or out-of-range dropoff location ID.",
    ),
    "missing_required_financial_values": (
        "fare_amount IS NULL OR tip_amount IS NULL "
        "OR total_amount IS NULL",
        "One or more required financial values are missing.",
    ),
    "missing_vendor": (
        "VendorID IS NULL",
        "Vendor ID is missing.",
    ),
    "missing_payment_type": (
        "payment_type IS NULL",
        "Payment type is missing.",
    ),
}


# ============================================================
# 3. CLEANING AUDIT
# ============================================================

def print_cleaning_audit(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> None:
    """
    Display how many raw records violate each cleaning rule.

    This is an audit, not a sequential deletion report.

    Important:
    A single record can violate multiple rules. Therefore, the counts
    below may overlap and must not be added together to calculate
    the total number of records excluded.

    We perform this audit before sampling so that we understand the
    impact of our cleaning policy on the source dataset.
    """

    print("\n========== CLEANING AUDIT ==========")

    total_rows = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(parquet_path)],
    ).fetchone()[0]

    print(f"Original records: {total_rows:,}\n")

    for rule_name, (condition, description) in CLEANING_RULES.items():

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet(?)
            WHERE {condition}
            """,
            [str(parquet_path)],
        ).fetchone()[0]

        percentage = (
            count / total_rows * 100
            if total_rows
            else 0
        )

        print(
            f"{rule_name:<38} "
            f"{count:>12,} "
            f"({percentage:.4f}%)"
        )
        print(f"    {description}")


# ============================================================
# 4. LOAD A CLEAN RANDOM SAMPLE
# ============================================================

def load_clean_sample(
    parquet_path: Path,
    limit: int = 50_000,
) -> pd.DataFrame:
    """
    Prepare a sample of records that satisfy our cleaning policy.

    DuckDB reads the Parquet file directly, applies SQL filters,
    randomly orders the remaining records, and returns the sample
    as a Pandas DataFrame.

    Why random sampling?
    --------------------
    Using LIMIT alone does not guarantee a representative sample.
    It can simply select the first records encountered.

    ORDER BY random() gives us a randomized sample from the eligible
    records, although it requires additional processing.

    Why explicit IS NOT NULL conditions?
    -------------------------------------
    SQL comparisons involving NULL do not evaluate to TRUE.
    Therefore, conditions like fare_amount >= 0 implicitly exclude
    NULL values.

    We explicitly state the NULL requirements here to make our
    cleaning policy easier to understand and maintain.

    Note:
    Random sampling is useful for schema testing, but passing this
    sample does not prove that the entire source dataset is valid.
    """

    if limit <= 0:
        raise ValueError("Sample limit must be greater than zero.")

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}"
        )

    con = duckdb.connect(":memory:")

    try:
        # Build the WHERE clause from our documented cleaning rules.
        # Every condition must be satisfied for a row to be included.
        where_clause = " AND ".join(
            f"NOT ({condition})"
            for condition, _ in CLEANING_RULES.values()
        )

        query = f"""
            SELECT *
            FROM read_parquet(?)
            WHERE {where_clause}
            ORDER BY random()
            LIMIT ?
        """

        df = con.execute(
            query,
            [str(parquet_path), limit],
        ).df()

    finally:
        con.close()

    return df


# ============================================================
# 5. CREATE INTENTIONALLY CORRUPTED DATA
# ============================================================

def make_corrupted_sample(
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a copy of clean data and inject known validation failures.

    We deliberately introduce four problems:

    1. Negative fare:
       Violates the fare_amount >= 0 constraint.

    2. Extreme distance:
       Violates the trip_distance <= 100 constraint.

    3. Negative trip duration:
       Dropoff timestamp occurs before pickup timestamp.

    4. Zero distance with high fare:
       Violates the custom cross-column business rule.

    The original clean DataFrame is not modified.
    """

    if len(clean_df) < 4:
        raise ValueError(
            "At least four clean records are required "
            "to create the corrupted sample."
        )

    bad = clean_df.head(100).copy()

    bad.loc[0, "fare_amount"] = -25.00

    bad.loc[1, "trip_distance"] = 450.00

    bad.loc[2, "tpep_dropoff_datetime"] = (
        bad.loc[2, "tpep_pickup_datetime"]
        - pd.Timedelta(minutes=10)
    )

    bad.loc[3, "trip_distance"] = 0.00
    bad.loc[3, "fare_amount"] = 95.00

    return bad


# ============================================================
# 6. CREATE DATA WITH MISSING VALUES
# ============================================================

def make_null_sample(
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a separate sample containing NULL values in required fields.

    This test verifies the nullable=False constraints in our schema.

    We intentionally set:
    - fare_amount to NaN.
    - tpep_pickup_datetime to NaT.

    NaN and NaT are Pandas representations of missing numeric and
    datetime values respectively.

    The schema should reject both records.
    """

    if len(clean_df) < 2:
        raise ValueError(
            "At least two clean records are required "
            "to create the NULL sample."
        )

    bad = clean_df.head(100).copy()

    bad.loc[0, "fare_amount"] = float("nan")
    bad.loc[1, "tpep_pickup_datetime"] = pd.NaT

    return bad


# ============================================================
# 7. REUSABLE VALIDATION TEST
# ============================================================

def validate_expected_failure(
    label: str,
    df: pd.DataFrame,
) -> None:
    """
    Verify that a deliberately invalid DataFrame is rejected.

    lazy=True tells Pandera to collect multiple validation failures
    before raising SchemaErrors.

    This is useful during development because we can see several
    problems in one execution instead of fixing them one by one.

    If validation unexpectedly succeeds, we raise AssertionError.
    """

    try:
        CLEAN_TRIP_SCHEMA.validate(df, lazy=True)

    except SchemaErrors as exc:

        print(f"\n[PASS - EXPECTED FAILURE] {label}")

        failure_columns = [
            "schema_context",
            "column",
            "check",
            "failure_case",
        ]

        print(
            exc.failure_cases[failure_columns].to_string(index=False)
        )

    else:
        raise AssertionError(
            f"{label} unexpectedly passed validation!"
        )


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main() -> None:
    """
    Run the complete Exercise 1.2 validation workflow.

    Execution order:

    1. Read and audit the original Parquet dataset.
    2. Prepare a randomized clean sample.
    3. Validate that the clean sample passes.
    4. Validate that intentionally corrupted data fails.
    5. Validate that required NULL values are rejected.

    This demonstrates both positive and negative testing of a
    data contract.
    """

    raw_file = Path(
        "data/raw/yellow_tripdata_2026-01.parquet"
    )

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_file}"
        )

    con = duckdb.connect(":memory:")

    try:
        # Understand how many raw records violate each cleaning rule.
        print_cleaning_audit(con, raw_file)

    finally:
        con.close()

    # Prepare a randomized sample using the documented cleaning policy.
    clean_df = load_clean_sample(raw_file, limit=50_000)

    if clean_df.empty:
        raise ValueError(
            "No records remained after applying the cleaning policy."
        )

    # TEST 1: Valid data should pass.
    validated_df = CLEAN_TRIP_SCHEMA.validate(
        clean_df,
        lazy=True,
    )

    print(
        f"\n[PASS] Clean sample validated successfully "
        f"({len(validated_df):,} rows)."
    )

    # TEST 2: Invalid values should be detected.
    corrupted_df = make_corrupted_sample(clean_df)

    validate_expected_failure(
        "Corrupted sample was blocked by the schema",
        corrupted_df,
    )

    # TEST 3: NULL values in required columns should be detected.
    null_df = make_null_sample(clean_df)

    validate_expected_failure(
        "NULL sample was blocked by the schema",
        null_df,
    )

    print("\n========== VALIDATION SUMMARY ==========")
    print("Clean sample: PASSED")
    print("Corrupted sample: CORRECTLY REJECTED")
    print("NULL sample: CORRECTLY REJECTED")


if __name__ == "__main__":
    main()
