#!/usr/bin/env python3
"""
Reusable NYC TLC Yellow Taxi data profiler.

Profiles one or more monthly Parquet files with DuckDB and writes a Markdown
report containing:
  - dataset and per-file row counts
  - null counts and percentages
  - numeric summaries and percentiles
  - data-quality rule violations
  - exact sequential row counts dropped by explicitly selected rules

By default, NO rows are dropped. To drop records, explicitly pass --drop-rule
for each rule and provide --cleaned-output. Raw source files are never changed.

Example:
    python src/data_quality_report.py \
        --input "data/raw/*.parquet" \
        --output reports/data_quality_report.md

    # Explicitly drop only negative-duration and missing-timestamp records:
    python src/data_quality_report.py \
        --input "data/raw/*.parquet" \
        --output reports/data_quality_report.md \
        --drop-rule negative_duration \
        --drop-rule missing_pickup_timestamp \
        --cleaned-output data/processed/taxi_trips_cleaned.parquet
"""

from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


# The thresholds are intentionally explicit and easy to change.
# A rule being detected does NOT mean it should automatically be dropped.
RULES: dict[str, dict[str, str]] = {
    "negative_fare": {
        "description": "Fare amount is below zero",
        "condition": "fare_amount < 0",
        "severity": "Warning — may represent a refund or financial reversal",
    },
    "negative_distance": {
        "description": "Trip distance is below zero",
        "condition": "trip_distance < 0",
        "severity": "Potential rejection — verify source semantics first",
    },
    "zero_distance": {
        "description": "Trip distance is exactly zero",
        "condition": "trip_distance = 0",
        "severity": "Warning — can be legitimate or indicate missing meter data",
    },
    "zero_distance_high_fare": {
        "description": "Zero distance with a fare greater than $50",
        "condition": "trip_distance = 0 AND fare_amount > 50",
        "severity": "High-priority warning — investigate before deciding",
    },
    "extreme_distance": {
        "description": "Trip distance is greater than 100 miles",
        "condition": "trip_distance > 100",
        "severity": "High-priority warning — threshold is an investigation rule",
    },
    "long_duration": {
        "description": "Trip duration is greater than 24 hours",
        "condition": (
            "tpep_pickup_datetime IS NOT NULL "
            "AND tpep_dropoff_datetime IS NOT NULL "
            "AND date_diff('second', tpep_pickup_datetime, "
            "tpep_dropoff_datetime) > 86400"
        ),
        "severity": "High-priority warning — investigate timestamp quality",
    },
    "negative_duration": {
        "description": "Drop-off timestamp is earlier than pickup timestamp",
        "condition": (
            "tpep_pickup_datetime IS NOT NULL "
            "AND tpep_dropoff_datetime IS NOT NULL "
            "AND tpep_dropoff_datetime < tpep_pickup_datetime"
        ),
        "severity": "Potential rejection — verify source semantics first",
    },
    "missing_pickup_timestamp": {
        "description": "Pickup timestamp is missing",
        "condition": "tpep_pickup_datetime IS NULL",
        "severity": "Potential rejection — required for time-based analysis",
    },
    "missing_dropoff_timestamp": {
        "description": "Drop-off timestamp is missing",
        "condition": "tpep_dropoff_datetime IS NULL",
        "severity": "Potential rejection — required for duration analysis",
    },
}

REQUIRED_COLUMNS = {
    "negative_fare": {"fare_amount"},
    "negative_distance": {"trip_distance"},
    "zero_distance": {"trip_distance"},
    "zero_distance_high_fare": {"trip_distance", "fare_amount"},
    "extreme_distance": {"trip_distance"},
    "long_duration": {"tpep_pickup_datetime", "tpep_dropoff_datetime"},
    "negative_duration": {"tpep_pickup_datetime", "tpep_dropoff_datetime"},
    "missing_pickup_timestamp": {"tpep_pickup_datetime"},
    "missing_dropoff_timestamp": {"tpep_dropoff_datetime"},
}


def sql_identifier(name: str) -> str:
    """Safely quote a DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str) -> str:
    """Safely quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def resolve_input_files(pattern: str) -> list[Path]:
    """Resolve a file path or glob pattern into existing Parquet files."""
    matches = glob.glob(pattern)
    if not matches and Path(pattern).is_file():
        matches = [pattern]

    files = sorted({Path(item).expanduser().resolve() for item in matches if Path(item).is_file()})
    if not files:
        raise FileNotFoundError(
            f"No input files matched {pattern!r}. Check the path or glob pattern."
        )

    non_parquet = [path for path in files if path.suffix.lower() != ".parquet"]
    if non_parquet:
        raise ValueError(
            "This script currently expects Parquet input. "
            f"Non-Parquet files found: {non_parquet}"
        )
    return files


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple Markdown table and escape embedded pipe characters."""
    def cell(value: Any) -> str:
        if value is None:
            return "—"
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
        if number != number:  # NaN
            return "—"
        return f"{number:,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_columns(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    rows = conn.execute("DESCRIBE SELECT * FROM source_data").fetchall()
    return [(row[0], row[1].upper()) for row in rows]


def get_numeric_columns(columns: list[tuple[str, str]]) -> list[str]:
    """Return numeric columns that represent measurements, not category codes."""
    numeric_types = (
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
        "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
        "FLOAT", "DOUBLE", "REAL", "DECIMAL",
    )
    # These fields are numeric in the source schema but are identifiers/codes.
    categorical_identifiers = {
        "vendorid", "ratecodeid", "payment_type",
        "pulocationid", "dolocationid",
    }
    result = []
    for name, dtype in columns:
        if name.lower() == "filename" or name.lower() in categorical_identifiers:
            continue
        if any(dtype.startswith(kind) for kind in numeric_types):
            result.append(name)
    return result


def get_categorical_columns(columns: list[tuple[str, str]]) -> list[str]:
    """Identify known code/ID fields and text columns for frequency profiling."""
    categorical_identifiers = {
        "vendorid", "ratecodeid", "payment_type",
        "pulocationid", "dolocationid",
    }
    text_types = ("VARCHAR", "CHAR", "TEXT", "STRING", "ENUM")
    result = []
    for name, dtype in columns:
        if name.lower() == "filename":
            continue
        if name.lower() in categorical_identifiers or dtype.startswith(text_types):
            result.append(name)
    return result


def build_report(
    conn: duckdb.DuckDBPyConnection,
    files: list[Path],
    selected_drop_rules: list[str],
    cleaned_output: Path | None,
) -> str:
    columns = get_columns(conn)
    column_names = {name for name, _ in columns}

    total_rows = conn.execute("SELECT COUNT(*) FROM source_data").fetchone()[0]
    # DuckDB's filename=true option adds a virtual provenance column. Keep it
    # visible in the report, but distinguish it from the source schema.
    source_columns = [(name, dtype) for name, dtype in columns if name.lower() != "filename"]
    source_column_count = len(source_columns)
    metadata_column_count = len(columns) - source_column_count

    # Dataset overview
    file_rows = conn.execute(
        """
        SELECT filename, COUNT(*) AS row_count
        FROM source_data
        GROUP BY filename
        ORDER BY filename
        """
    ).fetchall()

    overview_rows = [
        ["Total rows", f"{total_rows:,}"],
        ["Source data columns", f"{source_column_count:,}"],
        ["Technical metadata columns", f"{metadata_column_count:,}"],
        ["Total columns profiled", f"{len(columns):,}"],
        ["Number of input files", f"{len(files):,}"],
        ["Rows retained in current run", "Calculated below"],
    ]

    file_table_rows = []
    for filename, row_count in file_rows:
        path = Path(filename)
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            size_text = f"{size_mb:,.2f} MB"
        except OSError:
            size_text = "Unavailable"
        file_table_rows.append([path.name, f"{row_count:,}", size_text])

    # Missing-value analysis for every column.
    null_expressions = [
        f"COUNT(*) FILTER (WHERE {sql_identifier(name)} IS NULL) AS {sql_identifier(name)}"
        for name, _ in columns
    ]
    null_counts = conn.execute(
        "SELECT " + ", ".join(null_expressions) + " FROM source_data"
    ).fetchone()

    null_rows = []
    for (name, _dtype), null_count in zip(columns, null_counts):
        percentage = (null_count / total_rows * 100) if total_rows else 0
        null_rows.append([name, f"{null_count:,}", f"{percentage:.4f}%"])

    # Numeric profiling. Quantiles are calculated by DuckDB over the raw values.
    numeric_columns = get_numeric_columns(columns)
    numeric_rows = []
    for name in numeric_columns:
        col = sql_identifier(name)
        stats = conn.execute(
            f"""
            SELECT
                MIN({col}),
                MAX({col}),
                AVG({col}),
                MEDIAN({col}),
                STDDEV_SAMP({col}),
                QUANTILE_CONT({col}, 0.25),
                QUANTILE_CONT({col}, 0.75),
                QUANTILE_CONT({col}, 0.95)
            FROM source_data
            """
        ).fetchone()
        numeric_rows.append(
            [
                name,
                *[format_number(value) for value in stats],
            ]
        )

    # Categorical frequency profiles: do not calculate means for numeric codes.
    categorical_columns = get_categorical_columns(columns)
    categorical_sections: list[str] = []
    for name in categorical_columns:
        col = sql_identifier(name)
        values = conn.execute(
            f"""
            SELECT
                COALESCE(CAST({col} AS VARCHAR), '<NULL>') AS category,
                COUNT(*) AS row_count
            FROM source_data
            GROUP BY {col}
            ORDER BY row_count DESC, category
            LIMIT 30
            """
        ).fetchall()
        category_rows = [
            [category, f"{count:,}", f"{(count / total_rows * 100):.4f}%"]
            for category, count in values
        ]
        categorical_sections.extend([
            f"### `{name}`",
            "",
            "Top 30 values by frequency (including NULL when present).",
            "",
            markdown_table(["Value", "Rows", "% of all rows"], category_rows),
            "",
        ])

    # For TLC Yellow Taxi data, payment_type=0 (Flex Fare) may have a distinct
    # missingness pattern. Report it rather than treating those NULLs as random.
    missingness_cross_tab = []
    if {"payment_type", "passenger_count", "RatecodeID",
        "store_and_fwd_flag", "congestion_surcharge", "Airport_fee"}.issubset(column_names):
        flex_missing_columns = [
            "passenger_count", "RatecodeID", "store_and_fwd_flag",
            "congestion_surcharge", "Airport_fee",
        ]
        grouped = conn.execute(
            """
            SELECT
                payment_type,
                COUNT(*) AS row_count,
                COUNT(*) FILTER (WHERE passenger_count IS NULL) AS passenger_count_null,
                COUNT(*) FILTER (WHERE RatecodeID IS NULL) AS ratecodeid_null,
                COUNT(*) FILTER (WHERE store_and_fwd_flag IS NULL) AS store_and_fwd_flag_null,
                COUNT(*) FILTER (WHERE congestion_surcharge IS NULL) AS congestion_surcharge_null,
                COUNT(*) FILTER (WHERE Airport_fee IS NULL) AS airport_fee_null
            FROM source_data
            GROUP BY payment_type
            ORDER BY payment_type NULLS LAST
            """
        ).fetchall()
        for row in grouped:
            payment_type, count, *null_counts = row
            missingness_cross_tab.append([
                "<NULL>" if payment_type is None else str(payment_type),
                f"{count:,}",
                *[
                    f"{n:,} ({(n / count * 100):.2f}%)" if count else "—"
                    for n in null_counts
                ],
            ])

    # Rule violations are first counted on the original data.
    rule_rows = []
    raw_rule_counts: dict[str, int] = {}
    available_rules: list[str] = []

    for rule_name, rule in RULES.items():
        missing_columns = REQUIRED_COLUMNS[rule_name] - column_names
        if missing_columns:
            rule_rows.append(
                [
                    rule_name,
                    rule["description"],
                    "Not evaluated",
                    "—",
                    rule["severity"],
                    f"Missing columns: {', '.join(sorted(missing_columns))}",
                ]
            )
            continue

        available_rules.append(rule_name)
        count = conn.execute(
            f"SELECT COUNT(*) FROM source_data WHERE {rule['condition']}"
        ).fetchone()[0]
        raw_rule_counts[rule_name] = count
        pct = count / total_rows * 100 if total_rows else 0
        rule_rows.append(
            [
                rule_name,
                rule["description"],
                f"{count:,}",
                f"{pct:.4f}%",
                rule["severity"],
                "Evaluated",
            ]
        )

    # Build a working copy so the source_data table and raw files remain unchanged.
    conn.execute("CREATE OR REPLACE TEMP TABLE remaining_data AS SELECT * FROM source_data")
    drop_rows = []
    total_dropped = 0

    for rule_name in selected_drop_rules:
        if rule_name not in RULES:
            raise ValueError(f"Unknown drop rule: {rule_name}")
        if rule_name not in available_rules:
            raise ValueError(
                f"Cannot apply {rule_name!r}: one or more required columns are absent."
            )

        condition = RULES[rule_name]["condition"]
        before_count = conn.execute(
            "SELECT COUNT(*) FROM remaining_data"
        ).fetchone()[0]
        affected = conn.execute(
            f"SELECT COUNT(*) FROM remaining_data WHERE {condition}"
        ).fetchone()[0]
        conn.execute(f"DELETE FROM remaining_data WHERE {condition}")
        after_count = conn.execute(
            "SELECT COUNT(*) FROM remaining_data"
        ).fetchone()[0]
        total_dropped += affected

        drop_rows.append(
            [
                rule_name,
                f"{before_count:,}",
                f"{affected:,}",
                f"{after_count:,}",
            ]
        )

    retained_rows = conn.execute("SELECT COUNT(*) FROM remaining_data").fetchone()[0]
    overview_rows[-1][1] = f"{retained_rows:,}"

    # Optional cleaned output: only created when the user explicitly selects rules.
    output_note = "No cleaned Parquet file was written."
    if cleaned_output is not None:
        cleaned_output.parent.mkdir(parents=True, exist_ok=True)
        output_literal = sql_string(str(cleaned_output.resolve()))
        conn.execute(
            f"COPY (SELECT * FROM remaining_data) TO {output_literal} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        output_note = f"Cleaned output written to `{cleaned_output}`."

    dropped_pct = total_dropped / total_rows * 100 if total_rows else 0
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_parts = [
        "# Data Quality Report",
        "",
        f"- **Generated:** {generated_at}",
        "- **Engine:** DuckDB",
        f"- **Input pattern:** `{args_input_display(files)}`",
        "",
        "## 1. Dataset overview",
        "",
        markdown_table(["Metric", "Value"], overview_rows),
        "",
        "### Input files",
        "",
        markdown_table(["File", "Rows", "File size"], file_table_rows),
        "",
        "## 2. Missing-value analysis",
        "",
        "Percentages use the total row count as the denominator.",
        "",
        markdown_table(["Column", "Missing values", "Missing percentage"], null_rows),
        "",
        "## 3. Numeric profiling",
        "",
        "Statistics exclude NULL values. Percentiles use DuckDB `QUANTILE_CONT`.",
        "",
        markdown_table(
            [
                "Column", "Minimum", "Maximum", "Mean", "Median",
                "Std. dev.", "P25", "P75", "P95",
            ],
            numeric_rows,
        ),
        "",
        "## 4. Categorical profiling",
        "",
        "Numeric-looking identifiers and codes are treated as categories, not "
        "continuous measurements. Frequency distributions are limited to the "
        "30 most common values per column.",
        "",
        *categorical_sections,
        "### Missingness by payment type",
        "",
        "This cross-tab is included when the expected TLC columns are present. "
        "It helps identify category-dependent missingness (for example, fields "
        "that are systematically absent for a payment category) rather than "
        "assuming all NULL values are random.",
        "",
        markdown_table(
            [
                "payment_type", "Rows",
                "passenger_count NULL", "RatecodeID NULL",
                "store_and_fwd_flag NULL", "congestion_surcharge NULL",
                "Airport_fee NULL",
            ],
            missingness_cross_tab if missingness_cross_tab else [
                ["Not available", "—", "—", "—", "—", "—", "—"]
            ],
        ),
        "",
        "## 5. Data-quality rule violations",
        "",
        "Counts below are calculated against the original input data. "
        "Rules are investigation flags; they do not automatically imply invalid records.",
        "",
        markdown_table(
            ["Rule", "Description", "Affected rows", "% of rows", "Severity / interpretation", "Status"],
            rule_rows,
        ),
        "",
        "## 6. Cleaning and row-impact summary",
        "",
        f"- **Original rows:** {total_rows:,}",
        f"- **Rows dropped:** {total_dropped:,} ({dropped_pct:.4f}%)",
        f"- **Rows retained:** {retained_rows:,}",
        f"- **Selected drop rules:** {', '.join(selected_drop_rules) if selected_drop_rules else 'None'}",
        f"- **Output:** {output_note}",
        "",
        "The per-rule counts below are sequential: each rule is evaluated against "
        "the rows remaining after the previously selected rules. This avoids "
        "double-counting a row that violates multiple selected rules.",
        "",
        markdown_table(
            ["Applied rule", "Rows before rule", "Rows dropped by rule", "Rows after rule"],
            drop_rows if drop_rows else [["None", f"{total_rows:,}", "0", f"{total_rows:,}"]],
        ),
        "",
        "### Important interpretation notes",
        "",
        "- A rule violation is a flag for review, not automatic proof of bad data.",
        "- Numeric identifiers/codes are summarized as categories; their mean and "
        "standard deviation are generally not meaningful.",
        "- The `filename` column is DuckDB provenance metadata added for per-file "
        "reporting; it is not one of the original Parquet source columns.",
        "- Rule violation counts can overlap because one record may violate multiple rules.",
        "- The cleaning table reports sequential drops, so each dropped row is attributed "
        "to the first selected rule that removes it.",
        "- By default, no rows are dropped. Raw Parquet inputs are never modified.",
        "- Missingness should be investigated in context; a missing value may be expected "
        "for a particular category or source process.",
        "",
    ]
    return "\n".join(report_parts)


def args_input_display(files: list[Path]) -> str:
    if len(files) == 1:
        return str(files[0])
    common_parent = Path(files[0]).parent
    if all(path.parent == common_parent for path in files):
        return str(common_parent / "*.parquet")
    return f"{len(files)} resolved Parquet files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile TLC Parquet files and generate a Markdown data-quality report."
    )
    parser.add_argument(
        "--input",
        default="data/raw/*.parquet",
        help="Parquet file path or glob pattern (default: data/raw/*.parquet)",
    )
    parser.add_argument(
        "--output",
        default="reports/data_quality_report.md",
        help="Markdown report path (default: reports/data_quality_report.md)",
    )
    parser.add_argument(
        "--drop-rule",
        action="append",
        choices=sorted(RULES),
        default=[],
        help="Explicitly apply a rule to the working copy. Can be repeated. Default: no drops.",
    )
    parser.add_argument(
        "--cleaned-output",
        help="Optional output Parquet path. Requires at least one --drop-rule.",
    )
    return parser.parse_args()


def main() -> int:
    global args
    args = parse_args()

    try:
        files = resolve_input_files(args.input)
        if args.cleaned_output and not args.drop_rule:
            raise ValueError("--cleaned-output requires at least one --drop-rule.")

        conn = duckdb.connect(":memory:")
        file_list_sql = "[" + ", ".join(sql_string(str(path)) for path in files) + "]"
        conn.execute(
            f"""
            CREATE TABLE source_data AS
            SELECT *
            FROM read_parquet(
                {file_list_sql},
                union_by_name = true,
                filename = true
            )
            """
        )

        report = build_report(
            conn=conn,
            files=files,
            selected_drop_rules=args.drop_rule,
            cleaned_output=Path(args.cleaned_output) if args.cleaned_output else None,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        print(f"Report generated: {output_path.resolve()}")
        print(f"Input files: {len(files)}")
        print(f"Rows profiled: {conn.execute('SELECT COUNT(*) FROM source_data').fetchone()[0]:,}")
        print(f"Rules selected for dropping: {args.drop_rule or 'None'}")
        conn.close()
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
