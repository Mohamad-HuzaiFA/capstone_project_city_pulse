import duckdb

# Create an in-memory DuckDB connection
conn = duckdb.connect(":memory:")

# 1. Inspect dataset schema
print("\n========== DATASET SCHEMA ==========")

schema = conn.execute("""
    DESCRIBE
    SELECT *
    FROM read_parquet(
        'data/raw/*.parquet',
        union_by_name = true
    )
""").fetchdf()

print(schema.to_string(index=False))


# 2. Count total records across all three files
print("\n========== TOTAL RECORDS ==========")

total_records = conn.execute("""
    SELECT COUNT(*) AS total_records
    FROM read_parquet(
        'data/raw/*.parquet',
        union_by_name = true
    )
""").fetchone()

print(f"Total records: {total_records[0]:,}")


# 3. Inspect sample records
print("\n========== SAMPLE RECORDS ==========")

sample = conn.execute("""
    SELECT *
    FROM read_parquet(
        'data/raw/*.parquet',
        union_by_name = true
    )
    LIMIT 5
""").fetchdf()

print(sample.to_string(index=False))


# 4. Count records per individual file
print("\n========== RECORDS PER FILE ==========")

file_counts = conn.execute("""
    SELECT
        filename,
        COUNT(*) AS total_records
    FROM read_parquet(
        'data/raw/*.parquet',
        filename = true,
        union_by_name = true
    )
    GROUP BY filename
    ORDER BY filename
""").fetchdf()

print(file_counts.to_string(index=False))

conn.close()