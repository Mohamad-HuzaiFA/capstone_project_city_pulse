import duckdb

# 1. Create an in-memory connection
conn = duckdb.connect(":memory:")

# 4. Perform an analytical query
result = conn.execute("""
    SELECT
        COUNT(*) AS total_trips,
        AVG(fare_amount) AS average_fare,
        MAX(trip_distance) AS longest_distance
    FROM taxi_trips
""").fetchdf()

print(result)

conn.close()