import duckdb

# 1. Create an in-memory connection
conn = duckdb.connect(":memory:")

# 2. Create a sample dataset
conn.execute("""
    CREATE TABLE taxi_trips (
        trip_id INTEGER,
        passenger_count INTEGER,
        trip_distance DOUBLE,
        fare_amount DOUBLE
    )
""")

# 3. Insert sample records
conn.execute("""
    INSERT INTO taxi_trips VALUES
    (1, 2, 5.4, 18.50),
    (2, 1, 2.1, 12.00),
    (3, 3, 10.2, 35.00),
    (4, 1, 0.0, 50.00)
""")

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