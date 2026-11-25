with source_trips as (
    select
        *,
        current_timestamp as loaded_at_utc
    from read_parquet('../data/raw/yellow_tripdata_2026-*.parquet', union_by_name=true, filename=true)
),

typed_and_filtered as (
    select
        cast(VendorID as integer) as vendor_id,
        cast(tpep_pickup_datetime as timestamp) as pickup_datetime,
        cast(tpep_dropoff_datetime as timestamp) as dropoff_datetime,
        cast(passenger_count as double) as passenger_count,
        cast(trip_distance as double) as trip_distance_miles,
        cast(RatecodeID as integer) as ratecode_id,
        cast(store_and_fwd_flag as varchar) as store_and_fwd_flag,
        cast(PULocationID as integer) as pickup_location_id,
        cast(DOLocationID as integer) as dropoff_location_id,
        cast(payment_type as integer) as payment_type,
        cast(fare_amount as double) as fare_amount,
        cast(tip_amount as double) as tip_amount,
        cast(total_amount as double) as total_amount,
        date_diff('second', cast(tpep_pickup_datetime as timestamp), cast(tpep_dropoff_datetime as timestamp)) / 60.0 as trip_minutes,
        filename as source_file,
        loaded_at_utc
    from source_trips
    where tpep_pickup_datetime is not null
      and tpep_dropoff_datetime is not null
      and cast(tpep_dropoff_datetime as timestamp) >= cast(tpep_pickup_datetime as timestamp)
      and date_diff('second', cast(tpep_pickup_datetime as timestamp), cast(tpep_dropoff_datetime as timestamp)) <= 86400
      and coalesce(cast(trip_distance as double), 0) >= 0
      and coalesce(cast(trip_distance as double), 0) <= 100
      and coalesce(cast(fare_amount as double), 0) >= 0
      and coalesce(cast(tip_amount as double), 0) >= 0
      and coalesce(cast(total_amount as double), 0) >= 0
      and not (
          coalesce(cast(trip_distance as double), 0) = 0
          and coalesce(cast(fare_amount as double), 0) > 50
      )
      and cast(PULocationID as integer) between 1 and 265
      and cast(DOLocationID as integer) between 1 and 265
)

select * from typed_and_filtered
