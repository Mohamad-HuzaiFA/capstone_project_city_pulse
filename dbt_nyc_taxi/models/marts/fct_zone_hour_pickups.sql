{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['zone_id', 'pickup_hour']
    )
}}

with silver_trips as (
    select
        pickup_location_id as zone_id,
        date_trunc('hour', pickup_datetime) as pickup_hour,
        fare_amount,
        trip_minutes
    from {{ ref('stg_yellow_trips') }}
    {% if is_incremental() %}
    where date_trunc('hour', pickup_datetime) >= (
        select coalesce(max(pickup_hour), timestamp '1900-01-01 00:00:00') - interval '24 hours'
        from {{ this }}
    )
    {% endif %}
)

select
    zone_id,
    pickup_hour,
    count(*) as pickups,
    avg(fare_amount) as avg_fare,
    avg(trip_minutes) as avg_trip_minutes
from silver_trips
group by zone_id, pickup_hour
