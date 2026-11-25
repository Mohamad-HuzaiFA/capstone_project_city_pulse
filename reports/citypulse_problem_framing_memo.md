# CityPulse Problem-Framing Memo: NYC Taxi Zone-Hour Demand Forecasting

## 1. Problem & Users
CityPulse forecasts taxi pickups for every NYC taxi zone across each of the next 24 hours[cite: 8, 18].
- **Primary users:** Dispatch teams positioning drivers ahead of demand spikes, plus data/ML engineers maintaining the forecasting and lakehouse pipeline[cite: 8, 18].

## 2. Decisions Enabled
- Where and when to pre-position available vehicles across taxi zones over the next 24 hours to reduce idle wait time and missed pickups[cite: 8, 18].
- When to trigger model retraining based on observed drift rather than a blind calendar schedule[cite: 19].

## 3. Success Metric & Baseline
- **Primary evaluation metric:** **WAPE** (Weighted Absolute Percentage Error) across zone-hour pickup forecasts—chosen because total volume error relative to actual pickups directly reflects fleet/demand mismatch at scale[cite: 8, 14, 15].
- **Baseline:** A **seasonal-naive** baseline (e.g., same hour-of-week / prior seasonal pattern), which a gradient-boosted model (LightGBM) must beat by a clear, justified margin on rolling-origin backtests before earning its complexity[cite: 13, 14, 15, 19].

## 4. Key Risks & Mitigations
- **Data & temporal leakage:** Never use future timestamps or target information in lag/rolling features, and use rolling-origin backtests rather than random splits[cite: 13, 14, 15, 20].
- **Data quality & schema traps:** Handle missing/zero-distance/anomalous trips cleanly, validate schemas at boundaries, and avoid silent row multiplication in zone joins[cite: 9, 11, 19].
- **Cloud cost & operational risk:** Keep compute/storage in designated regions/resource groups with `course=ds` tags, avoid always-on endpoints/GPUs/Redshift/EMR/Marketplace extras, run teardown scripts, and monitor drift via Evidently[cite: 7, 8, 17, 19].