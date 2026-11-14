# Example data

Data used by the user guide examples, stored as Parquet.

## Real data (plotting page)

| File | Data | Source |
|---|---|---|
| `39027_gdf.parquet` | Gauged daily flow (m³/s), 1968 to 2025 | UK National River Flow Archive, station 39027, Pang at Pangbourne: <https://nrfa.ceh.ac.uk/data/station/info/39027>. The one missing day (2024-09-30, NRFA flag "M") is null. |
| `chimn_daily_ta_precip_2014-2025.parquet` | Daily air temperature (°C) and precipitation (mm), 2014 to 2025 | COSMOS-UK, Chimney Meadows site: <https://cosmos.ceh.ac.uk/sites/CHIMN> |

## Example data (other pages)

Derived from real 15-minute records, then rescaled and given random noise, so they are not real measurements of
any site. They keep realistic gaps.

| File | Data |
|---|---|
| `example_flow_15min.parquet` | 15-minute river flow (m³/s), 2015 to 2025 |
| `example_flow_15min_secondary.parquet` | A second 15-minute river flow dataset (m³/s), 2015 to 2025, about a quarter higher than the first |
| `example_rainfall_15min.parquet` | 15-minute rainfall (mm), 2015 to 2025 |
