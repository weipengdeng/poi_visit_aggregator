# poi_visit_aggregator

Two-stage aggregator for user POI stay/visit data.

## Install

```bash
pip install -e .
```

## Usage

The public API is in `poi_visit_aggregator/aggregator.py`:

- `process_large_poi_data(...)`: stage-1 chunk aggregation to intermediate parquet parts, then stage-2 bucketed final aggregation.
- `read_result(...)`: read the final parquet and parse JSON columns back into Python objects.

```python
from poi_visit_aggregator import process_large_poi_data, read_result

process_large_poi_data(
    input_file="input.parquet",   # or .csv
    output_dir="out",
    output_file="user_poi_agg.parquet",
    notebook=True,               # for Jupyter/Colab
    rhythm_specs={
        # Day/Night split used by daytime_ratio/night_ratio (hour granularity)
        "daytime": {"start_hour": 6, "end_hour": 18},  # 06:00-17:59 vs 18:00-05:59
        # Optional: custom comparisons (stored as JSON; see output columns below)
        "enter_periods": [
            {"name": "day", "start_hour": 6, "end_hour": 18},
            {"name": "night", "start_hour": 18, "end_hour": 6},
        ],
        "weekday_groups": [
            {"name": "weekday", "days": [1, 2, 3, 4, 5]},
            {"name": "weekend", "days": [6, 7]},
        ],
        # Note: date_groups are based on day-of-month counts (dom_01..dom_31)
        "date_groups": [
            {"name": "2024-12-03..05", "ranges": [["2024-12-03", "2024-12-05"]]},
            {"name": "2024-12-06..31", "ranges": [["2024-12-06", "2024-12-31"]]},
            {"name": "non_continuous", "ranges": [["2024-12-01", "2024-12-02"], ["2024-12-06", "2024-12-07"]]},
        ],
    },
)

df = read_result("out/user_poi_agg.parquet")
```

### `process_large_poi_data(...)` input

`process_large_poi_data(...)` expects (at least) these columns:

- `uuid`, `poi_id`, `start_hour`, `end_hour`, `stay_minutes`, `date_str`
- Optional: `location` (will be exported to `poi_coordination.csv`)
- Any other columns are ignored (e.g. `start_time`, `end_time`, admin codes, etc.).

Supported input files: `.csv` or `.parquet`.

### `process_large_poi_data(...)` output

Files written:

- Final parquet: `<output_dir>/<output_file>`
- POI coordinates (only if `location` exists): `<output_dir>/poi_coordination.csv`
- Intermediate temp files (deleted by default): `<parts_dir>/user_poi_part_*.parquet` and `<parts_dir>/buckets/`
- Log file (next to input file): `aggregator_log.txt`

Final parquet columns (schema you can build on):

- Identifiers: `uuid`, `poi_id`
- Counts/stats: `visit_count`, `total_stay_minutes`, `avg_stay_minutes`, `max_stay_minutes`, `std_stay_minutes`
- Dates: `first_visit_date`, `last_visit_date`, `last_visit_recency_days`
- Ratios: `weekday_visit_ratio`, `weekend_visit_ratio`, `daytime_ratio`, `night_ratio`
- Distributions (stored as JSON strings; use `read_result(...)` to parse): `enter_hour_dist`, `leave_hour_dist`, `visit_weekday_dist`, `visit_day_of_month_dist`
- Tops (stored as JSON strings; use `read_result(...)` to parse): `top_enter_hours`, `top_leave_hours`, `top_weekdays`
- Custom comparison stats (stored as JSON strings; empty `{}` unless `rhythm_specs` provided): `enter_period_stats`, `leave_period_stats`, `weekday_group_stats`, `date_group_stats`

## Remote export: user_grid_time (口径A)

This repo also provides a remote-friendly exporter that **does not require** any `grid_access` (exposure) data.
It only needs a non-sensitive `grid_meta_<city>.json` to map coordinates to `grid_id`.
By default it outputs `grid_uid = "grid_<code>_<col>_<row>"` (set `<code>` via `--grid_uid_code` or a `code/adcode` field in `grid_meta_<city>.json`) to avoid cross-city id conflicts when merging.
In nationwide runs, this usually means **one `grid_meta_<city>.json` per city** (because origin/extent differ by city), then exporting each city separately and merging by `grid_uid`.

Install with optional deps:

```bash
pip install -e ".[export]"
```

Run:

```bash
python -m poi_visit_aggregator.export_user_grid_time ^
  --city shenzhen ^
  --staypoints "D:\\data\\staypoints_*.csv" ^
  --poi_meta "D:\\data\\poi_meta.parquet" ^
  --uuid_table "D:\\data\\uuid_table.parquet" ^
  --grid_meta "D:\\data\\grid_meta_shenzhen.json" ^
  --out_dir "D:\\out" ^
  --grid_uid_code 4403 ^
  --filter_city_code true ^
  --city_code_col c_code ^
  --windows lunch,dinner
```

## Remote export: user_grid_time (strict/fill/filled)

For exposure-weight experiments, this repo also provides a **strict + filled** exporter:

- Interval records (`end>start` and `>=5min`) contribute strict overlap minutes.
- Point records (`end==start` or `<5min`) contribute midpoint-split weights, then are scaled to fill each window's missing time.

Run:

```bash
python -m poi_visit_aggregator.export_user_grid_time_strict_filled ^
  --city shenzhen ^
  --staypoints "D:\\data\\staypoints_*.csv" ^
  --uuid_table "D:\\data\\uuid_table.parquet" ^
  --grid_meta "D:\\data\\grid_meta_shenzhen.json" ^
  --out_dir "D:\\out" ^
  --drop_uuid_not_in_table true ^
  --grid_uid_code 4403 ^
  --filter_city_code true ^
  --city_code_col c_code ^
  --id_mode uuid ^
  --windows lunch,dinner
```

Outputs:
- `<out_dir>/<city>/user_grid_time_strict_filled_<city>.parquet` (columns include `grid_uid`, `window`, `is_weekend`, `tau_strict_min`, `tau_fill_min`, `tau_filled_min`, and optional `grid_id`, `uuid`, `uid64`)
- `<out_dir>/<city>/qa_summary_strict_filled_<city>.csv`

Colab demo notebook: `notebooks/Shenzhen_demo.ipynb`.

## License
This project is licensed under the Apache License 2.0.
