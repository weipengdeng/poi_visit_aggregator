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

## License
This project is licensed under the Apache License 2.0.
