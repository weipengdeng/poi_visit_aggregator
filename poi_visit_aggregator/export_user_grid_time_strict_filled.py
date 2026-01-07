"""
Export user×grid lunch/dinner stay-minute weights with strict/fill/filled scheme.

Definitions (per user-day-window):
- interval: end>start AND duration>=min_interval_minutes => tau_strict_min by overlap.
- point: end==start OR duration<min_interval_minutes (default only source==cell_appearance)
         => tau_point_min by midpoint-split within the window.
- missing = win_len_min - strict_sum_min
- fill_factor = missing / point_sum_min
- tau_fill_min = tau_point_min * fill_factor
- tau_filled_min = tau_strict_min + tau_fill_min

Output parquet is long format (one row per user×grid×window×is_weekend).

Run:
  python -m poi_visit_aggregator.export_user_grid_time_strict_filled --help
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from poi_visit_aggregator.export_user_grid_time import (
    DEFAULT_WINDOWS,
    GridMeta,
    MS_PER_DAY,
    MS_PER_MIN,
    WindowSpec,
    _as_posix,
    _col_as_float64,
    _col_as_int64,
    _col_as_str,
    _compute_window_overlap_minutes,
    _hash_uuid,
    _infer_col,
    _infer_df_col,
    _make_transformer,
    _map_xy_to_grid_id,
    _infer_grid_uid_code_from_grid_meta,
    _normalize_grid_uid_code,
    _parse_hhmm,
    _parse_location_to_lonlat,
    _require_col,
    _sql_grid_uid_expr,
    _write_table_zstd,
)

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _str2bool(v: str) -> bool:
    vv = v.strip().lower()
    if vv in {"1", "true", "t", "yes", "y"}:
        return True
    if vv in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v!r}")


def _load_json_arg(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    p = Path(value)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def _resolve_input_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        matches = glob(pat)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(pat))
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


@contextmanager
def _step(log, name: str):
    t0 = time.perf_counter()
    m0 = _rss_mb()
    log(f"[START] {name} (RAM={m0:,.0f} MB)")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        m1 = _rss_mb()
        log(f"[ END ] {name} (dt={dt:,.1f}s, RAM={m1:,.0f} MB, Δ={m1 - m0:,.0f} MB)")


def _is_weekend_from_epoch_day(day: np.ndarray) -> np.ndarray:
    # 1970-01-01 is Thursday. Monday=0..Sunday=6 => Thursday=3.
    weekday = (day.astype(np.int64, copy=False) + 3) % 7
    return weekday >= 5


def _in_window(mod_ms: np.ndarray, spec: WindowSpec) -> np.ndarray:
    out = np.zeros(len(mod_ms), dtype=bool)
    for seg_start, seg_end in spec.segments:
        out |= (mod_ms >= seg_start) & (mod_ms < seg_end)
    return out


def _mask_in_sorted_uint64(values: np.ndarray, sorted_unique: np.ndarray) -> np.ndarray:
    if len(sorted_unique) == 0:
        return np.zeros(len(values), dtype=bool)
    idx = np.searchsorted(sorted_unique, values)
    ok = idx < len(sorted_unique)
    out = np.zeros(len(values), dtype=bool)
    out[ok] = sorted_unique[idx[ok]] == values[ok]
    return out


def _load_uuid_whitelist_uid64(
    uuid_table: Path,
    *,
    schema_map: dict[str, Any],
    uid64_hash_method: str,
    log,
) -> np.ndarray:
    uuid_schema_map = schema_map.get("uuid_table", {})
    if uuid_table.suffix.lower() == ".parquet":
        t = pq.read_table(_as_posix(uuid_table))
        uuid_c = uuid_schema_map.get("uuid") or _infer_col(t.schema, ["uuid", "user", "uid"])
        uuid_c = _require_col("uuid_table.uuid", uuid_c)
        uuids = t.column(uuid_c).to_pandas().astype(str).to_numpy()
    else:
        df = pd.read_csv(uuid_table)
        uuid_c = uuid_schema_map.get("uuid") or ("uuid" if "uuid" in df.columns else None)
        uuid_c = _require_col("uuid_table.uuid", uuid_c)
        uuids = df[uuid_c].astype(str).to_numpy()

    with _step(log, f"Hash uuid whitelist ({len(uuids):,})"):
        uid64 = _hash_uuid(uuids, uid64_hash_method)
        uid64 = np.unique(uid64)
        uid64.sort()
    log(f"Loaded uuid whitelist uid64: {len(uid64):,}")
    return uid64


def export_user_grid_time_strict_filled(
    *,
    city: str,
    staypoints: list[Path],
    staypoints_format: str,
    uuid_table: Optional[Path],
    grid_meta_path: Path,
    out_dir: Path,
    schema_map: dict[str, Any],
    output_grid_uid: bool,
    output_grid_id: bool,
    grid_uid_code: Optional[str],
    grid_uid_prefix: str,
    grid_uid_order: str,
    filter_city_code: bool,
    city_code_col: Optional[str],
    city_code_value: Optional[str],
    windows: list[str],
    min_interval_minutes: int,
    point_source_filter: bool,
    point_source_value: str,
    drop_uuid_not_in_table: bool,
    timestamps_are_utc: bool,
    tz_offset_hours: int,
    epoch_unit: str,
    coords_already_projected: bool,
    uid64_hash_method: str,
    buckets: int,  # reserved for future bucketed stage-2
    batch_size: int,
    overlap_rounding: str,
    oob_mode: str,
    threads: int,
    memory_limit: str,
    id_mode: str,
    keep_intermediate: bool,
) -> None:
    del buckets  # reserved

    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = city_dir / f"run_log_strict_filled_{city}.txt"

    def log(msg: str) -> None:
        line = f"[{_now_str()}] {msg}"
        print(line)
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    id_mode_n = id_mode.strip().lower()
    if id_mode_n not in {"uuid", "uid64", "both"}:
        raise ValueError("--id_mode must be uuid|uid64|both")
    store_uuid = id_mode_n in {"uuid", "both"}
    store_uid64 = id_mode_n in {"uid64", "both"}

    if duckdb is None:
        raise RuntimeError("This exporter requires DuckDB: `pip install duckdb`")

    tz_offset_ms = int(tz_offset_hours) * 3600 * 1000
    epoch_unit_n = epoch_unit.strip().lower()
    if epoch_unit_n not in {"ms", "s"}:
        raise ValueError("--epoch_unit must be ms or s")
    if drop_uuid_not_in_table and not uuid_table:
        raise ValueError("--uuid_table is required when drop_uuid_not_in_table=true")

    log("=" * 70)
    log(f"Start export_user_grid_time_strict_filled city={city!r}")
    log(f"staypoints files/paths: {len(staypoints)}")
    log(f"id_mode={id_mode_n}, uid64_hash_method={uid64_hash_method}, store_uuid={store_uuid}, store_uid64={store_uid64}")

    grid_meta = GridMeta.from_json(grid_meta_path)
    log(
        f"grid_meta loaded: cell_size_m={grid_meta.cell_size_m}, origin=({grid_meta.origin_x},{grid_meta.origin_y}), n_cols={grid_meta.n_cols}"
    )

    if not output_grid_uid and not output_grid_id:
        raise ValueError("At least one of output_grid_uid/output_grid_id must be true")
    grid_uid_code_from_meta = _infer_grid_uid_code_from_grid_meta(grid_meta_path)
    grid_uid_code_n = _normalize_grid_uid_code(city=city, grid_meta_path=grid_meta_path, grid_uid_code=grid_uid_code)
    grid_uid_prefix_n = str(grid_uid_prefix).strip()
    if not grid_uid_prefix_n:
        raise ValueError("--grid_uid_prefix cannot be empty")
    grid_uid_order_n = str(grid_uid_order).strip().lower()
    if grid_uid_order_n not in {"col_row", "row_col"}:
        raise ValueError("--grid_uid_order must be col_row|row_col")

    windows_norm = [w.strip().lower() for w in windows if w.strip()]
    window_specs: dict[str, WindowSpec] = {}
    for w in windows_norm:
        if w not in DEFAULT_WINDOWS:
            raise ValueError(f"Unknown window: {w!r}. Supported: {sorted(DEFAULT_WINDOWS)}")
        s, e = DEFAULT_WINDOWS[w]
        window_specs[w] = WindowSpec(name=w, start_ms=_parse_hhmm(s), end_ms=_parse_hhmm(e))

    window_meta_df = pd.DataFrame(
        [
            {
                "window": w,
                "start_mod_ms": int(spec.start_ms),
                "end_mod_ms": int(spec.end_ms),
                "win_len_min": float(spec.duration_ms / MS_PER_MIN),
            }
            for w, spec in window_specs.items()
        ]
    )

    whitelist_uid64_sorted = None
    if drop_uuid_not_in_table and uuid_table:
        with _step(log, "Load uuid whitelist"):
            whitelist_uid64_sorted = _load_uuid_whitelist_uid64(
                uuid_table, schema_map=schema_map, uid64_hash_method=uid64_hash_method, log=log
            )

    stay_schema_map = schema_map.get("staypoints", schema_map)

    ds_paths = [_as_posix(p) for p in staypoints]
    staypoints_format_n = staypoints_format.strip().lower()
    if staypoints_format_n not in {"auto", "parquet", "csv"}:
        raise ValueError("--staypoints_format must be auto|parquet|csv")
    if staypoints_format_n == "auto":
        exts = {p.suffix.lower() for p in staypoints if p.suffix}
        if len(exts) == 1 and ".parquet" in exts:
            staypoints_format_n = "parquet"
        elif len(exts) == 1 and ".csv" in exts:
            staypoints_format_n = "csv"
        else:
            raise ValueError("Cannot infer staypoints format; please set --staypoints_format=parquet|csv")

    with _step(log, f"Open dataset ({staypoints_format_n})"):
        if staypoints_format_n == "csv":
            sample = ds.dataset([ds_paths[0]], format="csv")
            dataset = ds.dataset(ds_paths, format="csv", schema=sample.schema)
        else:
            dataset = ds.dataset(ds_paths, format="parquet")
        schema = dataset.schema

    uuid_col = stay_schema_map.get("uuid") or _infer_col(schema, ["uuid", "user", "uid"])
    start_col = stay_schema_map.get("start_time") or _infer_col(schema, ["start_time", "start", "start_ms", "stime"])
    end_col = stay_schema_map.get("end_time") or _infer_col(schema, ["end_time", "end", "end_ms", "etime"])
    uuid_col = _require_col("staypoints.uuid", uuid_col)
    start_col = _require_col("staypoints.start_time", start_col)
    end_col = _require_col("staypoints.end_time", end_col)

    source_col = stay_schema_map.get("source") or _infer_col(schema, ["source", "src"])

    lon_col = stay_schema_map.get("lon") or _infer_col(schema, ["lon", "lng", "longitude"])
    lat_col = stay_schema_map.get("lat") or _infer_col(schema, ["lat", "latitude"])
    x_col = stay_schema_map.get("x") or _infer_col(schema, ["x", "coord_x", "easting"])
    y_col = stay_schema_map.get("y") or _infer_col(schema, ["y", "coord_y", "northing"])
    loc_col = stay_schema_map.get("location") or _infer_col(schema, ["location", "loc"])

    coord_mode = None
    if x_col and y_col:
        coord_mode = "xy"
    elif lon_col and lat_col:
        coord_mode = "lonlat"
    elif loc_col:
        coord_mode = "location"
    else:
        raise ValueError("Cannot infer coordinate columns (need lon/lat, x/y, or location). Please provide --schema_map.")

    transformer = None
    if coord_mode in {"lonlat", "location"}:
        if coords_already_projected:
            raise ValueError("coords_already_projected=true requires x/y columns (already in grid_crs).")
        transformer = _make_transformer(grid_meta)
        if transformer is None:
            raise RuntimeError("Coordinate projection requires `pip install pyproj`")

    scan_cols = [uuid_col, start_col, end_col]
    if source_col:
        scan_cols.append(source_col)
    if coord_mode == "xy":
        scan_cols += [x_col, y_col]
    elif coord_mode == "lonlat":
        scan_cols += [lon_col, lat_col]
    else:
        scan_cols.append(loc_col)
    scan_cols = [c for c in scan_cols if c is not None]
    log(f"Staypoints scan columns: {scan_cols}")

    tmp_dir = city_dir / "_tmp_strict_filled"
    interval_parts_dir = tmp_dir / "interval_parts"
    point_parts_dir = tmp_dir / "point_parts"
    interval_parts_dir.mkdir(parents=True, exist_ok=True)
    point_parts_dir.mkdir(parents=True, exist_ok=True)
    if not keep_intermediate:
        with _step(log, "Clean intermediate parts"):
            for p in list(interval_parts_dir.glob("part_*.parquet")) + list(point_parts_dir.glob("part_*.parquet")):
                p.unlink()

    qa: dict[str, Any] = {
        "city": city,
        "id_mode": id_mode_n,
        "output_grid_uid": bool(output_grid_uid),
        "output_grid_id": bool(output_grid_id),
        "grid_uid_prefix": grid_uid_prefix_n,
        "grid_uid_code": grid_uid_code_n,
        "grid_uid_order": grid_uid_order_n,
        "filter_city_code": bool(filter_city_code),
        "city_code_col": city_code_col,
        "city_code_value": city_code_value,
        "groupby_id": "uid64",
        "generated_uid64": True,
        "uid64_hash_method": uid64_hash_method,
        "drop_uuid_not_in_table": bool(drop_uuid_not_in_table),
        "point_source_filter": bool(point_source_filter),
        "point_source_value": point_source_value,
        "min_interval_minutes": int(min_interval_minutes),
        "windows": ",".join(windows_norm),
        "input_rows": 0,
        "kept_rows": 0,
        "invalid_time_rows": 0,
        "filtered_uuid_not_in_table": 0,
        "filtered_oob_grid": 0,
        "interval_rows": 0,
        "point_rows": 0,
        "point_rows_used": 0,
        "point_rows_outside_windows": 0,
        "interval_cross_day_rows": 0,
        "interval_multi_day_rows_dropped": 0,
    }

    point_source_value_l = point_source_value.strip().lower()

    filter_expr = None
    if filter_city_code:
        city_col = (
            (stay_schema_map.get("city_code") or stay_schema_map.get("c_code"))
            if isinstance(stay_schema_map, dict)
            else None
        )
        if city_code_col:
            city_col = city_code_col
        if not city_col:
            city_col = _infer_col(schema, ["c_code", "city_code", "adcode", "city", "city_id", "cityid"])
        city_col = _require_col("staypoints.city_code", city_col)

        if city_col not in scan_cols:
            scan_cols.append(city_col)

        val = city_code_value
        if val is None or str(val).strip() == "":
            if grid_uid_code is not None and str(grid_uid_code).strip() != "":
                val = str(grid_uid_code).strip()
            elif grid_uid_code_from_meta is not None and str(grid_uid_code_from_meta).strip() != "":
                val = str(grid_uid_code_from_meta).strip()
            else:
                raise ValueError(
                    "filter_city_code=true requires --city_code_value or --grid_uid_code, or `code/adcode` in grid_meta json."
                )

        qa["city_code_col"] = city_col
        qa["city_code_value"] = str(val)
        filter_expr = ds.field(city_col).cast(pa.string()) == str(val)

    scanner = dataset.scanner(columns=scan_cols, filter=filter_expr, batch_size=batch_size, use_threads=True)

    part_id = 0
    batch_i = 0

    with _step(log, "Stage-1 scan + write parts"):
        for batch in scanner.to_batches():
            batch_i += 1
            n = batch.num_rows
            qa["input_rows"] += n
            if n == 0:
                continue

            uuid_vals = _col_as_str(batch.column(uuid_col))
            start_ms = _col_as_int64(batch.column(start_col))
            end_ms = _col_as_int64(batch.column(end_col))

            if epoch_unit_n == "s":
                start_ms = start_ms * 1000
                end_ms = end_ms * 1000
            if timestamps_are_utc:
                start_ms = start_ms + tz_offset_ms
                end_ms = end_ms + tz_offset_ms

            valid_time = (start_ms >= 0) & (end_ms >= start_ms)
            qa["invalid_time_rows"] += int((~valid_time).sum())
            mask = valid_time & (uuid_vals != "")
            if not mask.any():
                continue

            idx = np.nonzero(mask)[0]
            uuid_keep = uuid_vals[idx]
            uid64_keep = _hash_uuid(uuid_keep, uid64_hash_method)

            if whitelist_uid64_sorted is not None:
                before = len(idx)
                in_tbl = _mask_in_sorted_uint64(uid64_keep, whitelist_uid64_sorted)
                qa["filtered_uuid_not_in_table"] += before - int(in_tbl.sum())
                idx = idx[in_tbl]
                uuid_keep = uuid_keep[in_tbl]
                uid64_keep = uid64_keep[in_tbl]
                if len(idx) == 0:
                    continue

            if coord_mode == "xy":
                x = _col_as_float64(batch.column(x_col))[idx]  # type: ignore[arg-type]
                y = _col_as_float64(batch.column(y_col))[idx]  # type: ignore[arg-type]
            elif coord_mode == "lonlat":
                lon = _col_as_float64(batch.column(lon_col))[idx]  # type: ignore[arg-type]
                lat = _col_as_float64(batch.column(lat_col))[idx]  # type: ignore[arg-type]
                x, y = transformer.transform(lon, lat)  # type: ignore[union-attr]
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)
            else:
                lon, lat = _parse_location_to_lonlat(batch.column(loc_col))  # type: ignore[arg-type]
                lon = lon[idx]
                lat = lat[idx]
                x, y = transformer.transform(lon, lat)  # type: ignore[union-attr]
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)

            grid_id_raw, oob_mask = _map_xy_to_grid_id(x, y, grid_meta, oob_mode=oob_mode)
            if oob_mode.lower() == "drop":
                before = len(grid_id_raw)
                keep = ~oob_mask
                qa["filtered_oob_grid"] += before - int(keep.sum())
                grid_id_raw = grid_id_raw[keep]
                idx = idx[keep]
                uuid_keep = uuid_keep[keep]
                uid64_keep = uid64_keep[keep]
                if len(idx) == 0:
                    continue
            else:
                qa["filtered_oob_grid"] += int(oob_mask.sum())

            start_ms_f = start_ms[idx]
            end_ms_f = end_ms[idx]
            qa["kept_rows"] += len(idx)

            grid_id_i32 = grid_id_raw.astype(np.int32, copy=False)
            dur_ms = (end_ms_f - start_ms_f).astype(np.int64, copy=False)
            is_interval = (dur_ms > 0) & (dur_ms >= int(min_interval_minutes) * MS_PER_MIN)
            is_point = ~is_interval
            qa["interval_rows"] += int(is_interval.sum())
            qa["point_rows"] += int(is_point.sum())

            part_id += 1

            interval_rows_out = 0
            point_rows_out = 0

            # ---- interval: compute tau_strict_min ----
            if is_interval.any():
                int_idx = np.nonzero(is_interval)[0]
                int_start = start_ms_f[int_idx]
                int_end = end_ms_f[int_idx]
                day0 = (int_start // MS_PER_DAY).astype(np.int64)
                day1 = ((int_end - 1) // MS_PER_DAY).astype(np.int64)
                same_day = day1 == day0
                cross_1 = day1 == (day0 + 1)
                multi_day = day1 > (day0 + 1)
                qa["interval_cross_day_rows"] += int(cross_1.sum())
                qa["interval_multi_day_rows_dropped"] += int(multi_day.sum())

                seg_uid64 = []
                seg_uuid = []
                seg_grid = []
                seg_day = []
                seg_start = []
                seg_end = []

                if same_day.any():
                    j = int_idx[same_day]
                    seg_uid64.append(uid64_keep[j])
                    seg_uuid.append(uuid_keep[j])
                    seg_grid.append(grid_id_i32[j])
                    seg_day.append(day0[same_day].astype(np.int32))
                    seg_start.append(int_start[same_day])
                    seg_end.append(int_end[same_day])

                if cross_1.any():
                    j = int_idx[cross_1]
                    d0 = day0[cross_1]
                    d1 = day1[cross_1]
                    d0_end = (d0 + 1) * MS_PER_DAY
                    d1_start = d1 * MS_PER_DAY
                    seg_uid64.append(uid64_keep[j])
                    seg_uuid.append(uuid_keep[j])
                    seg_grid.append(grid_id_i32[j])
                    seg_day.append(d0.astype(np.int32))
                    seg_start.append(int_start[cross_1])
                    seg_end.append(d0_end.astype(np.int64))
                    seg_uid64.append(uid64_keep[j])
                    seg_uuid.append(uuid_keep[j])
                    seg_grid.append(grid_id_i32[j])
                    seg_day.append(d1.astype(np.int32))
                    seg_start.append(d1_start.astype(np.int64))
                    seg_end.append(int_end[cross_1])

                if seg_uid64:
                    s_uid64 = np.concatenate(seg_uid64).astype(np.uint64, copy=False)
                    s_uuid = np.concatenate(seg_uuid).astype(object, copy=False)
                    s_grid = np.concatenate(seg_grid).astype(np.int32, copy=False)
                    s_day = np.concatenate(seg_day).astype(np.int32, copy=False)
                    s_start = np.concatenate(seg_start).astype(np.int64, copy=False)
                    s_end = np.concatenate(seg_end).astype(np.int64, copy=False)

                    out_uid64 = []
                    out_uuid = []
                    out_grid = []
                    out_day = []
                    out_wkend = []
                    out_win = []
                    out_tau = []

                    for wname, spec in window_specs.items():
                        tau = _compute_window_overlap_minutes(s_start, s_end, spec, rounding=overlap_rounding)
                        ok = tau > 0
                        if not ok.any():
                            continue
                        out_uid64.append(s_uid64[ok])
                        out_uuid.append(s_uuid[ok])
                        out_grid.append(s_grid[ok])
                        out_day.append(s_day[ok])
                        out_wkend.append(_is_weekend_from_epoch_day(s_day[ok].astype(np.int64)).astype(bool))
                        out_win.append(np.full(int(ok.sum()), wname, dtype=object))
                        out_tau.append(tau[ok].astype(np.int32, copy=False))

                    if out_uid64:
                        t_uid64 = np.concatenate(out_uid64)
                        t = pa.table(
                            {
                                "uid64": pa.array(t_uid64, type=pa.uint64()),
                                "grid_id": pa.array(np.concatenate(out_grid), type=pa.int32()),
                                "day": pa.array(np.concatenate(out_day), type=pa.int32()),
                                "is_weekend": pa.array(np.concatenate(out_wkend), type=pa.bool_()),
                                "window": pa.array(np.concatenate(out_win), type=pa.string()),
                                "tau_strict_min": pa.array(np.concatenate(out_tau), type=pa.int32()),
                            }
                        )
                        if store_uuid:
                            t = t.append_column("uuid", pa.array(np.concatenate(out_uuid), type=pa.string()))
                        _write_table_zstd(t, interval_parts_dir / f"part_{part_id:06d}.parquet")
                        interval_rows_out = t.num_rows

            # ---- point: write t_ms for midpoint split ----
            if is_point.any():
                p_idx = np.nonzero(is_point)[0]
                p_uid64 = uid64_keep[p_idx]
                p_uuid = uuid_keep[p_idx]
                p_grid = grid_id_i32[p_idx]
                p_start = start_ms_f[p_idx]
                p_end = end_ms_f[p_idx]
                p_mid = ((p_start.astype(np.int64) + p_end.astype(np.int64)) // 2).astype(np.int64)

                use_mask = np.ones(len(p_idx), dtype=bool)
                if point_source_filter:
                    if source_col and source_col in batch.schema.names:
                        src_vals = _col_as_str(batch.column(source_col))[idx][p_idx]  # type: ignore[arg-type]
                        src_vals = np.asarray(src_vals, dtype=str)
                        use_mask &= np.char.lower(src_vals) == point_source_value_l
                    else:
                        log("WARN: point_source_filter=true but `source` column not found; using all point rows.")
                qa["point_rows_used"] += int(use_mask.sum())
                if use_mask.any():
                    p_uid64 = p_uid64[use_mask]
                    p_uuid = p_uuid[use_mask]
                    p_grid = p_grid[use_mask]
                    p_mid = p_mid[use_mask]

                    p_day = (p_mid // MS_PER_DAY).astype(np.int64)
                    p_mod = (p_mid - p_day * MS_PER_DAY).astype(np.int64)
                    p_wkend = _is_weekend_from_epoch_day(p_day).astype(bool)

                    out_uid64 = []
                    out_uuid = []
                    out_grid = []
                    out_day = []
                    out_wkend = []
                    out_win = []
                    out_t = []

                    in_any = np.zeros(len(p_mid), dtype=bool)
                    for wname, spec in window_specs.items():
                        ok = _in_window(p_mod, spec)
                        if not ok.any():
                            continue
                        in_any |= ok
                        out_uid64.append(p_uid64[ok])
                        out_uuid.append(p_uuid[ok])
                        out_grid.append(p_grid[ok])
                        out_day.append(p_day[ok].astype(np.int32))
                        out_wkend.append(p_wkend[ok])
                        out_win.append(np.full(int(ok.sum()), wname, dtype=object))
                        out_t.append(p_mid[ok].astype(np.int64))

                    qa["point_rows_outside_windows"] += int((~in_any).sum())
                    if out_uid64:
                        t = pa.table(
                            {
                                "uid64": pa.array(np.concatenate(out_uid64), type=pa.uint64()),
                                "grid_id": pa.array(np.concatenate(out_grid), type=pa.int32()),
                                "day": pa.array(np.concatenate(out_day), type=pa.int32()),
                                "is_weekend": pa.array(np.concatenate(out_wkend), type=pa.bool_()),
                                "window": pa.array(np.concatenate(out_win), type=pa.string()),
                                "t_ms": pa.array(np.concatenate(out_t), type=pa.int64()),
                            }
                        )
                        if store_uuid:
                            t = t.append_column("uuid", pa.array(np.concatenate(out_uuid), type=pa.string()))
                        _write_table_zstd(t, point_parts_dir / f"part_{part_id:06d}.parquet")
                        point_rows_out = t.num_rows

            if batch_i % 50 == 0:
                log(
                    f"batches={batch_i:,}, input_rows={qa['input_rows']:,}, kept_rows={qa['kept_rows']:,}, "
                    f"interval_out={interval_rows_out:,}, point_out={point_rows_out:,}"
                )

    out_path = city_dir / f"user_grid_time_strict_filled_{city}.parquet"
    qa_path = city_dir / f"qa_summary_strict_filled_{city}.csv"

    # Stage-2 + output is implemented below (added in the next patch).
    _run_stage2_and_write(
        city=city,
        city_dir=city_dir,
        tmp_dir=tmp_dir,
        interval_parts_dir=interval_parts_dir,
        point_parts_dir=point_parts_dir,
        out_path=out_path,
        qa=qa,
        qa_path=qa_path,
        window_meta_df=window_meta_df,
        output_grid_uid=output_grid_uid,
        output_grid_id=output_grid_id,
        grid_n_cols=grid_meta.n_cols,
        grid_uid_prefix=grid_uid_prefix_n,
        grid_uid_code=grid_uid_code_n,
        grid_uid_order=grid_uid_order_n,
        store_uuid=store_uuid,
        store_uid64=store_uid64,
        threads=threads,
        memory_limit=memory_limit,
        keep_intermediate=keep_intermediate,
        log=log,
    )


def _run_stage2_and_write(
    *,
    city: str,
    city_dir: Path,
    tmp_dir: Path,
    interval_parts_dir: Path,
    point_parts_dir: Path,
    out_path: Path,
    qa: dict[str, Any],
    qa_path: Path,
    window_meta_df: pd.DataFrame,
    output_grid_uid: bool,
    output_grid_id: bool,
    grid_n_cols: int,
    grid_uid_prefix: str,
    grid_uid_code: str,
    grid_uid_order: str,
    store_uuid: bool,
    store_uid64: bool,
    threads: int,
    memory_limit: str,
    keep_intermediate: bool,
    log,
) -> None:
    interval_files = list(interval_parts_dir.glob("part_*.parquet"))
    point_files = list(point_parts_dir.glob("part_*.parquet"))
    has_interval = bool(interval_files)
    has_point = bool(point_files)

    if not (has_interval or has_point):
        log("No interval/point parts written; writing empty output parquet.")
        cols: dict[str, pa.Array] = {
            "window": pa.array([], type=pa.string()),
            "is_weekend": pa.array([], type=pa.bool_()),
            "tau_strict_min": pa.array([], type=pa.int32()),
            "tau_fill_min": pa.array([], type=pa.float64()),
            "tau_filled_min": pa.array([], type=pa.float64()),
        }
        if output_grid_uid:
            cols["grid_uid"] = pa.array([], type=pa.string())
        if output_grid_id:
            cols["grid_id"] = pa.array([], type=pa.int32())
        if store_uuid:
            cols["uuid"] = pa.array([], type=pa.string())
        if store_uid64:
            cols["uid64"] = pa.array([], type=pa.uint64())
        _write_table_zstd(pa.table(cols), out_path)
        qa["output_rows"] = 0
        qa["output_parquet_bytes"] = int(out_path.stat().st_size)
        qa["output_parquet_mb"] = round(qa["output_parquet_bytes"] / 1024 / 1024, 3)
        pd.DataFrame([qa]).to_csv(qa_path, index=False, encoding="utf-8-sig")
        log(f"Wrote: {out_path}")
        log(f"Wrote: {qa_path}")
        log("Done.")
        log("=" * 70)
        return

    with _step(log, "Stage-2 DuckDB (midpoint split + strict/fill aggregation)"):
        temp_dir = tmp_dir / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(database=":memory:")
        con.execute(f"PRAGMA threads={int(threads)}")
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA temp_directory='{_as_posix(temp_dir)}'")

        con.register("window_meta_df", window_meta_df)
        con.execute("CREATE TABLE window_meta AS SELECT * FROM window_meta_df")

        if has_interval:
            interval_glob = _as_posix(interval_parts_dir / "part_*.parquet")
            con.execute(f"CREATE VIEW interval_raw AS SELECT * FROM read_parquet('{interval_glob}')")
        else:
            con.execute(
                "CREATE VIEW interval_raw AS SELECT 0::UBIGINT AS uid64, 0::INTEGER AS grid_id, 0::INTEGER AS day, false::BOOLEAN AS is_weekend, ''::VARCHAR AS window, 0::INTEGER AS tau_strict_min LIMIT 0"
            )

        if has_point:
            point_glob = _as_posix(point_parts_dir / "part_*.parquet")
            con.execute(f"CREATE VIEW point_raw AS SELECT * FROM read_parquet('{point_glob}')")
        else:
            con.execute(
                "CREATE VIEW point_raw AS SELECT 0::UBIGINT AS uid64, 0::INTEGER AS grid_id, 0::INTEGER AS day, false::BOOLEAN AS is_weekend, ''::VARCHAR AS window, 0::BIGINT AS t_ms LIMIT 0"
            )

        if store_uuid:
            con.execute(
                """
                CREATE TABLE uuid_map AS
                SELECT uid64, MIN(uuid) AS uuid, COUNT(DISTINCT uuid) AS n_uuid
                FROM (
                  SELECT uid64, uuid FROM interval_raw
                  UNION ALL
                  SELECT uid64, uuid FROM point_raw
                ) u
                GROUP BY uid64
                """
            )
            qa["uid64_uuid_collision_rows"] = int(
                con.execute("SELECT COALESCE(SUM(CASE WHEN n_uuid>1 THEN 1 ELSE 0 END), 0) FROM uuid_map").fetchone()[0]
            )
        else:
            qa["uid64_uuid_collision_rows"] = 0

        con.execute(
            """
            CREATE VIEW interval_agg AS
            SELECT uid64, day, is_weekend, window, grid_id, CAST(SUM(tau_strict_min) AS DOUBLE) AS tau_strict_min
            FROM interval_raw
            GROUP BY uid64, day, is_weekend, window, grid_id
            """
        )

        con.execute(
            f"""
            CREATE VIEW point_ordered AS
            SELECT
              p.uid64,
              p.day,
              p.is_weekend,
              p.window,
              p.grid_id,
              p.t_ms,
              LAG(p.t_ms) OVER (PARTITION BY p.uid64, p.day, p.window ORDER BY p.t_ms) AS prev_t,
              LEAD(p.t_ms) OVER (PARTITION BY p.uid64, p.day, p.window ORDER BY p.t_ms) AS next_t,
              m.start_mod_ms,
              m.end_mod_ms,
              m.win_len_min
            FROM point_raw p
            JOIN window_meta m USING (window)
            """
        )

        con.execute(
            f"""
            CREATE VIEW point_agg AS
            SELECT
              uid64,
              day,
              is_weekend,
              window,
              grid_id,
              SUM(
                GREATEST(
                  0,
                  LEAST(day*{MS_PER_DAY} + end_mod_ms, COALESCE((t_ms + next_t)/2, day*{MS_PER_DAY} + end_mod_ms))
                  - GREATEST(day*{MS_PER_DAY} + start_mod_ms, COALESCE((prev_t + t_ms)/2, day*{MS_PER_DAY} + start_mod_ms))
                )::DOUBLE / {MS_PER_MIN}
              ) AS tau_point_min
            FROM point_ordered
            GROUP BY uid64, day, is_weekend, window, grid_id
            """
        )

        con.execute(
            """
            CREATE VIEW base AS
            SELECT
              uid64,
              day,
              is_weekend,
              window,
              grid_id,
              SUM(tau_strict_min) AS tau_strict_min,
              SUM(tau_point_min) AS tau_point_min
            FROM (
              SELECT uid64, day, is_weekend, window, grid_id, tau_strict_min, 0.0 AS tau_point_min FROM interval_agg
              UNION ALL
              SELECT uid64, day, is_weekend, window, grid_id, 0.0 AS tau_strict_min, tau_point_min FROM point_agg
            ) u
            GROUP BY uid64, day, is_weekend, window, grid_id
            """
        )

        con.execute(
            """
            CREATE VIEW factors AS
            SELECT
              b.uid64,
              b.day,
              b.is_weekend,
              b.window,
              SUM(b.tau_strict_min) AS strict_sum_min,
              SUM(b.tau_point_min) AS point_sum_min,
              m.win_len_min,
              GREATEST(0.0, m.win_len_min - SUM(b.tau_strict_min)) AS missing_min,
              CASE
                WHEN SUM(b.tau_point_min) > 0 THEN GREATEST(0.0, m.win_len_min - SUM(b.tau_strict_min)) / SUM(b.tau_point_min)
                ELSE 0.0
              END AS fill_factor
            FROM base b
            JOIN window_meta m USING (window)
            GROUP BY b.uid64, b.day, b.is_weekend, b.window, m.win_len_min
            """
        )

        id_select_inner = ""
        id_join = ""
        group_extra = ""
        id_select_outer = ""
        if store_uuid and store_uid64:
            id_select_inner = ", b.uid64 AS uid64, um.uuid AS uuid"
            id_select_outer = ",\n                uid64,\n                uuid"
            id_join = "LEFT JOIN uuid_map um USING (uid64)"
            group_extra = ", b.uid64, um.uuid"
        elif store_uuid:
            id_select_inner = ", um.uuid AS uuid"
            id_select_outer = ",\n                uuid"
            id_join = "LEFT JOIN uuid_map um USING (uid64)"
            group_extra = ", um.uuid"
        elif store_uid64:
            id_select_inner = ", b.uid64 AS uid64"
            id_select_outer = ",\n                uid64"
            group_extra = ", b.uid64"

        outer_cols: list[str] = []
        if output_grid_uid:
            grid_uid_expr = _sql_grid_uid_expr(
                "agg.grid_id",
                n_cols=grid_n_cols,
                prefix=grid_uid_prefix,
                code=grid_uid_code,
                order=grid_uid_order,
            )
            outer_cols.append(f"{grid_uid_expr} AS grid_uid")
        if output_grid_id:
            outer_cols.append("agg.grid_id AS grid_id")
        outer_cols += [
            "window",
            "is_weekend",
        ]
        outer_select = ",\n                ".join(outer_cols)

        con.execute(
            f"""
            COPY (
              SELECT
                {outer_select}{id_select_outer},
                tau_strict_min,
                tau_fill_min,
                tau_filled_min
              FROM (
                SELECT
                  b.grid_id,
                  b.window,
                  b.is_weekend
                  {id_select_inner},
                  CAST(SUM(b.tau_strict_min) AS INTEGER) AS tau_strict_min,
                  SUM(b.tau_point_min * f.fill_factor) AS tau_fill_min,
                  SUM(b.tau_strict_min + b.tau_point_min * f.fill_factor) AS tau_filled_min
                FROM base b
                JOIN factors f USING (uid64, day, is_weekend, window)
                {id_join}
                WHERE b.grid_id IS NOT NULL AND b.grid_id >= 0
                GROUP BY b.grid_id, b.window, b.is_weekend{group_extra}
              ) agg
            ) TO '{_as_posix(out_path)}' (FORMAT PARQUET, CODEC ZSTD);
            """
        )

        qa["output_rows"] = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{_as_posix(out_path)}')").fetchone()[0])
        qa["n_users_uid64"] = int(con.execute("SELECT COUNT(DISTINCT uid64) FROM factors").fetchone()[0])
        qa["n_user_day_windows"] = int(con.execute("SELECT COUNT(*) FROM factors").fetchone()[0])
        qa["n_user_day_windows_no_points"] = int(con.execute("SELECT COUNT(*) FROM factors WHERE point_sum_min=0").fetchone()[0])
        qa["n_user_day_windows_no_strict"] = int(con.execute("SELECT COUNT(*) FROM factors WHERE strict_sum_min=0").fetchone()[0])
        qa["output_parquet_bytes"] = int(out_path.stat().st_size)
        qa["output_parquet_mb"] = round(qa["output_parquet_bytes"] / 1024 / 1024, 3)

        win_stats = con.execute(
            """
            SELECT
              window,
              COUNT(*) AS n_user_day_windows,
              SUM(strict_sum_min) AS strict_sum_total_min,
              SUM(point_sum_min) AS point_sum_total_min,
              SUM(missing_min) AS missing_total_min,
              AVG(fill_factor) AS fill_factor_mean,
              quantile_cont(fill_factor, 0.5) AS fill_factor_p50,
              quantile_cont(fill_factor, 0.9) AS fill_factor_p90,
              quantile_cont(fill_factor, 0.99) AS fill_factor_p99,
              quantile_cont(missing_min, 0.5) AS missing_p50,
              quantile_cont(missing_min, 0.9) AS missing_p90,
              quantile_cont(missing_min, 0.99) AS missing_p99
            FROM factors
            GROUP BY window
            """
        ).fetchdf()
        for _, r in win_stats.iterrows():
            w = str(r["window"])
            qa[f"{w}__n_user_day_windows"] = int(r["n_user_day_windows"])
            qa[f"{w}__strict_sum_total_min"] = float(r["strict_sum_total_min"])
            qa[f"{w}__point_sum_total_min"] = float(r["point_sum_total_min"])
            qa[f"{w}__missing_total_min"] = float(r["missing_total_min"])
            qa[f"{w}__fill_factor_mean"] = float(r["fill_factor_mean"])
            qa[f"{w}__fill_factor_p50"] = float(r["fill_factor_p50"])
            qa[f"{w}__fill_factor_p90"] = float(r["fill_factor_p90"])
            qa[f"{w}__fill_factor_p99"] = float(r["fill_factor_p99"])
            qa[f"{w}__missing_p50"] = float(r["missing_p50"])
            qa[f"{w}__missing_p90"] = float(r["missing_p90"])
            qa[f"{w}__missing_p99"] = float(r["missing_p99"])

        # point_max_gap_min: max gap between point timestamps (including window edges), per user-day-window.
        con.execute(
            f"""
            CREATE VIEW point_gap AS
            WITH diffs AS (
              SELECT
                uid64,
                day,
                window,
                MIN(t_ms) AS first_t,
                MAX(t_ms) AS last_t,
                MAX(t_ms - prev_t) AS max_diff
              FROM (
                SELECT
                  uid64,
                  day,
                  window,
                  t_ms,
                  LAG(t_ms) OVER (PARTITION BY uid64, day, window ORDER BY t_ms) AS prev_t
                FROM point_raw
              ) o
              GROUP BY uid64, day, window
            )
            SELECT
              d.uid64,
              d.day,
              d.window,
              GREATEST(
                COALESCE(d.max_diff, 0),
                COALESCE(d.first_t - (d.day*{MS_PER_DAY} + m.start_mod_ms), 0),
                COALESCE((d.day*{MS_PER_DAY} + m.end_mod_ms) - d.last_t, 0)
              )::DOUBLE / {MS_PER_MIN} AS point_max_gap_min
            FROM diffs d
            JOIN window_meta m USING (window)
            """
        )
        gap_stats = con.execute(
            """
            SELECT
              f.window,
              quantile_cont(COALESCE(g.point_max_gap_min, f.win_len_min), 0.5) AS point_max_gap_p50,
              quantile_cont(COALESCE(g.point_max_gap_min, f.win_len_min), 0.9) AS point_max_gap_p90,
              quantile_cont(COALESCE(g.point_max_gap_min, f.win_len_min), 0.99) AS point_max_gap_p99
            FROM factors f
            LEFT JOIN point_gap g USING (uid64, day, window)
            GROUP BY f.window
            """
        ).fetchdf()
        for _, r in gap_stats.iterrows():
            w = str(r["window"])
            qa[f"{w}__point_max_gap_p50"] = float(r["point_max_gap_p50"])
            qa[f"{w}__point_max_gap_p90"] = float(r["point_max_gap_p90"])
            qa[f"{w}__point_max_gap_p99"] = float(r["point_max_gap_p99"])

    pd.DataFrame([qa]).to_csv(qa_path, index=False, encoding="utf-8-sig")
    log(f"Wrote: {out_path}")
    log(f"Wrote: {qa_path}")

    if not keep_intermediate:
        with _step(log, "Clean intermediate parts"):
            for p in list(interval_parts_dir.glob("part_*.parquet")) + list(point_parts_dir.glob("part_*.parquet")):
                p.unlink()

    log("Done.")
    log("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export user×grid lunch/dinner weights with strict/fill/filled scheme")
    p.add_argument("--city", required=True)
    p.add_argument("--staypoints", required=True, nargs="+", help="Parquet/CSV path(s) or glob(s)")
    p.add_argument("--staypoints_format", default="auto", choices=["auto", "parquet", "csv"])
    p.add_argument("--uuid_table", default=None, help="Optional uuid table (.parquet or .csv) for whitelist filtering")
    p.add_argument("--grid_meta", required=True, help="grid_meta_<city>.json")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--schema_map", default=None, help="JSON string or path to JSON file")
    p.add_argument("--output_grid_uid", type=_str2bool, default=True, help="If true, output string grid_uid")
    p.add_argument("--output_grid_id", type=_str2bool, default=False, help="If true, also output numeric grid_id (=row*n_cols+col)")
    p.add_argument("--grid_uid_code", default=None, help="City code used in grid_uid (e.g. 4403); defaults to grid_meta.code/adcode or --city")
    p.add_argument("--grid_uid_prefix", default="grid", help="Prefix used in grid_uid (default: grid)")
    p.add_argument("--grid_uid_order", default="col_row", choices=["col_row", "row_col"], help="Order of col/row suffix in grid_uid")
    p.add_argument("--filter_city_code", type=_str2bool, default=False, help="If true, pre-filter staypoints by city code column (e.g. c_code)")
    p.add_argument("--city_code_col", default=None, help="Staypoints city code column name (default: infer, e.g. c_code)")
    p.add_argument("--city_code_value", default=None, help="City code value to filter (default: --grid_uid_code or grid_meta.code/adcode)")
    p.add_argument("--windows", default="lunch,dinner", help="Comma-separated: lunch,dinner")
    p.add_argument("--min_interval_minutes", type=int, default=5)
    p.add_argument("--point_source_filter", type=_str2bool, default=True)
    p.add_argument("--point_source_value", default="cell_appearance")
    p.add_argument("--drop_uuid_not_in_table", type=_str2bool, default=False)

    p.add_argument("--timestamps_are_utc", type=_str2bool, default=True)
    p.add_argument("--tz_offset_hours", type=int, default=8)
    p.add_argument("--epoch_unit", default="ms", choices=["ms", "s"])

    p.add_argument("--coords_already_projected", type=_str2bool, default=False)
    p.add_argument(
        "--uid64_hash_method",
        default="sha256_64",
        choices=["sha256_64", "md5_64", "pandas_64", "xxh64"],
        help="Stable 64-bit uid hash method (xxh64 is fastest if available)",
    )
    p.add_argument("--id_mode", default="uuid", choices=["uuid", "uid64", "both"])

    p.add_argument("--buckets", type=int, default=256, help="(reserved) future stage-2 bucketing")
    p.add_argument("--batch_size", type=int, default=1_000_000)
    p.add_argument("--overlap_rounding", default="floor", choices=["floor", "round", "ceil"])
    p.add_argument("--oob_mode", default="drop", choices=["drop", "null", "keep"])

    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory_limit", default="16GB")
    p.add_argument("--keep_intermediate", type=_str2bool, default=False)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    schema_map = _load_json_arg(args.schema_map)

    staypoints = _resolve_input_paths(args.staypoints)
    if not staypoints:
        raise SystemExit("No staypoints files found.")

    uuid_table = Path(args.uuid_table).resolve() if args.uuid_table else None
    grid_meta_path = Path(args.grid_meta).resolve()
    out_dir = Path(args.out_dir).resolve()

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]

    export_user_grid_time_strict_filled(
        city=args.city,
        staypoints=staypoints,
        staypoints_format=args.staypoints_format,
        uuid_table=uuid_table,
        grid_meta_path=grid_meta_path,
        out_dir=out_dir,
        schema_map=schema_map,
        output_grid_uid=args.output_grid_uid,
        output_grid_id=args.output_grid_id,
        grid_uid_code=args.grid_uid_code,
        grid_uid_prefix=args.grid_uid_prefix,
        grid_uid_order=args.grid_uid_order,
        filter_city_code=args.filter_city_code,
        city_code_col=args.city_code_col,
        city_code_value=args.city_code_value,
        windows=windows,
        min_interval_minutes=args.min_interval_minutes,
        point_source_filter=args.point_source_filter,
        point_source_value=args.point_source_value,
        drop_uuid_not_in_table=args.drop_uuid_not_in_table,
        timestamps_are_utc=args.timestamps_are_utc,
        tz_offset_hours=args.tz_offset_hours,
        epoch_unit=args.epoch_unit,
        coords_already_projected=args.coords_already_projected,
        uid64_hash_method=args.uid64_hash_method,
        buckets=args.buckets,
        batch_size=args.batch_size,
        overlap_rounding=args.overlap_rounding,
        oob_mode=args.oob_mode,
        threads=args.threads,
        memory_limit=args.memory_limit,
        id_mode=args.id_mode,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
