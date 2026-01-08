"""
Remote export: uuid_hash × grid_uid stay-minute weights (口径A).

This module is designed to run on a partner's Windows machine using only:
  - staypoints table (huge)
  - poi_id category table (medium, optional for filtering)
  - uuid table (medium, optional for covariates export)
  - grid_meta_<city>.json (non-sensitive, provided by researcher)

It outputs per-city sparse weight parquet:
  - user_grid_time_<city>.parquet
  - qa_summary_<city>.csv
  - run_log_<city>.txt
  - (optional) user_covariates_<city>.parquet

Run:
  python -m poi_visit_aggregator.export_user_grid_time --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

try:
    from pyproj import CRS, Transformer  # type: ignore
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None


MS_PER_MIN = 60_000
MS_PER_DAY = 86_400_000


DEFAULT_WINDOWS = {
    "lunch": ("11:30", "14:00"),
    "dinner": ("17:00", "21:00"),
}


DEFAULT_FOOD_CATEGORY_KEYWORDS = [
    "餐",
    "餐饮",
    "餐厅",
    "饭",
    "美食",
    "小吃",
    "奶茶",
    "咖啡",
    "茶饮",
    "restaurant",
    "food",
    "cafe",
    "coffee",
    "tea",
]


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


def _as_posix(p: Path) -> str:
    return p.resolve().as_posix()


def _parse_hhmm(s: str) -> int:
    s = s.strip()
    hh, mm = s.split(":")
    return (int(hh) * 60 + int(mm)) * MS_PER_MIN


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_ms: int
    end_ms: int

    @property
    def segments(self) -> list[tuple[int, int]]:
        if self.end_ms > self.start_ms:
            return [(self.start_ms, self.end_ms)]
        return [(self.start_ms, MS_PER_DAY), (0, self.end_ms)]

    @property
    def duration_ms(self) -> int:
        if self.end_ms > self.start_ms:
            return self.end_ms - self.start_ms
        return (MS_PER_DAY - self.start_ms) + self.end_ms


@dataclass(frozen=True)
class GridMeta:
    grid_crs: Any
    coord_crs: Any
    cell_size_m: float
    origin_x: float
    origin_y: float
    n_cols: int
    n_rows: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y) in grid_crs

    @classmethod
    def from_json(cls, path: Path) -> "GridMeta":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("grid_meta json must be an object (dict) containing origin/cell_size/n_cols fields.")

        obj = raw
        for key in ("grid_meta", "meta", "params"):
            sub = raw.get(key)
            if isinstance(sub, dict) and ("origin_x" in sub or "origin_y" in sub or "n_cols" in sub):
                obj = {**raw, **sub}
                break

        bbox = None
        if isinstance(obj.get("bbox"), (list, tuple)) and len(obj["bbox"]) == 4:
            bbox = tuple(float(x) for x in obj["bbox"])
        elif isinstance(obj.get("bbox"), dict):
            b = obj["bbox"]
            bbox = (float(b["min_x"]), float(b["min_y"]), float(b["max_x"]), float(b["max_y"]))

        origin_x = obj.get("origin_x")
        origin_y = obj.get("origin_y")
        if (origin_x is None or origin_y is None) and isinstance(obj.get("origin"), dict):
            origin = obj["origin"]
            origin_x = origin_x if origin_x is not None else origin.get("x", origin.get("origin_x"))
            origin_y = origin_y if origin_y is not None else origin.get("y", origin.get("origin_y"))

        n_cols = obj.get("n_cols", obj.get("ncols"))
        cell_size_m = obj.get("cell_size_m", obj.get("cell_size", 500))

        grid_crs = obj.get("grid_crs") or obj.get("crs")
        if grid_crs is None:
            raise ValueError("grid_meta missing required field: grid_crs")
        if origin_x is None or origin_y is None:
            raise ValueError("grid_meta missing required fields: origin_x/origin_y")
        if n_cols is None:
            raise ValueError("grid_meta missing required field: n_cols")

        n_rows = obj.get("n_rows")
        if n_rows is not None:
            n_rows = int(n_rows)
        return cls(
            grid_crs=grid_crs,
            coord_crs=obj.get("coord_crs", "EPSG:4326"),
            cell_size_m=float(cell_size_m),
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            n_cols=int(n_cols),
            n_rows=n_rows,
            bbox=bbox,
        )


def _infer_grid_uid_code_from_grid_meta(path: Path) -> Optional[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    obj = raw
    for key in ("grid_meta", "meta", "params"):
        sub = raw.get(key)
        if isinstance(sub, dict):
            obj = {**raw, **sub}
            break

    for k in ("grid_uid_code", "city_code", "code", "adcode"):
        v = obj.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _normalize_grid_uid_code(*, city: str, grid_meta_path: Path, grid_uid_code: Optional[str]) -> str:
    if grid_uid_code is not None and str(grid_uid_code).strip():
        return str(grid_uid_code).strip()
    inferred = _infer_grid_uid_code_from_grid_meta(grid_meta_path)
    if inferred:
        return inferred
    return str(city).strip()


def _grid_uid_from_grid_id_int(
    grid_id: pd.Series,
    *,
    n_cols: int,
    prefix: str,
    code: str,
    order: str,
) -> pd.Series:
    order_n = order.strip().lower()
    if order_n not in {"col_row", "row_col"}:
        raise ValueError("grid_uid_order must be col_row|row_col")

    gid = pd.to_numeric(grid_id, errors="coerce").astype("Int64")
    col = (gid % int(n_cols)).astype("Int64")
    row = (gid // int(n_cols)).astype("Int64")

    if order_n == "col_row":
        suffix = col.astype("string") + "_" + row.astype("string")
    else:
        suffix = row.astype("string") + "_" + col.astype("string")

    out = (f"{prefix}_{code}_" + suffix).astype("string")
    return out.where(gid.notna(), pd.NA)


def _sql_grid_uid_expr(
    grid_id_sql: str,
    *,
    n_cols: int,
    prefix: str,
    code: str,
    order: str,
) -> str:
    order_n = order.strip().lower()
    if order_n not in {"col_row", "row_col"}:
        raise ValueError("grid_uid_order must be col_row|row_col")

    col_expr = f"({grid_id_sql} % {int(n_cols)})"
    row_expr = f"CAST(FLOOR(CAST({grid_id_sql} AS DOUBLE) / {int(n_cols)}) AS BIGINT)"
    if order_n == "col_row":
        suffix = f"CAST({col_expr} AS VARCHAR) || '_' || CAST({row_expr} AS VARCHAR)"
    else:
        suffix = f"CAST({row_expr} AS VARCHAR) || '_' || CAST({col_expr} AS VARCHAR)"

    prefix_sql = prefix.replace("'", "''")
    code_sql = code.replace("'", "''")
    return f"('{prefix_sql}_' || '{code_sql}_' || {suffix})"


def _infer_col(schema: pa.Schema, candidates: Iterable[str]) -> Optional[str]:
    names = list(schema.names)
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in candidates:
        cl = c.lower()
        for n in names:
            if cl in n.lower():
                return n
    return None


def _infer_df_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    names = list(columns)
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in candidates:
        cl = c.lower()
        for n in names:
            if cl in n.lower():
                return n
    return None


def _require_col(name: str, col: Optional[str]) -> str:
    if not col:
        raise ValueError(f"Cannot infer required column: {name}. Please provide --schema_map.")
    return col


def _col_as_int64(arr: pa.Array, *, fill: int = -1) -> np.ndarray:
    if pa.types.is_timestamp(arr.type):
        arr = pc.cast(arr, pa.timestamp("ms"))
        arr = pc.cast(arr, pa.int64())
    elif pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        arr = pc.utf8_trim_whitespace(arr)
        arr = pc.cast(arr, pa.int64())
    else:
        arr = pc.cast(arr, pa.int64())
    arr = arr.fill_null(fill)
    return arr.to_numpy(zero_copy_only=False)


def _col_as_float64(arr: pa.Array, *, fill_nan: bool = True) -> np.ndarray:
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        arr = pc.utf8_trim_whitespace(arr)
        arr = pc.cast(arr, pa.float64())
    else:
        arr = pc.cast(arr, pa.float64())
    if fill_nan:
        arr = arr.fill_null(np.nan)
    return arr.to_numpy(zero_copy_only=False)


def _col_as_str(arr: pa.Array) -> np.ndarray:
    if not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)):
        arr = pc.cast(arr, pa.string())
    arr = arr.fill_null("")
    return arr.to_numpy(zero_copy_only=False)


def _parse_location_to_lonlat(location_arr: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    if not (pa.types.is_string(location_arr.type) or pa.types.is_large_string(location_arr.type)):
        location_arr = pc.cast(location_arr, pa.string())
    location_arr = location_arr.fill_null("")
    clean = pc.replace_substring_regex(location_arr, pattern=r"[\[\]\(\)\s]", replacement="")
    parts = pc.split_pattern(clean, ",")
    lon_s = pc.list_element(parts, 0)
    lat_s = pc.list_element(parts, 1)
    lon = _col_as_float64(lon_s)
    lat = _col_as_float64(lat_s)
    return lon, lat


def _hash_uuid(values: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "pandas_64":
        s = pd.Series(values, copy=False)
        return hash_pandas_object(s, index=False).to_numpy(dtype="uint64")

    if method in {"sha256_64", "md5_64"}:
        out = np.empty(len(values), dtype=np.uint64)
        algo = hashlib.sha256 if method == "sha256_64" else hashlib.md5
        for i, v in enumerate(values):
            if v is None:
                out[i] = 0
                continue
            sv = str(v)
            if not sv:
                out[i] = 0
                continue
            digest = algo(sv.encode("utf-8")).digest()
            out[i] = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return out

    if method in {"xxh64", "xxhash64"}:
        try:
            import xxhash  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("uuid_hash_method=xxh64 requires `pip install xxhash`") from e
        out = np.empty(len(values), dtype=np.uint64)
        for i, v in enumerate(values):
            if v is None:
                out[i] = 0
                continue
            sv = str(v)
            if not sv:
                out[i] = 0
                continue
            out[i] = xxhash.xxh64(sv).intdigest()
        return out

    raise ValueError(f"Unknown uuid_hash_method: {method!r}")


def _overlap_one_day(start_ms: np.ndarray, end_ms: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    overlap = np.zeros(len(start_ms), dtype=np.int64)
    for seg_start, seg_end in segments:
        left = np.maximum(start_ms, seg_start)
        right = np.minimum(end_ms, seg_end)
        overlap += np.maximum(0, right - left).astype(np.int64)
    return overlap


def _compute_window_overlap_minutes(
    start_ms: np.ndarray,
    end_ms: np.ndarray,
    window: WindowSpec,
    *,
    rounding: str,
) -> np.ndarray:
    start_ms = start_ms.astype(np.int64, copy=False)
    end_ms = end_ms.astype(np.int64, copy=False)

    end_inclusive = end_ms - 1
    valid = (start_ms >= 0) & (end_ms > start_ms)
    start_ms = np.where(valid, start_ms, 0)
    end_inclusive = np.where(valid, end_inclusive, 0)

    start_day = start_ms // MS_PER_DAY
    end_day = end_inclusive // MS_PER_DAY
    day_span = (end_day - start_day).astype(np.int64)

    start_mod = start_ms - start_day * MS_PER_DAY
    end_mod = (end_inclusive - end_day * MS_PER_DAY) + 1  # exclusive within day

    overlap = np.zeros(len(start_ms), dtype=np.int64)
    same_day = day_span == 0
    if same_day.any():
        overlap[same_day] = _overlap_one_day(start_mod[same_day], end_mod[same_day], window.segments)

    multi_day = day_span >= 1
    if multi_day.any():
        overlap[multi_day] = (
            _overlap_one_day(start_mod[multi_day], np.full(multi_day.sum(), MS_PER_DAY, dtype=np.int64), window.segments)
            + _overlap_one_day(np.zeros(multi_day.sum(), dtype=np.int64), end_mod[multi_day], window.segments)
            + (day_span[multi_day] - 1) * window.duration_ms
        )

    overlap = np.where(valid, overlap, 0)

    rounding = rounding.lower()
    if rounding == "floor":
        return (overlap // MS_PER_MIN).astype(np.int32)
    if rounding == "round":
        return ((overlap + MS_PER_MIN // 2) // MS_PER_MIN).astype(np.int32)
    if rounding == "ceil":
        return ((overlap + MS_PER_MIN - 1) // MS_PER_MIN).astype(np.int32)
    raise ValueError(f"Unknown overlap rounding: {rounding!r}")


def _make_transformer(grid_meta: GridMeta) -> Optional[Any]:
    if Transformer is None or CRS is None:  # pragma: no cover
        return None
    return Transformer.from_crs(
        CRS.from_user_input(grid_meta.coord_crs),
        CRS.from_user_input(grid_meta.grid_crs),
        always_xy=True,
    )


def _map_xy_to_grid_id(
    x: np.ndarray,
    y: np.ndarray,
    grid_meta: GridMeta,
    *,
    oob_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    col = np.floor((x - grid_meta.origin_x) / grid_meta.cell_size_m).astype(np.int64)
    row = np.floor((y - grid_meta.origin_y) / grid_meta.cell_size_m).astype(np.int64)

    grid_id = (row * grid_meta.n_cols + col).astype(np.int64)

    valid = np.isfinite(x) & np.isfinite(y) & (row >= 0) & (col >= 0)
    valid &= col < grid_meta.n_cols
    if grid_meta.n_rows is not None:
        valid &= row < grid_meta.n_rows
    if grid_meta.bbox is not None:
        min_x, min_y, max_x, max_y = grid_meta.bbox
        valid &= (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

    oob_mode = oob_mode.lower()
    if oob_mode == "keep":
        return grid_id, ~valid
    if oob_mode == "null":
        return np.where(valid, grid_id, -1), ~valid
    if oob_mode == "drop":
        return grid_id, ~valid
    raise ValueError(f"Unknown oob_mode: {oob_mode!r}")


def _load_poi_food_set(
    poi_meta_path: Optional[Path],
    poi_filter_mode: str,
    schema_map: dict[str, Any],
    food_category_keywords: list[str],
    log,
) -> tuple[set[int], set[str]]:
    poi_filter_mode = poi_filter_mode.lower()
    if poi_filter_mode == "all_stops":
        return set(), set()
    if not poi_meta_path:
        raise ValueError("--poi_meta is required when poi_filter_mode != all_stops")

    if not poi_meta_path.exists():
        raise FileNotFoundError(f"poi_meta not found: {poi_meta_path}")

    poi_map = schema_map.get("poi_meta", {})
    if poi_meta_path.suffix.lower() == ".parquet":
        t = pq.read_table(_as_posix(poi_meta_path))
        schema = t.schema
        poi_id_col = poi_map.get("poi_id") or _infer_col(schema, ["poi_id", "poi"])
        cat_col = poi_map.get("category") or _infer_col(schema, ["poi_category", "poi_type", "category", "type"])
        poi_id_col = _require_col("poi_meta.poi_id", poi_id_col)
        cat_col = _require_col("poi_meta.category", cat_col)
        df = t.select([poi_id_col, cat_col]).to_pandas()
    else:
        df = pd.read_csv(poi_meta_path, on_bad_lines="skip")
        poi_id_col = poi_map.get("poi_id") or ("poi_id" if "poi_id" in df.columns else None)
        cat_col = poi_map.get("category") or next((c for c in df.columns if "category" in c.lower() or "type" in c.lower()), None)
        poi_id_col = _require_col("poi_meta.poi_id", poi_id_col)
        cat_col = _require_col("poi_meta.category", cat_col)
        df = df[[poi_id_col, cat_col]]

    keywords = [k.lower() for k in food_category_keywords if k.strip()]
    cat = df[cat_col].astype(str).str.lower()
    is_food = np.zeros(len(df), dtype=bool)
    for kw in keywords:
        is_food |= cat.str.contains(kw, na=False).to_numpy()

    food_ids_int: set[int] = set()
    food_ids_str: set[str] = set()

    poi_id_series = df[poi_id_col]
    poi_id_numeric = pd.to_numeric(poi_id_series, errors="coerce")
    for i, ok in enumerate(is_food.tolist()):
        if not ok:
            continue
        num = poi_id_numeric.iat[i]
        raw = poi_id_series.iat[i]
        if pd.notna(num):
            food_ids_int.add(int(num))
        if raw is not None:
            s = str(raw)
            if s and s.lower() != "nan":
                food_ids_str.add(s)

    log(f"Loaded food POI ids: int={len(food_ids_int):,}, str={len(food_ids_str):,}")
    return food_ids_int, food_ids_str


def _write_table_zstd(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        _as_posix(path),
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def export_user_grid_time(
    *,
    city: str,
    staypoints: list[Path],
    staypoints_format: str,
    uuid_table: Optional[Path],
    poi_meta: Optional[Path],
    grid_meta_path: Path,
    out_dir: Path,
    tmp_root: Optional[Path] = None,
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
    min_stay_minutes: int,
    exclude_poi_id_zero: bool,
    poi_filter_mode: str,
    food_category_keywords: list[str],
    timestamps_are_utc: bool,
    tz_offset_hours: int,
    epoch_unit: str,
    coords_already_projected: bool,
    uuid_hash_method: str,
    buckets: int,
    batch_size: int,
    log_every_batches: int = 500,
    overlap_rounding: str,
    oob_mode: str,
    threads: int,
    memory_limit: str,
    export_covariates: bool,
    keep_intermediate: bool,
) -> None:
    city_dir = out_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = city_dir / f"run_log_{city}.txt"

    log_file_failed = False

    def log(msg: str) -> None:
        nonlocal log_file_failed
        line = f"[{_now_str()}] {msg}"
        print(line)
        if log_file_failed:
            return
        try:
            run_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(run_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # pragma: no cover
            log_file_failed = True
            print(f"[{_now_str()}] WARN: cannot write run log to {run_log_path}: {e}")

    log("=" * 70)
    log(f"Start export_user_grid_time city={city!r}")
    log(f"staypoints files/paths: {len(staypoints)}")

    grid_meta = GridMeta.from_json(grid_meta_path)
    log(f"grid_meta loaded: cell_size_m={grid_meta.cell_size_m}, origin=({grid_meta.origin_x},{grid_meta.origin_y}), n_cols={grid_meta.n_cols}")

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

    tz_offset_ms = int(tz_offset_hours) * 3600 * 1000
    epoch_unit = epoch_unit.lower()
    if epoch_unit not in {"ms", "s"}:
        raise ValueError("--epoch_unit must be ms or s")
    log_every_batches_n = int(log_every_batches)
    if log_every_batches_n < 0:
        log_every_batches_n = 0
    if export_covariates and not uuid_table:
        raise ValueError("--uuid_table is required when export_covariates=true")

    windows_norm = [w.strip().lower() for w in windows]
    window_specs: dict[str, WindowSpec] = {}
    for w in windows_norm:
        if w not in DEFAULT_WINDOWS:
            raise ValueError(f"Unknown window: {w!r}. Supported: {sorted(DEFAULT_WINDOWS)}")
        s, e = DEFAULT_WINDOWS[w]
        window_specs[w] = WindowSpec(name=w, start_ms=_parse_hhmm(s), end_ms=_parse_hhmm(e))

    food_ids_int, food_ids_str = _load_poi_food_set(poi_meta, poi_filter_mode, schema_map, food_category_keywords, log)
    food_ids_int_arr = np.fromiter(food_ids_int, dtype=np.int64) if food_ids_int else None
    food_ids_str_arr = np.fromiter(food_ids_str, dtype=object) if food_ids_str else None

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

    if staypoints_format_n == "csv":
        # Avoid schema inference over many daily files by inferring from first file only.
        sample = ds.dataset([ds_paths[0]], format="csv")
        dataset = ds.dataset(ds_paths, format="csv", schema=sample.schema)
    else:
        dataset = ds.dataset(ds_paths, format="parquet")
    schema = dataset.schema

    uuid_col = stay_schema_map.get("uuid") or _infer_col(schema, ["uuid", "user", "uid"])
    poi_id_col = stay_schema_map.get("poi_id") or _infer_col(schema, ["poi_id", "poi"])
    start_col = stay_schema_map.get("start_time") or _infer_col(schema, ["start_time", "start", "start_ms", "stime"])
    end_col = stay_schema_map.get("end_time") or _infer_col(schema, ["end_time", "end", "end_ms", "etime"])
    stay_min_col = stay_schema_map.get("stay_minutes") or _infer_col(schema, ["stay_minutes", "duration_min", "stay_min", "minutes"])

    uuid_col = _require_col("staypoints.uuid", uuid_col)
    poi_id_col = _require_col("staypoints.poi_id", poi_id_col)
    start_col = _require_col("staypoints.start_time", start_col)
    end_col = _require_col("staypoints.end_time", end_col)

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
    elif coord_mode == "xy" and not coords_already_projected:
        log("coords_already_projected=false but x/y columns detected; using x/y without projection.")

    scan_cols = [uuid_col, poi_id_col, start_col, end_col]
    if stay_min_col:
        scan_cols.append(stay_min_col)
    if coord_mode == "xy":
        scan_cols += [x_col, y_col]
    elif coord_mode == "lonlat":
        scan_cols += [lon_col, lat_col]
    else:
        scan_cols.append(loc_col)

    scan_cols = [c for c in scan_cols if c is not None]
    log(f"Staypoints scan columns: {scan_cols}")

    tmp_root_n = Path(tmp_root).expanduser() if tmp_root is not None else None
    tmp_dir = (tmp_root_n / city / "_tmp") if tmp_root_n is not None else (city_dir / "_tmp")
    bucket_root = tmp_dir / "buckets"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if bucket_root.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = bucket_root.with_name(f"{bucket_root.name}_old_{ts}")
        i = 0
        while backup.exists():
            i += 1
            backup = bucket_root.with_name(f"{bucket_root.name}_old_{ts}_{i}")
        try:
            bucket_root.rename(backup)
            log(f"Moved old bucket files aside: {backup}")
        except Exception as e:
            log(f"WARN: failed to rename {bucket_root} ({e}); trying to delete it instead.")
            try:
                shutil.rmtree(bucket_root)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to reset intermediate bucket dir {bucket_root}. "
                    "On Colab, avoid writing intermediates to Google Drive."
                ) from e2

    bucket_root.mkdir(parents=True, exist_ok=True)

    bucket_part = np.zeros(int(buckets), dtype=np.int64)

    qa: dict[str, Any] = {
        "city": city,
        "tmp_root": _as_posix(tmp_root_n) if tmp_root_n is not None else "",
        "tmp_dir": _as_posix(tmp_dir),
        "output_grid_uid": bool(output_grid_uid),
        "output_grid_id": bool(output_grid_id),
        "grid_uid_prefix": grid_uid_prefix_n,
        "grid_uid_code": grid_uid_code_n,
        "grid_uid_order": grid_uid_order_n,
        "filter_city_code": bool(filter_city_code),
        "city_code_col": city_code_col,
        "city_code_value": city_code_value,
        "log_every_batches": int(log_every_batches_n),
        "input_rows": 0,
        "kept_rows": 0,
        "filtered_poi_id_zero": 0,
        "filtered_min_stay": 0,
        "filtered_poi_category": 0,
        "filtered_not_in_windows": 0,
        "filtered_oob_grid": 0,
        "invalid_time_rows": 0,
        "pct_cross_midnight": 0.0,
        "cross_midnight_rows": 0,
        "tau_lunch_total_min": 0,
        "tau_dinner_total_min": 0,
        "tau_total_min": 0,
    }
    stay_error_samples: list[float] = []
    max_error_samples = 1_000_000

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
    batch_i = 0
    for batch in scanner.to_batches():
        batch_i += 1
        n = batch.num_rows
        qa["input_rows"] += n
        if n == 0:
            continue

        uuid_vals = _col_as_str(batch.column(uuid_col))
        poi_vals_raw = batch.column(poi_id_col)
        poi_is_int = True
        try:
            poi_vals_int = _col_as_int64(poi_vals_raw)
        except Exception:
            poi_is_int = False
            poi_vals_int = None
            poi_vals_str = _col_as_str(poi_vals_raw)

        start_ms = _col_as_int64(batch.column(start_col))
        end_ms = _col_as_int64(batch.column(end_col))
        if epoch_unit == "s":
            start_ms = start_ms * 1000
            end_ms = end_ms * 1000
        if timestamps_are_utc:
            start_ms = start_ms + tz_offset_ms
            end_ms = end_ms + tz_offset_ms

        valid_time = (start_ms >= 0) & (end_ms > start_ms)
        qa["invalid_time_rows"] += int((~valid_time).sum())

        if stay_min_col and stay_min_col in batch.schema.names:
            stay_min_raw = batch.column(stay_min_col)
            stay_min = _col_as_float64(stay_min_raw)
            computed_min = (end_ms - start_ms) / MS_PER_MIN
            err = stay_min - computed_min
            if len(stay_error_samples) < max_error_samples:
                remain = max_error_samples - len(stay_error_samples)
                take = min(remain, len(err))
                stay_error_samples.extend(err[:take].astype(float).tolist())
        else:
            stay_min = (end_ms - start_ms) / MS_PER_MIN

        mask = valid_time & (uuid_vals != "")
        if poi_is_int:
            mask &= poi_vals_int >= 0  # type: ignore[operator]
        else:
            mask &= poi_vals_str != ""  # type: ignore[has-type]

        if exclude_poi_id_zero:
            before = int(mask.sum())
            if poi_is_int:
                mask &= poi_vals_int != 0  # type: ignore[operator]
            else:
                mask &= poi_vals_str != "0"  # type: ignore[has-type]
            qa["filtered_poi_id_zero"] += before - int(mask.sum())

        if min_stay_minutes > 0:
            before = int(mask.sum())
            mask &= stay_min >= float(min_stay_minutes)
            qa["filtered_min_stay"] += before - int(mask.sum())

        poi_filter_mode_n = poi_filter_mode.lower()
        if poi_filter_mode_n != "all_stops":
            before = int(mask.sum())
            if poi_is_int and food_ids_int_arr is not None:
                is_food = np.isin(poi_vals_int, food_ids_int_arr, assume_unique=False)  # type: ignore[arg-type]
            elif (not poi_is_int) and food_ids_str_arr is not None:
                is_food = np.isin(poi_vals_str, food_ids_str_arr, assume_unique=False)  # type: ignore[arg-type]
            else:
                is_food = np.zeros(n, dtype=bool)

            if poi_filter_mode_n == "exclude_food_poi":
                mask &= ~is_food
            elif poi_filter_mode_n == "only_food_poi":
                mask &= is_food
            else:
                raise ValueError(f"Unknown poi_filter_mode: {poi_filter_mode}")
            qa["filtered_poi_category"] += before - int(mask.sum())

        if not mask.any():
            continue

        idx = np.nonzero(mask)[0]
        start_ms_f = start_ms[idx]
        end_ms_f = end_ms[idx]

        cross_midnight = (start_ms_f // MS_PER_DAY) != ((end_ms_f - 1) // MS_PER_DAY)
        qa["cross_midnight_rows"] += int(cross_midnight.sum())

        tau_lunch = np.zeros(len(idx), dtype=np.int32)
        tau_dinner = np.zeros(len(idx), dtype=np.int32)
        if "lunch" in window_specs:
            tau_lunch = _compute_window_overlap_minutes(start_ms_f, end_ms_f, window_specs["lunch"], rounding=overlap_rounding)
        if "dinner" in window_specs:
            tau_dinner = _compute_window_overlap_minutes(start_ms_f, end_ms_f, window_specs["dinner"], rounding=overlap_rounding)

        in_windows = (tau_lunch > 0) | (tau_dinner > 0)
        before = len(idx)
        idx = idx[in_windows]
        tau_lunch = tau_lunch[in_windows]
        tau_dinner = tau_dinner[in_windows]
        qa["filtered_not_in_windows"] += before - len(idx)

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
            tau_lunch = tau_lunch[keep]
            tau_dinner = tau_dinner[keep]
            idx = idx[keep]
        else:
            qa["filtered_oob_grid"] += int(oob_mask.sum())

        if len(idx) == 0:
            continue

        uuid_hash = _hash_uuid(uuid_vals[idx], uuid_hash_method)
        bucket_ids = (uuid_hash % np.uint64(buckets)).astype(np.int32)

        n_stops_lunch = (tau_lunch > 0).astype(np.int32)
        n_stops_dinner = (tau_dinner > 0).astype(np.int32)
        tau_total = (tau_lunch.astype(np.int64) + tau_dinner.astype(np.int64)).astype(np.int32)

        qa["tau_lunch_total_min"] += int(tau_lunch.sum())
        qa["tau_dinner_total_min"] += int(tau_dinner.sum())
        qa["tau_total_min"] += int(tau_total.sum())
        qa["kept_rows"] += len(idx)

        order = np.argsort(bucket_ids, kind="stable")
        bucket_sorted = bucket_ids[order]
        if len(bucket_sorted) == 0:
            continue
        changes = np.nonzero(np.diff(bucket_sorted))[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(bucket_sorted)]))

        for s, e in zip(starts, ends):
            b = int(bucket_sorted[s])
            sub_idx = order[s:e]
            out_path = bucket_root / f"{b:03d}" / f"part_{int(bucket_part[b]):06d}.parquet"
            bucket_part[b] += 1
            table = pa.table(
                {
                    "uuid_hash": pa.array(uuid_hash[sub_idx], type=pa.uint64()),
                    "grid_id": pa.array(grid_id_raw[sub_idx].astype(np.int32), type=pa.int32()),
                    "tau_lunch_min": pa.array(tau_lunch[sub_idx].astype(np.int32), type=pa.int32()),
                    "tau_dinner_min": pa.array(tau_dinner[sub_idx].astype(np.int32), type=pa.int32()),
                    "tau_total_min": pa.array(tau_total[sub_idx].astype(np.int32), type=pa.int32()),
                    "n_stops_lunch": pa.array(n_stops_lunch[sub_idx].astype(np.int32), type=pa.int32()),
                    "n_stops_dinner": pa.array(n_stops_dinner[sub_idx].astype(np.int32), type=pa.int32()),
                }
            )
            _write_table_zstd(table, out_path)

        if log_every_batches_n > 0 and batch_i % log_every_batches_n == 0:
            log(f"Processed batches={batch_i:,}, input_rows={qa['input_rows']:,}, kept_rows={qa['kept_rows']:,}")

    if qa["input_rows"] > 0:
        qa["pct_cross_midnight"] = qa["cross_midnight_rows"] / qa["input_rows"]
        qa["kept_row_ratio"] = qa["kept_rows"] / qa["input_rows"]
    else:
        qa["kept_row_ratio"] = 0.0

    if qa["tau_total_min"] > 0:
        qa["tau_lunch_ratio"] = qa["tau_lunch_total_min"] / qa["tau_total_min"]
        qa["tau_dinner_ratio"] = qa["tau_dinner_total_min"] / qa["tau_total_min"]
    else:
        qa["tau_lunch_ratio"] = 0.0
        qa["tau_dinner_ratio"] = 0.0

    user_grid_time_path = city_dir / f"user_grid_time_{city}.parquet"
    if qa["kept_rows"] == 0:
        log("No rows kept after filtering; writing empty user_grid_time parquet.")
        empty_cols: dict[str, pa.Array] = {
            "uuid_hash": pa.array([], type=pa.uint64()),
        }
        if output_grid_uid:
            empty_cols["grid_uid"] = pa.array([], type=pa.string())
        if output_grid_id:
            empty_cols["grid_id"] = pa.array([], type=pa.int32())
        empty_cols.update(
            {
                "tau_lunch_min": pa.array([], type=pa.int32()),
                "tau_dinner_min": pa.array([], type=pa.int32()),
                "tau_total_min": pa.array([], type=pa.int32()),
                "n_stops_lunch": pa.array([], type=pa.int32()),
                "n_stops_dinner": pa.array([], type=pa.int32()),
            }
        )
        empty = pa.table(empty_cols)
        _write_table_zstd(empty, user_grid_time_path)
        qa["output_rows"] = 0
        qa["output_parquet_bytes"] = int(user_grid_time_path.stat().st_size)
        qa["output_parquet_mb"] = round(qa["output_parquet_bytes"] / 1024 / 1024, 3)
        qa["avg_unique_grids_per_user_lunch"] = 0.0
        qa["avg_unique_grids_per_user_dinner"] = 0.0
        qa_path = city_dir / f"qa_summary_{city}.csv"
        pd.DataFrame([qa]).to_csv(qa_path, index=False, encoding="utf-8-sig")
        log(f"Wrote: {user_grid_time_path}")
        log(f"Wrote: {qa_path}")
        log("Done.")
        log("=" * 70)
        return
    if duckdb is None:
        raise RuntimeError("Stage-2 aggregation requires DuckDB: `pip install duckdb`")

    log("Stage-2 DuckDB aggregation...")
    temp_dir = tmp_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{_as_posix(temp_dir)}'")

    bucket_glob = _as_posix(bucket_root) + "/*/*.parquet"
    con.execute(f"CREATE VIEW bucket_data AS SELECT * FROM read_parquet('{bucket_glob}')")
    select_cols: list[str] = []
    if output_grid_id:
        select_cols.append("grid_id")
    if output_grid_uid:
        grid_uid_expr = _sql_grid_uid_expr(
            "grid_id",
            n_cols=grid_meta.n_cols,
            prefix=grid_uid_prefix_n,
            code=grid_uid_code_n,
            order=grid_uid_order_n,
        )
        select_cols.append(f"{grid_uid_expr} AS grid_uid")
    spatial_select = ""
    if select_cols:
        spatial_select = ",\n                " + ",\n                ".join(select_cols)
    con.execute(
        f"""
        COPY (
            SELECT
                uuid_hash{spatial_select},
                tau_lunch_min,
                tau_dinner_min,
                tau_total_min,
                n_stops_lunch,
                n_stops_dinner
            FROM (
                SELECT
                    uuid_hash,
                    grid_id,
                    CAST(SUM(tau_lunch_min) AS INTEGER) AS tau_lunch_min,
                    CAST(SUM(tau_dinner_min) AS INTEGER) AS tau_dinner_min,
                    CAST(SUM(tau_total_min) AS INTEGER) AS tau_total_min,
                    CAST(SUM(n_stops_lunch) AS INTEGER) AS n_stops_lunch,
                    CAST(SUM(n_stops_dinner) AS INTEGER) AS n_stops_dinner
                FROM bucket_data
                WHERE grid_id IS NOT NULL AND grid_id >= 0
                GROUP BY uuid_hash, grid_id
            ) agg
        ) TO '{_as_posix(user_grid_time_path)}' (FORMAT PARQUET, CODEC ZSTD);
        """
    )

    log("Computing QA on output parquet...")
    out_rel = f"read_parquet('{_as_posix(user_grid_time_path)}')"
    qa["output_rows"] = int(con.execute(f"SELECT COUNT(*) FROM {out_rel}").fetchone()[0])
    qa["output_parquet_bytes"] = int(user_grid_time_path.stat().st_size)
    qa["output_parquet_mb"] = round(qa["output_parquet_bytes"] / 1024 / 1024, 3)
    qa["avg_unique_grids_per_user_lunch"] = float(
        con.execute(
            f"""
            SELECT COALESCE(AVG(cnt), 0)
            FROM (SELECT uuid_hash, COUNT(*) AS cnt FROM {out_rel} WHERE tau_lunch_min > 0 GROUP BY uuid_hash)
            """
        ).fetchone()[0]
    )
    qa["avg_unique_grids_per_user_dinner"] = float(
        con.execute(
            f"""
            SELECT COALESCE(AVG(cnt), 0)
            FROM (SELECT uuid_hash, COUNT(*) AS cnt FROM {out_rel} WHERE tau_dinner_min > 0 GROUP BY uuid_hash)
            """
        ).fetchone()[0]
    )

    if stay_error_samples:
        arr = np.asarray(stay_error_samples, dtype=np.float64)
        qa["stay_minutes_error_p50"] = float(np.nanpercentile(arr, 50))
        qa["stay_minutes_error_p90"] = float(np.nanpercentile(arr, 90))
        qa["stay_minutes_error_p99"] = float(np.nanpercentile(arr, 99))

    qa_path = city_dir / f"qa_summary_{city}.csv"
    pd.DataFrame([qa]).to_csv(qa_path, index=False, encoding="utf-8-sig")
    log(f"Wrote: {user_grid_time_path}")
    log(f"Wrote: {qa_path}")

    if export_covariates and uuid_table:
        log("Exporting user_covariates...")
        uuid_schema_map = schema_map.get("uuid_table", {})
        if uuid_table.suffix.lower() == ".parquet":
            ut = pq.read_table(_as_posix(uuid_table))
            us = ut.schema
            uuid_c = uuid_schema_map.get("uuid") or _infer_col(us, ["uuid", "user", "uid"])
            uuid_c = _require_col("uuid_table.uuid", uuid_c)
            ses_cols = uuid_schema_map.get("ses_cols")
            if isinstance(ses_cols, str):
                ses_cols = [c.strip() for c in ses_cols.split(",") if c.strip()]
            if not ses_cols:
                ses_cols = [c for c in ut.column_names if c.lower().startswith("ses")]

            available = set(ut.column_names)
            home_lon_c = uuid_schema_map.get("home_lon") or _infer_col(us, ["home_lon", "home_lng", "home_longitude"])
            home_lat_c = uuid_schema_map.get("home_lat") or _infer_col(us, ["home_lat", "home_latitude"])
            home_x_c = uuid_schema_map.get("home_x") or _infer_col(us, ["home_x", "home_easting"])
            home_y_c = uuid_schema_map.get("home_y") or _infer_col(us, ["home_y", "home_northing"])
            work_lon_c = uuid_schema_map.get("work_lon") or _infer_col(us, ["work_lon", "work_lng", "work_longitude"])
            work_lat_c = uuid_schema_map.get("work_lat") or _infer_col(us, ["work_lat", "work_latitude"])
            work_x_c = uuid_schema_map.get("work_x") or _infer_col(us, ["work_x", "work_easting"])
            work_y_c = uuid_schema_map.get("work_y") or _infer_col(us, ["work_y", "work_northing"])
            coord_cols = [
                c
                for c in [home_lon_c, home_lat_c, home_x_c, home_y_c, work_lon_c, work_lat_c, work_x_c, work_y_c]
                if c and c in available
            ]

            ses_keep = [c for c in ses_cols if c in available]
            keep_cols = list(dict.fromkeys([uuid_c] + ses_keep + coord_cols))
            udf = ut.select(keep_cols).to_pandas()
        else:
            udf = pd.read_csv(uuid_table, on_bad_lines="skip")
            uuid_c = uuid_schema_map.get("uuid") or ("uuid" if "uuid" in udf.columns else None)
            uuid_c = _require_col("uuid_table.uuid", uuid_c)
            ses_cols = uuid_schema_map.get("ses_cols")
            if isinstance(ses_cols, str):
                ses_cols = [c.strip() for c in ses_cols.split(",") if c.strip()]
            if not ses_cols:
                ses_cols = [c for c in udf.columns if c.lower().startswith("ses")]

            home_lon_c = uuid_schema_map.get("home_lon") or _infer_df_col(udf.columns, ["home_lon", "home_lng", "home_longitude"])
            home_lat_c = uuid_schema_map.get("home_lat") or _infer_df_col(udf.columns, ["home_lat", "home_latitude"])
            home_x_c = uuid_schema_map.get("home_x") or _infer_df_col(udf.columns, ["home_x", "home_easting"])
            home_y_c = uuid_schema_map.get("home_y") or _infer_df_col(udf.columns, ["home_y", "home_northing"])
            work_lon_c = uuid_schema_map.get("work_lon") or _infer_df_col(udf.columns, ["work_lon", "work_lng", "work_longitude"])
            work_lat_c = uuid_schema_map.get("work_lat") or _infer_df_col(udf.columns, ["work_lat", "work_latitude"])
            work_x_c = uuid_schema_map.get("work_x") or _infer_df_col(udf.columns, ["work_x", "work_easting"])
            work_y_c = uuid_schema_map.get("work_y") or _infer_df_col(udf.columns, ["work_y", "work_northing"])

            ses_keep = [c for c in ses_cols if c in udf.columns]
            coord_cols = [c for c in [home_lon_c, home_lat_c, home_x_c, home_y_c, work_lon_c, work_lat_c, work_x_c, work_y_c] if c and c in udf.columns]
            keep_cols = list(dict.fromkeys([uuid_c] + ses_keep + coord_cols))
            udf = udf[keep_cols]

        udf["uuid_hash"] = _hash_uuid(udf[uuid_c].astype(str).to_numpy(), uuid_hash_method)

        cov_transformer = transformer or _make_transformer(grid_meta)

        def maybe_add_grid(col_x: Optional[str], col_y: Optional[str], col_lon: Optional[str], col_lat: Optional[str], out_col: str) -> None:
            if col_x and col_y and col_x in udf.columns and col_y in udf.columns:
                x = pd.to_numeric(udf[col_x], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                y = pd.to_numeric(udf[col_y], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            elif col_lon and col_lat and col_lon in udf.columns and col_lat in udf.columns:
                if cov_transformer is None:
                    log(f"Skip {out_col}: need pyproj for lon/lat -> grid projection")
                    return
                lon = pd.to_numeric(udf[col_lon], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                lat = pd.to_numeric(udf[col_lat], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                x, y = cov_transformer.transform(lon, lat)  # type: ignore[union-attr]
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)
            else:
                return

            gid, bad = _map_xy_to_grid_id(x, y, grid_meta, oob_mode="null")
            s = pd.Series(gid)
            s = s.mask(bad | (s < 0), pd.NA).astype("Int32")
            udf[out_col] = s

        maybe_add_grid(home_x_c, home_y_c, home_lon_c, home_lat_c, "home_grid_id")
        maybe_add_grid(work_x_c, work_y_c, work_lon_c, work_lat_c, "work_grid_id")

        if output_grid_uid:
            if "home_grid_id" in udf.columns:
                udf["home_grid_uid"] = _grid_uid_from_grid_id_int(
                    udf["home_grid_id"],
                    n_cols=grid_meta.n_cols,
                    prefix=grid_uid_prefix_n,
                    code=grid_uid_code_n,
                    order=grid_uid_order_n,
                )
            if "work_grid_id" in udf.columns:
                udf["work_grid_uid"] = _grid_uid_from_grid_id_int(
                    udf["work_grid_id"],
                    n_cols=grid_meta.n_cols,
                    prefix=grid_uid_prefix_n,
                    code=grid_uid_code_n,
                    order=grid_uid_order_n,
                )

        out_cols = ["uuid_hash"]
        out_cols += [c for c in ses_keep if c in udf.columns]
        if output_grid_id:
            out_cols += [c for c in ["home_grid_id", "work_grid_id"] if c in udf.columns]
        if output_grid_uid:
            out_cols += [c for c in ["home_grid_uid", "work_grid_uid"] if c in udf.columns]
        cov = udf[out_cols].copy()

        cov_path = city_dir / f"user_covariates_{city}.parquet"
        cov_table = pa.Table.from_pandas(cov, preserve_index=False)
        _write_table_zstd(cov_table, cov_path)
        log(f"Wrote: {cov_path}")

    if not keep_intermediate and bucket_root.exists():
        log("Cleaning intermediate bucket parquet...")
        try:
            shutil.rmtree(bucket_root)
        except Exception as e:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = bucket_root.with_name(f"{bucket_root.name}_old_{ts}")
            i = 0
            while backup.exists():
                i += 1
                backup = bucket_root.with_name(f"{bucket_root.name}_old_{ts}_{i}")
            try:
                bucket_root.rename(backup)
                log(f"WARN: failed to delete {bucket_root} ({e}); renamed to {backup} instead.")
            except Exception:
                log(f"WARN: failed to delete {bucket_root}: {e}")

    log("Done.")
    log("=" * 70)


def _resolve_input_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        matches = glob(pat)
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(pat))
    uniq: list[Path] = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export uuid_hash×grid_uid stay-minute weights (口径A)")
    p.add_argument("--city", required=True)
    p.add_argument("--staypoints", required=True, nargs="+", help="Parquet/CSV path(s) or glob(s)")
    p.add_argument("--staypoints_format", default="auto", choices=["auto", "parquet", "csv"])
    p.add_argument("--uuid_table", default=None, help="Optional uuid table (.parquet or .csv)")
    p.add_argument("--poi_meta", default=None, help="Optional POI category table (.parquet or .csv)")
    p.add_argument("--grid_meta", required=True, help="grid_meta_<city>.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tmp_root", default=None, help="Optional intermediate tmp root (recommended on Colab: /tmp)")

    p.add_argument("--schema_map", default=None, help="JSON string or path to JSON file")
    p.add_argument("--output_grid_uid", type=_str2bool, default=True, help="If true, output string grid_uid")
    p.add_argument("--output_grid_id", type=_str2bool, default=False, help="If true, also output numeric grid_id (=row*n_cols+col)")
    p.add_argument("--grid_uid_code", default=None, help="City code used in grid_uid (e.g. 4403); defaults to grid_meta.code/adcode or --city")
    p.add_argument("--grid_uid_prefix", default="grid", help="Prefix used in grid_uid (default: grid)")
    p.add_argument("--grid_uid_order", default="col_row", choices=["col_row", "row_col"], help="Order of col/row suffix in grid_uid")
    p.add_argument("--filter_city_code", type=_str2bool, default=False, help="If true, pre-filter staypoints by city code column (e.g. c_code)")
    p.add_argument("--city_code_col", default=None, help="Staypoints city code column name (default: infer, e.g. c_code)")
    p.add_argument("--city_code_value", default=None, help="City code value to filter (default: --grid_uid_code or grid_meta.code/adcode)")
    p.add_argument("--windows", default="lunch", help="Comma-separated: lunch,dinner")
    p.add_argument("--min_stay_minutes", type=int, default=5)
    p.add_argument("--exclude_poi_id_zero", type=_str2bool, default=True)
    p.add_argument("--poi_filter_mode", default="all_stops", choices=["all_stops", "exclude_food_poi", "only_food_poi"])
    p.add_argument("--food_category_keywords", default=",".join(DEFAULT_FOOD_CATEGORY_KEYWORDS))

    p.add_argument("--timestamps_are_utc", type=_str2bool, default=True)
    p.add_argument("--tz_offset_hours", type=int, default=8)
    p.add_argument("--epoch_unit", default="ms", choices=["ms", "s"])

    p.add_argument("--coords_already_projected", type=_str2bool, default=False)
    p.add_argument("--uuid_hash_method", default="sha256_64", choices=["sha256_64", "md5_64", "pandas_64", "xxh64"])
    p.add_argument("--buckets", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1_000_000)
    p.add_argument("--log_every_batches", type=int, default=500, help="Log progress every N batches (0 disables)")
    p.add_argument("--overlap_rounding", default="floor", choices=["floor", "round", "ceil"])
    p.add_argument("--oob_mode", default="drop", choices=["drop", "null", "keep"])

    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory_limit", default="16GB")
    p.add_argument("--export_covariates", type=_str2bool, default=False)
    p.add_argument("--keep_intermediate", type=_str2bool, default=False)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    schema_map = _load_json_arg(args.schema_map)

    staypoints = _resolve_input_paths(args.staypoints)
    if not staypoints:
        raise SystemExit("No staypoints files found.")

    uuid_table = Path(args.uuid_table).resolve() if args.uuid_table else None
    poi_meta = Path(args.poi_meta).resolve() if args.poi_meta else None
    grid_meta_path = Path(args.grid_meta).resolve()
    out_dir = Path(args.out_dir).resolve()
    tmp_root = Path(args.tmp_root).resolve() if args.tmp_root else None

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    food_keywords = [k.strip() for k in args.food_category_keywords.split(",") if k.strip()]

    export_user_grid_time(
        city=args.city,
        staypoints=staypoints,
        staypoints_format=args.staypoints_format,
        uuid_table=uuid_table,
        poi_meta=poi_meta,
        grid_meta_path=grid_meta_path,
        out_dir=out_dir,
        tmp_root=tmp_root,
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
        min_stay_minutes=args.min_stay_minutes,
        exclude_poi_id_zero=args.exclude_poi_id_zero,
        poi_filter_mode=args.poi_filter_mode,
        food_category_keywords=food_keywords,
        timestamps_are_utc=args.timestamps_are_utc,
        tz_offset_hours=args.tz_offset_hours,
        epoch_unit=args.epoch_unit,
        coords_already_projected=args.coords_already_projected,
        uuid_hash_method=args.uuid_hash_method,
        buckets=args.buckets,
        batch_size=args.batch_size,
        log_every_batches=args.log_every_batches,
        overlap_rounding=args.overlap_rounding,
        oob_mode=args.oob_mode,
        threads=args.threads,
        memory_limit=args.memory_limit,
        export_covariates=args.export_covariates,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
