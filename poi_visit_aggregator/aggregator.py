# Copyright 2025 Weipeng Deng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


"""
用户POI驻留数据两阶段聚合核心逻辑

Stage 1: 按 chunk 部分聚合，输出中间 parquet
Stage 2: 流式分桶聚合，避免内存溢出
"""

__version__ = "1.1.4"

import json
import math
import os
import psutil
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object

# 全局变量
_tqdm = None
_tqdm_pandas = None
_log_file = None
_quiet_mode = False  # Stage 2 bucket 处理时静默模式


def _init_tqdm(notebook: bool = False):
    """初始化 tqdm，根据环境选择合适的版本"""
    global _tqdm, _tqdm_pandas
    
    if notebook:
        from tqdm.notebook import tqdm as tqdm_nb
        from tqdm.notebook import tqdm as tqdm_pandas_nb
        _tqdm = tqdm_nb
        _tqdm_pandas = tqdm_pandas_nb
    else:
        from tqdm import tqdm as tqdm_std
        _tqdm = tqdm_std
        _tqdm_pandas = tqdm_std
    
    return _tqdm


def _init_log(log_path: Path):
    """初始化日志文件"""
    global _log_file
    _log_file = log_path
    # 确保目录存在
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    # 追加模式写入分隔线
    with open(_log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"任务开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")


def _log_status(msg: str, force_print: bool = False):
    """
    打印带时间戳和内存占用的状态信息
    
    Args:
        msg: 日志消息
        force_print: 是否强制打印到控制台（即使在静默模式下）
    """
    now = datetime.now().strftime("%H:%M:%S")
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    log_line = f"[{now}] [RAM: {mem_mb:,.0f} MB] {msg}"
    
    # 写入日志文件
    if _log_file is not None:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    
    # 控制台输出（静默模式下只输出 force_print=True 的内容）
    if not _quiet_mode or force_print:
        print(log_line)


# -----------------------------
# 全局常量（分布列名定义）
# -----------------------------
ENTER_HOURS = list(range(24))
LEAVE_HOURS = list(range(24))
WEEKDAYS = list(range(1, 8))
DAYS_OF_MONTH = list(range(1, 32))

ENTER_COLS = [f"enter_h_{h:02d}" for h in ENTER_HOURS]
LEAVE_COLS = [f"leave_h_{h:02d}" for h in LEAVE_HOURS]
WEEKDAY_COLS = [f"wd_{d}" for d in WEEKDAYS]
DOM_COLS = [f"dom_{d:02d}" for d in DAYS_OF_MONTH]

DIST_COLS = ENTER_COLS + LEAVE_COLS + WEEKDAY_COLS + DOM_COLS

# 最终输出的列
FINAL_COLS = [
    "uuid", "poi_id", "visit_count", "total_stay_minutes", "avg_stay_minutes",
    "max_stay_minutes", "std_stay_minutes", "first_visit_date", "last_visit_date",
    "last_visit_recency_days", "weekday_visit_ratio", "weekend_visit_ratio",
    "daytime_ratio", "night_ratio", "enter_hour_dist", "leave_hour_dist",
    "visit_weekday_dist", "visit_day_of_month_dist", "top_enter_hours",
    "top_leave_hours", "top_weekdays",
    "enter_period_stats", "leave_period_stats", "weekday_group_stats", "date_group_stats",
]


def _aggregate_chunk(chunk: pd.DataFrame) -> tuple:
    """
    对单个 chunk 进行 uuid+poi_id 粗聚合
    
    Args:
        chunk: 包含原始数据的 DataFrame
        
    Returns:
        (聚合后的 DataFrame, POI坐标 DataFrame)
    """
    # 1. 过滤掉 poi_id == 0
    if "poi_id" in chunk.columns:
        if pd.api.types.is_numeric_dtype(chunk["poi_id"]):
            chunk = chunk[chunk["poi_id"] != 0]
        else:
            chunk = chunk[chunk["poi_id"] != "0"]

    if chunk.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 提取 POI 坐标（如果有 location 字段）
    poi_coords = pd.DataFrame()
    if "location" in chunk.columns:
        poi_coords = chunk[["poi_id", "location"]].drop_duplicates(subset=["poi_id"])

    # 2. 只保留需要的列
    cols_keep = [
        "uuid",
        "poi_id",
        "start_hour",
        "end_hour",
        "stay_minutes",
        "date_str",
    ]
    chunk = chunk[cols_keep].copy()

    # 类型转换和派生字段
    chunk["stay_minutes"] = chunk["stay_minutes"].astype(float)
    chunk["start_hour"] = chunk["start_hour"].astype("Int64")
    chunk["end_hour"] = chunk["end_hour"].astype("Int64")
    chunk["date"] = pd.to_datetime(chunk["date_str"])
    chunk["weekday"] = chunk["date"].dt.dayofweek + 1
    chunk["day_of_month"] = chunk["date"].dt.day

    # 3. 基础统计
    grp = chunk.groupby(["uuid", "poi_id"], sort=False)

    base = grp.agg(
        visit_count=("stay_minutes", "size"),
        total_stay_minutes=("stay_minutes", "sum"),
        max_stay_minutes=("stay_minutes", "max"),
        sum_sq_stay_minutes=("stay_minutes", lambda x: (x**2).sum()),
        first_visit_date=("date", "min"),
        last_visit_date=("date", "max"),
    )

    # 4. 进入小时分布
    eh = (
        chunk[chunk["start_hour"] >= 0]
        .groupby(["uuid", "poi_id", "start_hour"])
        .size()
        .unstack(fill_value=0)
    )
    eh = eh.reindex(columns=ENTER_HOURS, fill_value=0)
    eh.columns = ENTER_COLS

    # 5. 离开小时分布
    lh = (
        chunk[chunk["end_hour"] >= 0]
        .groupby(["uuid", "poi_id", "end_hour"])
        .size()
        .unstack(fill_value=0)
    )
    lh = lh.reindex(columns=LEAVE_HOURS, fill_value=0)
    lh.columns = LEAVE_COLS

    # 6. 星期分布
    wd = (
        chunk.groupby(["uuid", "poi_id", "weekday"])
        .size()
        .unstack(fill_value=0)
    )
    wd = wd.reindex(columns=WEEKDAYS, fill_value=0)
    wd.columns = WEEKDAY_COLS

    # 7. 日期分布
    dom = (
        chunk.groupby(["uuid", "poi_id", "day_of_month"])
        .size()
        .unstack(fill_value=0)
    )
    dom = dom.reindex(columns=DAYS_OF_MONTH, fill_value=0)
    dom.columns = DOM_COLS

    # 8. 合并
    agg_chunk = base.join([eh, lh, wd, dom], how="left")

    for col in ["visit_count", "total_stay_minutes", "max_stay_minutes", "sum_sq_stay_minutes"] + DIST_COLS:
        if col in agg_chunk.columns:
            agg_chunk[col] = agg_chunk[col].fillna(0)

    return agg_chunk.reset_index(), poi_coords


def _postprocess_grouped(grouped: pd.DataFrame, global_last, rhythm_specs: Optional[dict] = None) -> pd.DataFrame:
    """
    对已经按 (uuid, poi_id) 聚合好的 DataFrame 计算各种特征并返回最终列
    
    Args:
        grouped: 聚合后的 DataFrame
        global_last: 全局最大 last_visit_date
        
    Returns:
        包含最终特征列的 DataFrame
    """
    _log_status(f"开始计算统计特征（当前子集 {len(grouped):,} 行）...")

    # 1. 统计特征
    grouped["avg_stay_minutes"] = grouped["total_stay_minutes"] / grouped["visit_count"]

    n = grouped["visit_count"].astype("float64")
    mean = grouped["avg_stay_minutes"].astype("float64")
    mean_sq = (grouped["sum_sq_stay_minutes"] / n).astype("float64")
    var = (mean_sq - mean**2).clip(lower=0)
    grouped["std_stay_minutes"] = np.sqrt(var)
    grouped.loc[n <= 1, "std_stay_minutes"] = 0.0

    grouped["last_visit_recency_days"] = (global_last - grouped["last_visit_date"]).dt.days

    # 2. 比例特征
    _log_status("计算比例特征...")
    grouped["weekday_total_visits"] = grouped[[f"wd_{d}" for d in range(1, 6)]].sum(axis=1)
    grouped["weekend_total_visits"] = grouped[[f"wd_{d}" for d in range(6, 8)]].sum(axis=1)
    denom = (grouped["weekday_total_visits"] + grouped["weekend_total_visits"]).replace(0, pd.NA)
    grouped["weekday_visit_ratio"] = (grouped["weekday_total_visits"] / denom).fillna(0.0)
    grouped["weekend_visit_ratio"] = (1 - grouped["weekday_visit_ratio"]).fillna(0.0)

    day_start = 7
    day_end = 19
    if isinstance(rhythm_specs, dict):
        daytime_cfg = rhythm_specs.get("daytime", {})
        if isinstance(daytime_cfg, dict):
            day_start = int(daytime_cfg.get("start_hour", day_start))
            day_end = int(daytime_cfg.get("end_hour", day_end))

    day_start = max(0, min(23, day_start))
    day_end = max(0, min(24, day_end))
    if day_start == day_end:
        day_hours = []
    elif day_start < day_end:
        day_hours = list(range(day_start, day_end))
    else:
        day_hours = list(range(day_start, 24)) + list(range(0, day_end))

    grouped["daytime_visits"] = grouped[[f"enter_h_{h:02d}" for h in day_hours]].sum(axis=1) if day_hours else 0
    total_enter = grouped[ENTER_COLS].sum(axis=1).replace(0, pd.NA)
    grouped["daytime_ratio"] = (grouped["daytime_visits"] / total_enter).fillna(0.0)
    grouped["night_ratio"] = (1 - grouped["daytime_ratio"]).fillna(0.0)

    # 3. 分布 JSON & Top3
    _log_status("生成分布 JSON 与 Top3...")
    
    # 静默模式下禁用 pandas 进度条
    if _quiet_mode:
        pd.options.mode.chained_assignment = None
    else:
        _tqdm_pandas.pandas(desc="bucket 内 JSON 生成")

    def _hist_to_json(row, cols, label_func):
        d = {}
        for c in cols:
            v = int(row[c])
            if v > 0:
                d[str(label_func(c))] = v
        return json.dumps(d, ensure_ascii=False)

    def _topk_from_hist(row, cols, k=3, parser=lambda c: c):
        ser = pd.to_numeric(row[cols], errors="coerce").fillna(0)
        top = ser.nlargest(k)
        return json.dumps([parser(c) for c in top.index], ensure_ascii=False)

    # 根据是否静默模式选择 apply 方法
    if _quiet_mode:
        apply_func = lambda df, func: df.apply(func, axis=1)
    else:
        apply_func = lambda df, func: df.progress_apply(func, axis=1)

    # 2.5 可选：自定义节律统计（用于快速对比不同时间划分）
    grouped["enter_period_stats"] = "{}"
    grouped["leave_period_stats"] = "{}"
    grouped["weekday_group_stats"] = "{}"
    grouped["date_group_stats"] = "{}"

    def _hours_from_period_spec(p: dict) -> list[int]:
        if "hours" in p and isinstance(p["hours"], (list, tuple)):
            hours = [int(h) for h in p["hours"]]
            return [h for h in hours if 0 <= h <= 23]
        start = int(p.get("start_hour", 0))
        end = int(p.get("end_hour", 0))
        start = max(0, min(23, start))
        end = max(0, min(24, end))
        if start == end:
            return []
        if start < end:
            return list(range(start, end))
        return list(range(start, 24)) + list(range(0, end))

    def _days_from_date_group_spec(g: dict) -> list[int]:
        days: set[int] = set()
        if "days_of_month" in g and isinstance(g["days_of_month"], (list, tuple)):
            for d in g["days_of_month"]:
                try:
                    di = int(d)
                except Exception:
                    continue
                if 1 <= di <= 31:
                    days.add(di)
        if "ranges" in g and isinstance(g["ranges"], (list, tuple)):
            for r in g["ranges"]:
                if not (isinstance(r, (list, tuple)) and len(r) == 2):
                    continue
                a, b = r
                try:
                    a_dt = pd.to_datetime(a)
                    b_dt = pd.to_datetime(b)
                    a_day = int(a_dt.day)
                    b_day = int(b_dt.day)
                except Exception:
                    try:
                        a_day = int(a)
                        b_day = int(b)
                    except Exception:
                        continue
                if a_day > b_day:
                    a_day, b_day = b_day, a_day
                for di in range(a_day, b_day + 1):
                    if 1 <= di <= 31:
                        days.add(di)
        return sorted(days)

    if isinstance(rhythm_specs, dict):
        enter_periods = rhythm_specs.get("enter_periods")
        if isinstance(enter_periods, (list, tuple)) and enter_periods:
            total_enter_for_ratio = grouped[ENTER_COLS].sum(axis=1).replace(0, pd.NA)
            period_defs = []
            for p in enter_periods:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                hours = _hours_from_period_spec(p)
                cols = [f"enter_h_{h:02d}" for h in hours]
                period_defs.append((str(p["name"]), cols))

            if period_defs:
                grouped["__total_enter_for_ratio"] = total_enter_for_ratio

                def _row_enter_period_stats(r):
                    out = {}
                    denom_v = r.get("__total_enter_for_ratio")
                    for name, cols in period_defs:
                        v = int(pd.to_numeric(r[cols], errors="coerce").fillna(0).sum()) if cols else 0
                        ratio = float(v / denom_v) if denom_v is not None and not pd.isna(denom_v) else 0.0
                        out[name] = {"visits": v, "ratio": round(ratio, 6)}
                    return json.dumps(out, ensure_ascii=False)

                grouped["enter_period_stats"] = apply_func(grouped, _row_enter_period_stats)
                grouped.drop(columns=["__total_enter_for_ratio"], inplace=True, errors="ignore")

        leave_periods = rhythm_specs.get("leave_periods")
        if isinstance(leave_periods, (list, tuple)) and leave_periods:
            total_leave_for_ratio = grouped[LEAVE_COLS].sum(axis=1).replace(0, pd.NA)
            period_defs = []
            for p in leave_periods:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                hours = _hours_from_period_spec(p)
                cols = [f"leave_h_{h:02d}" for h in hours]
                period_defs.append((str(p["name"]), cols))

            if period_defs:
                grouped["__total_leave_for_ratio"] = total_leave_for_ratio

                def _row_leave_period_stats(r):
                    out = {}
                    denom_v = r.get("__total_leave_for_ratio")
                    for name, cols in period_defs:
                        v = int(pd.to_numeric(r[cols], errors="coerce").fillna(0).sum()) if cols else 0
                        ratio = float(v / denom_v) if denom_v is not None and not pd.isna(denom_v) else 0.0
                        out[name] = {"visits": v, "ratio": round(ratio, 6)}
                    return json.dumps(out, ensure_ascii=False)

                grouped["leave_period_stats"] = apply_func(grouped, _row_leave_period_stats)
                grouped.drop(columns=["__total_leave_for_ratio"], inplace=True, errors="ignore")

        weekday_groups = rhythm_specs.get("weekday_groups")
        if isinstance(weekday_groups, (list, tuple)) and weekday_groups:
            total_wd_for_ratio = grouped[WEEKDAY_COLS].sum(axis=1).replace(0, pd.NA)
            group_defs = []
            for g in weekday_groups:
                if not isinstance(g, dict) or not g.get("name"):
                    continue
                days = g.get("days")
                if not isinstance(days, (list, tuple)):
                    continue
                cols = []
                for d in days:
                    try:
                        di = int(d)
                    except Exception:
                        continue
                    if 1 <= di <= 7:
                        cols.append(f"wd_{di}")
                group_defs.append((str(g["name"]), cols))

            if group_defs:
                grouped["__total_wd_for_ratio"] = total_wd_for_ratio

                def _row_weekday_group_stats(r):
                    out = {}
                    denom_v = r.get("__total_wd_for_ratio")
                    for name, cols in group_defs:
                        v = int(pd.to_numeric(r[cols], errors="coerce").fillna(0).sum()) if cols else 0
                        ratio = float(v / denom_v) if denom_v is not None and not pd.isna(denom_v) else 0.0
                        out[name] = {"visits": v, "ratio": round(ratio, 6)}
                    return json.dumps(out, ensure_ascii=False)

                grouped["weekday_group_stats"] = apply_func(grouped, _row_weekday_group_stats)
                grouped.drop(columns=["__total_wd_for_ratio"], inplace=True, errors="ignore")

        date_groups = rhythm_specs.get("date_groups")
        if isinstance(date_groups, (list, tuple)) and date_groups:
            total_dom_for_ratio = grouped[DOM_COLS].sum(axis=1).replace(0, pd.NA)
            group_defs = []
            for g in date_groups:
                if not isinstance(g, dict) or not g.get("name"):
                    continue
                days = _days_from_date_group_spec(g)
                cols = [f"dom_{d:02d}" for d in days]
                group_defs.append((str(g["name"]), cols))

            if group_defs:
                grouped["__total_dom_for_ratio"] = total_dom_for_ratio

                def _row_date_group_stats(r):
                    out = {}
                    denom_v = r.get("__total_dom_for_ratio")
                    for name, cols in group_defs:
                        v = int(pd.to_numeric(r[cols], errors="coerce").fillna(0).sum()) if cols else 0
                        ratio = float(v / denom_v) if denom_v is not None and not pd.isna(denom_v) else 0.0
                        out[name] = {"visits": v, "ratio": round(ratio, 6)}
                    return json.dumps(out, ensure_ascii=False)

                grouped["date_group_stats"] = apply_func(grouped, _row_date_group_stats)
                grouped.drop(columns=["__total_dom_for_ratio"], inplace=True, errors="ignore")

    grouped["enter_hour_dist"] = apply_func(
        grouped, lambda r: _hist_to_json(r, ENTER_COLS, lambda c: int(c.split("_")[-1]))
    )
    grouped["leave_hour_dist"] = apply_func(
        grouped, lambda r: _hist_to_json(r, LEAVE_COLS, lambda c: int(c.split("_")[-1]))
    )
    grouped["visit_weekday_dist"] = apply_func(
        grouped, lambda r: _hist_to_json(r, WEEKDAY_COLS, lambda c: int(c.split("_")[-1]))
    )
    grouped["visit_day_of_month_dist"] = apply_func(
        grouped, lambda r: _hist_to_json(r, DOM_COLS, lambda c: int(c.split("_")[-1]))
    )

    grouped["top_enter_hours"] = apply_func(
        grouped, lambda r: _topk_from_hist(r, ENTER_COLS, k=3, parser=lambda c: int(c.split("_")[-1]))
    )
    grouped["top_leave_hours"] = apply_func(
        grouped, lambda r: _topk_from_hist(r, LEAVE_COLS, k=3, parser=lambda c: int(c.split("_")[-1]))
    )
    grouped["top_weekdays"] = apply_func(
        grouped, lambda r: _topk_from_hist(r, WEEKDAY_COLS, k=3, parser=lambda c: int(c.split("_")[-1]))
    )

    # 4. 整理输出列
    grouped["first_visit_date"] = grouped["first_visit_date"].dt.strftime("%Y-%m-%d")
    grouped["last_visit_date"] = grouped["last_visit_date"].dt.strftime("%Y-%m-%d")

    return grouped[FINAL_COLS].copy()


def _finalize_from_parts(
    parts_dir: Path,
    output_path: Path,
    num_buckets: int = 32,
    clean_buckets: bool = True,
    rhythm_specs: Optional[dict] = None,
) -> int:
    """
    流式 + 分桶 的 Stage 2：
      1) 先扫描所有 part 得到 global_last_visit_date
      2) 将所有 part 按 (uuid, poi_id) hash 分成 num_buckets 个 bucket 文件
      3) 对每个 bucket 单独 groupby + 计算特征 + 追加写入最终 parquet
    
    Args:
        parts_dir: 中间文件目录
        output_path: 最终输出文件路径
        num_buckets: 分桶数量，默认 32
        clean_buckets: 是否在完成后清理 bucket 临时文件
        
    Returns:
        最终记录数
    """
    part_files = sorted(parts_dir.glob("user_poi_part_*.parquet"))
    if not part_files:
        print(f"✗ 未在 {parts_dir} 找到任何 part 文件")
        return 0

    # 0. 先算 global_last_visit_date
    _log_status(f"在 {parts_dir} 找到 {len(part_files)} 个中间文件，先扫描最大日期...", force_print=True)
    global_last = None
    for f in _tqdm(part_files, desc="扫描 last_visit_date", unit="文件"):
        s = pd.read_parquet(f, columns=["last_visit_date"])["last_visit_date"]
        m = s.max()
        if pd.isna(m):
            continue
        if global_last is None or m > global_last:
            global_last = m

    if global_last is None:
        print("✗ 无法从中间文件中获取 last_visit_date")
        return 0

    _log_status(f"全局最大 last_visit_date = {global_last}", force_print=True)

    # 1. 分桶重排
    bucket_dir = parts_dir / "buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    _log_status(f"开始按 (uuid, poi_id) 分为 {num_buckets} 个 bucket...", force_print=True)

    for idx, f in enumerate(_tqdm(part_files, desc="分桶重排", unit="文件")):
        df = pd.read_parquet(f)

        # 对 (uuid, poi_id) 做 hash，映射到 [0, num_buckets)
        buckets = hash_pandas_object(df[["uuid", "poi_id"]], index=False) % num_buckets
        df["__bucket"] = buckets.astype("int16")

        for b, sub in df.groupby("__bucket"):
            b = int(b)
            sub = sub.drop(columns=["__bucket"])
            out_path = bucket_dir / f"bucket_{b:03d}_part_{idx:04d}.parquet"
            sub.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

        del df

    _log_status("分桶完成，开始按 bucket 聚合并写入最终文件...", force_print=True)

    # 开启静默模式，只显示主进度条
    global _quiet_mode
    _quiet_mode = True

    # 2. 对每个 bucket 独立聚合 + 写入
    writer = None
    total_records = 0

    agg_dict = {
        "visit_count": "sum",
        "total_stay_minutes": "sum",
        "max_stay_minutes": "max",
        "sum_sq_stay_minutes": "sum",
        "first_visit_date": "min",
        "last_visit_date": "max",
    }
    for col in DIST_COLS:
        agg_dict[col] = "sum"

    for b in _tqdm(range(num_buckets), desc="处理 buckets", unit="bucket"):
        bucket_files = sorted(bucket_dir.glob(f"bucket_{b:03d}_part_*.parquet"))
        if not bucket_files:
            continue

        _log_status(f"处理 bucket {b}/{num_buckets}，包含 {len(bucket_files)} 个子文件...")

        dfs = []
        for f in bucket_files:
            dfs.append(pd.read_parquet(f))
        df_b = pd.concat(dfs, ignore_index=True)
        del dfs

        _log_status(f"bucket {b} 合并后记录数: {len(df_b):,}，开始 groupby...")

        grouped_b = (
            df_b
            .groupby(["uuid", "poi_id"], sort=False)
            .agg(agg_dict)
            .reset_index()
        )
        del df_b

        _log_status(f"bucket {b} groupby 后记录数: {len(grouped_b):,}，开始特征计算...")
        final_b = _postprocess_grouped(grouped_b, global_last, rhythm_specs=rhythm_specs)
        del grouped_b

        table = pa.Table.from_pandas(final_b, preserve_index=False)
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
        writer.write_table(table)

        total_records += len(final_b)
        del final_b
        _log_status(f"bucket {b} 已写入最终文件（累计 {total_records:,} 条）")

    # 关闭静默模式
    _quiet_mode = False

    if writer is not None:
        writer.close()
        _log_status(f"✓ 最终结果已保存到: {output_path}", force_print=True)
        _log_status(f"✓ 总记录数: {total_records:,}", force_print=True)

    # 3. 清理 bucket 临时文件
    if clean_buckets:
        _log_status("清理 bucket 临时文件...")
        shutil.rmtree(bucket_dir, ignore_errors=True)
        _log_status("清理完成", force_print=True)

    return total_records


def process_large_poi_data(
    input_file: Union[str, Path],
    output_dir: Union[str, Path] = "./output",
    chunk_size: int = 1_000_000,
    total_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    output_file: str = "user_poi_analysis.parquet",
    parts_dir: Optional[Union[str, Path]] = None,
    num_buckets: int = 32,
    clean_buckets: bool = True,
    notebook: bool = False,
    clean_parts: bool = True,
    rhythm_specs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    两阶段POI数据聚合主函数（带分桶优化，支持大数据量）
    
    Args:
        input_file: 输入文件路径 (支持 .csv 和 .parquet)
        output_dir: 输出目录路径
        chunk_size: 每批处理的行数，默认100万行
        total_rows: 总行数（可选，仅用于进度条显示）
        max_rows: 最多处理的行数（可选，用于调试/抽样）
        output_file: 最终输出文件名
        parts_dir: 中间文件存放路径（可选，默认为 output_dir/user_poi_parts）
        num_buckets: Stage 2 分桶数量，默认 32（内存紧张可调大）
        clean_buckets: 是否在完成后清理 bucket 临时文件，默认 True
        notebook: 是否在 Jupyter/Colab notebook 环境运行，默认 False
        clean_parts: 是否在完成后清理 Stage 1 生成的 part 临时文件，默认 True
        rhythm_specs: 可选的节律统计配置（用于生成自定义时间段/日期段/周段对比统计）
        
    Returns:
        空 DataFrame（全量数据建议直接读取输出文件，避免内存溢出）
        
    Example:
        >>> process_large_poi_data(
        ...     input_file='track.csv',
        ...     output_dir='./output',
        ...     chunk_size=500_000,
        ...     parts_dir='/tmp/poi_parts',
        ...     num_buckets=64,  # 内存紧张时增大此值
        ...     notebook=True,   # Colab/Jupyter 环境
        ... )
        >>> # 处理完成后读取结果
        >>> df = read_result('./output/user_poi_analysis.parquet')
    """
    # 初始化 tqdm
    tqdm_func = _init_tqdm(notebook=notebook)
    
    # 初始化日志文件（与输入文件同目录）
    input_path = Path(input_file)
    log_path = input_path.parent / "aggregator_log.txt"
    _init_log(log_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 中间文件目录
    parts_dir_provided = parts_dir is not None
    if parts_dir is not None:
        parts_dir = Path(parts_dir)
    else:
        parts_dir = output_dir / "user_poi_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    file_ext = Path(input_file).suffix.lower()

    _log_status(f"开始 Stage 1 分批聚合: {input_file}", force_print=True)
    print(f"输出目录: {output_dir.resolve()}")
    print(f"中间文件目录: {parts_dir.resolve()}")
    print(f"每批处理 {chunk_size:,} 行")
    print(f"Stage 2 分桶数: {num_buckets}")
    print(f"日志文件: {log_path}")
    if total_rows:
        print(f"总行数(估计/真实): {total_rows:,}")
    if max_rows:
        print(f"本次最多处理: {max_rows:,} 行（调试/抽样）")
    print("-" * 60)

    processed_rows = 0
    part_id = 0
    pbar_total = max_rows if max_rows is not None else total_rows
    pbar = tqdm_func(total=pbar_total, desc="Stage 1 进度", unit="行", unit_scale=True)

    # 收集 POI 坐标
    poi_coords_list = []

    try:
        if file_ext == ".parquet":
            parquet_file = pq.ParquetFile(input_file)
            if not total_rows and max_rows is None:
                try:
                    total_rows = parquet_file.metadata.num_rows
                    pbar.total = total_rows
                    pbar.refresh()
                except Exception:
                    pass

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                if max_rows is not None and processed_rows >= max_rows:
                    break

                chunk_df = batch.to_pandas()

                if max_rows is not None:
                    remain = max_rows - processed_rows
                    if remain <= 0:
                        break
                    if len(chunk_df) > remain:
                        chunk_df = chunk_df.iloc[:remain].copy()

                agg_chunk, poi_coords = _aggregate_chunk(chunk_df)
                if not agg_chunk.empty:
                    part_path = parts_dir / f"user_poi_part_{part_id:04d}.parquet"
                    agg_chunk.to_parquet(part_path, engine="pyarrow", compression="snappy", index=False)
                    part_id += 1
                
                if not poi_coords.empty:
                    poi_coords_list.append(poi_coords)

                processed_rows += len(chunk_df)
                pbar.update(len(chunk_df))
        else:
            reader = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)

            for chunk_df in reader:
                if max_rows is not None and processed_rows >= max_rows:
                    break

                if max_rows is not None:
                    remain = max_rows - processed_rows
                    if remain <= 0:
                        break
                    if len(chunk_df) > remain:
                        chunk_df = chunk_df.iloc[:remain].copy()

                agg_chunk, poi_coords = _aggregate_chunk(chunk_df)
                if not agg_chunk.empty:
                    part_path = parts_dir / f"user_poi_part_{part_id:04d}.parquet"
                    agg_chunk.to_parquet(part_path, engine="pyarrow", compression="snappy", index=False)
                    part_id += 1
                
                if not poi_coords.empty:
                    poi_coords_list.append(poi_coords)

                processed_rows += len(chunk_df)
                pbar.update(len(chunk_df))

    except Exception as e:
        pbar.close()
        print(f"\n✗ Stage 1 处理出错: {e}")
        raise

    pbar.close()

    print("-" * 60)
    _log_status(f"Stage 1 完成，共生成 {part_id} 个中间文件，累计处理 {processed_rows:,} 行", force_print=True)
    
    # 保存 POI 坐标文件
    if poi_coords_list:
        _log_status("生成 POI 坐标文件...", force_print=True)
        all_poi_coords = pd.concat(poi_coords_list, ignore_index=True)
        all_poi_coords = all_poi_coords.drop_duplicates(subset=["poi_id"])
        poi_coords_path = output_dir / "poi_coordination.csv"
        all_poi_coords.to_csv(poi_coords_path, index=False)
        _log_status(f"✓ POI 坐标已保存到: {poi_coords_path}（共 {len(all_poi_coords):,} 个 POI）", force_print=True)
        del poi_coords_list, all_poi_coords
    
    _log_status("开始 Stage 2 分桶聚合...", force_print=True)

    final_output_path = output_dir / output_file
    _finalize_from_parts(
        parts_dir,
        final_output_path,
        num_buckets=num_buckets,
        clean_buckets=clean_buckets,
        rhythm_specs=rhythm_specs,
    )

    # 清理 Stage 1 part 临时文件（默认开启）
    if clean_parts:
        _log_status("清理 Stage 1 part 临时文件...", force_print=True)
        for part_file in parts_dir.glob("user_poi_part_*.parquet"):
            try:
                part_file.unlink()
            except Exception:
                pass

        # 若为默认 parts_dir 且已为空，则删除目录本身
        if not parts_dir_provided:
            try:
                parts_dir.rmdir()
            except Exception:
                pass

    # 全量数据不返回 DataFrame，避免内存溢出
    return pd.DataFrame()


def read_result(parquet_file: Union[str, Path]) -> pd.DataFrame:
    """
    读取聚合结果并解析 JSON 字段
    
    Args:
        parquet_file: parquet 文件路径
        
    Returns:
        解析后的 DataFrame，JSON 字段已转为 Python 对象
    """
    df = pd.read_parquet(parquet_file)

    json_cols = [
        "enter_hour_dist", "leave_hour_dist", "visit_weekday_dist",
        "visit_day_of_month_dist", "top_enter_hours", "top_leave_hours", "top_weekdays",
        "enter_period_stats", "leave_period_stats", "weekday_group_stats", "date_group_stats",
    ]
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(json.loads)

    return df
