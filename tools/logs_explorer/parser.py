"""Pure parsing helpers for the DyNACO logs explorer."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


SIZE_BUCKETS = {
    "<1K": "small",
    "[1K,10K)": "medium",
    ">=10K": "large",
}
BUCKET_ORDER = ["small", "medium", "large", "overall"]
DECODER_TYPES = {"lora", "deep_lora", "film", "multi_decoder", "lowrank", "polynet"}
INTERRUPT_MARKERS = ("KeyboardInterrupt", "Traceback", "OutOfMemoryError", "CUDA out of memory")
SUMMARY_METHODS = ("Model(anneal)", "Model(no anneal)", "Base", "Mix(anneal)", "Mix(no anneal)")


@dataclass(frozen=True)
class RunConfig:
    stem: str
    problem: str | None = None
    n_node: int | None = None
    k_sparse: int | None = None
    n_ants: int | None = None
    H: int | None = None
    mini_H: int | None = None
    rho: float | None = None
    mne: int | None = None
    algo: str | None = None
    lr: float | None = None
    edge_feature_set: str | None = None
    is_multihead: bool = False
    num_heads: int = 1
    decoder_type: str | None = None
    router: str | None = None
    head_init: str | None = "anchored"
    head_ant_weights: str | None = None
    dataset: str | None = None
    annealing: bool | None = None
    timestamp: datetime | None = None
    checkpoint: str | None = None
    extra_flags: list[str] = field(default_factory=list)
    default_fields: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class RunMeta:
    stem: str
    txt: str | None = None
    summary_csv: str | None = None
    instances_csv: str | None = None
    iters_csv: str | None = None
    config: RunConfig | None = None
    status: str = "partial"
    done_instances: int = 0
    total_instances: int | None = None
    mtime: float = 0.0


def discover_runs(logs_dir: str | Path) -> list[RunMeta]:
    logs_path = Path(logs_dir)
    groups: dict[str, dict[str, Path]] = {}

    for path in logs_path.glob("*.txt"):
        groups.setdefault(path.stem, {})["txt"] = path

    csv_dir = logs_path / "csv"
    for path in csv_dir.glob("*.csv"):
        stem, kind = _split_csv_stem(path.stem)
        groups.setdefault(stem, {})[kind] = path

    runs: list[RunMeta] = []
    for stem, paths in sorted(groups.items(), key=lambda kv: kv[0]):
        path_strings = {key: str(value) for key, value in paths.items()}
        mtime = max((p.stat().st_mtime for p in paths.values() if p.exists()), default=0.0)
        config = parse_config(_checkpoint_or_stem(stem, path_strings.get("summary_csv")))
        total = _loaded_instance_count(path_strings.get("txt"))
        done = _count_csv_rows(path_strings.get("instances_csv"))
        provisional = RunMeta(
            stem=stem,
            txt=path_strings.get("txt"),
            summary_csv=path_strings.get("summary_csv"),
            instances_csv=path_strings.get("instances_csv"),
            iters_csv=path_strings.get("iters_csv"),
            config=config,
            done_instances=done,
            total_instances=total,
            mtime=mtime,
        )
        runs.append(
            RunMeta(
                **{
                    **asdict(provisional),
                    "config": config,
                    "status": classify_status(provisional),
                }
            )
        )
    return runs


def parse_config(checkpoint_or_stem: str | Path | None) -> RunConfig:
    raw = "" if checkpoint_or_stem is None else str(checkpoint_or_stem)
    checkpoint_raw, test_stem = raw.split("|", 1) if "|" in raw else (raw, raw)
    stem = _logical_stem(checkpoint_raw) if checkpoint_raw else ""
    if stem.endswith("_best") or stem.endswith("_last"):
        model_stem = stem.rsplit("_", 1)[0]
    else:
        model_stem = _extract_checkpoint_stem(stem)

    test_meta = _parse_test_stem(test_stem)
    tokens = model_stem.split("_") if model_stem else []
    values: dict[str, Any] = {
        "stem": stem,
        "problem": test_meta.get("problem"),
        "dataset": test_meta.get("dataset"),
        "timestamp": test_meta.get("timestamp"),
        "annealing": test_meta.get("annealing"),
        "checkpoint": raw if raw else None,
        "num_heads": 1,
        "head_init": "anchored",
        "extra_flags": [],
    }

    i = 0
    if tokens and tokens[0] in {"tsp", "cvrp"}:
        values["problem"] = tokens[0]
        i = 1
    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None
        if (match := re.fullmatch(r"n(\d+)", token)):
            values["n_node"] = int(match.group(1))
        elif (match := re.fullmatch(r"k(\d+)", token)):
            values["k_sparse"] = int(match.group(1))
        elif (match := re.fullmatch(r"ants(\d+)", token)):
            values["n_ants"] = int(match.group(1))
        elif (match := re.fullmatch(r"H(\d+)", token)):
            values["H"] = int(match.group(1))
        elif (match := re.fullmatch(r"miniH(\d+)", token)):
            values["mini_H"] = int(match.group(1))
        elif (match := re.fullmatch(r"rho(.+)", token)):
            values["rho"] = _to_float(match.group(1))
        elif (match := re.fullmatch(r"mne(\d+)", token)):
            values["mne"] = int(match.group(1))
        elif token in {"ppo", "reinforce"}:
            values["algo"] = token
        elif token == "lr" and next_token is not None:
            values["lr"] = _to_float(next_token)
            i += 1
        elif token.startswith("lr") and len(token) > 2:
            values["lr"] = _to_float(token[2:])
        elif token in {"compact", "compact2", "compact3", "full"}:
            values["edge_feature_set"] = token
        elif (match := re.fullmatch(r"mh(\d+)", token)):
            values["num_heads"] = int(match.group(1))
        elif token in DECODER_TYPES:
            if values.get("decoder_type") == "polynet" and token != "polynet":
                values["extra_flags"].append(token)
            else:
                values["decoder_type"] = token
        elif token == "deep" and next_token == "lora":
            values["decoder_type"] = "deep_lora"
            i += 1
        elif token.startswith("hr") and len(token) > 2:
            values["router"] = token[2:]
        elif token == "hreta":
            values["router"] = "eta"
        elif token.startswith("hi") and len(token) > 2:
            values["head_init"] = token[2:]
        elif token.startswith("ha") and len(token) > 2:
            values["head_ant_weights"] = token[2:]
        elif token in {"anchored", "anchor", "balanced", "balanced_anchor"} or token.startswith("anchor"):
            values["head_init"] = "anchored"
            values["extra_flags"].append(token)
        elif token in {"best", "last"}:
            pass
        elif token == "nockpt":
            values["extra_flags"].append(token)
        elif _is_known_nonconfig_token(token):
            values["extra_flags"].append(token)
        else:
            values["extra_flags"].append(token)
        i += 1

    values["is_multihead"] = int(values.get("num_heads") or 1) > 1
    if values["is_multihead"] and not values.get("decoder_type"):
        values["decoder_type"] = "polynet"
    values["default_fields"] = _default_flags(values)
    return RunConfig(**values)


def classify_status(run: RunMeta) -> str:
    if run.txt and _tail_has_interrupt(run.txt):
        return "interrupted"
    total = run.total_instances if run.total_instances is not None else _loaded_instance_count(run.txt)
    done = run.done_instances or _count_csv_rows(run.instances_csv)
    if run.summary_csv and total is not None and done == total and done > 0:
        return "finished"
    return "partial"


def load_instances(run: RunMeta) -> pd.DataFrame:
    if not run.instances_csv:
        return pd.DataFrame()
    return _load_instances_cached(run.instances_csv, _mtime(run.instances_csv))


def aggregate_perf(df: pd.DataFrame, run: RunMeta | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if df.empty or "gap%" not in df.columns:
        for bucket in BUCKET_ORDER:
            result[bucket] = {"gap%": math.nan, "count": 0}
    else:
        bucket_series = df["bucket"] if "bucket" in df.columns else df.get("size_group", pd.Series(dtype=str)).map(SIZE_BUCKETS)
        for bucket in BUCKET_ORDER[:-1]:
            part = df[bucket_series == bucket]
            result[bucket] = {"gap%": _safe_mean(part["gap%"]), "count": int(len(part))}
        result["overall"] = {"gap%": _safe_mean(df["gap%"]), "count": int(len(df))}

    if run and run.summary_csv:
        summary = parse_summary(run.summary_csv)
        method = _primary_summary_method(summary.get("methods", pd.DataFrame()))
        if method:
            result["mean_time"] = _parse_time_value(method.get("MeanTime"))
            result["total_time"] = _parse_time_value(method.get("TotalTime"))
            result["summary_gap"] = _to_float(method.get("Gap%"))
    return result


def load_iters(run: RunMeta, instance: str | int | None = None) -> pd.DataFrame:
    if not run.iters_csv:
        return pd.DataFrame()
    usecols = [
        "idx",
        "name",
        "method",
        "anneal",
        "iter",
        "t",
        "mean",
        "best",
        "mean_before_ls",
        "mean_after_ls",
        "incumbent_after_aco",
        "best_head",
    ]
    header = pd.read_csv(run.iters_csv, nrows=0).columns.tolist()
    usecols = [col for col in usecols if col in header]
    filters: dict[str, Any] = {}
    if instance not in (None, ""):
        if isinstance(instance, int) or str(instance).isdigit():
            filters["idx"] = int(instance)
        else:
            filters["name"] = str(instance)
    return _load_iters_cached(run.iters_csv, _mtime(run.iters_csv), tuple(usecols), tuple(filters.items()))


def parse_summary(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"meta": {}, "methods": pd.DataFrame(), "size_gaps": pd.DataFrame()}
    return _parse_summary_cached(str(path), _mtime(path))


def run_to_row(run: RunMeta) -> dict[str, Any]:
    cfg = run.config or parse_config(run.stem)
    perf = aggregate_perf(load_instances(run), run)
    return {
        "stem": run.stem,
        "status": run.status,
        "problem": cfg.problem,
        "train_size": cfg.n_node,
        "head_config": head_label(cfg),
        "multihead": cfg.is_multihead,
        "num_heads": cfg.num_heads,
        "decoder_type": cfg.decoder_type,
        "router": cfg.router,
        "dataset": cfg.dataset,
        "timestamp": cfg.timestamp,
        "overall_gap%": perf["overall"]["gap%"],
        "small_gap%": perf["small"]["gap%"],
        "medium_gap%": perf["medium"]["gap%"],
        "large_gap%": perf["large"]["gap%"],
        "mean_time": perf.get("mean_time"),
        "done": run.done_instances,
        "total": run.total_instances,
        "txt": run.txt,
        "summary_csv": run.summary_csv,
        "instances_csv": run.instances_csv,
        "iters_csv": run.iters_csv,
    }


def config_table(config: RunConfig, show_all: bool = False) -> pd.DataFrame:
    rows = []
    defaults = config.default_fields or {}
    hidden = {"stem", "checkpoint", "default_fields", "extra_flags"}
    for key, value in asdict(config).items():
        if key in hidden:
            continue
        is_default = defaults.get(key, False)
        if not show_all and is_default:
            continue
        rows.append({"field": key, "value": value, "default": is_default})
    if config.extra_flags and show_all:
        rows.append({"field": "extra_flags", "value": ", ".join(config.extra_flags), "default": False})
    return pd.DataFrame(rows)


def head_label(config: RunConfig) -> str:
    if not config.is_multihead:
        return "single"
    label = f"MHx{config.num_heads}"
    detail = "/".join(part for part in [config.decoder_type, config.router] if part)
    return f"{label} {detail}" if detail else label


def _split_csv_stem(stem: str) -> tuple[str, str]:
    for suffix, kind in (("_summary", "summary_csv"), ("_instances", "instances_csv"), ("_iters", "iters_csv")):
        if stem.endswith(suffix):
            return stem[: -len(suffix)], kind
    return stem, "csv"


def _logical_stem(value: str) -> str:
    if value.endswith((".pt", ".txt", ".csv")):
        return Path(value).stem
    return Path(value).name


def _checkpoint_or_stem(stem: str, summary_csv: str | None) -> str:
    summary = parse_summary(summary_csv)
    checkpoint = summary.get("meta", {}).get("checkpoint")
    if checkpoint:
        return f"{checkpoint}|{stem}"
    return stem


def _extract_checkpoint_stem(stem: str) -> str:
    if stem.startswith("test_"):
        match = re.match(r"^test_(tsp|cvrp)_(\d+)_(.*)_\d{8}_\d{6}_annealing(?:True|False)$", stem)
        if match:
            body = match.group(3)
            if body.startswith(("tsp_", "cvrp_", "nockpt")):
                dataset_match = re.search(r"_(TSP|CVRP|CVRPlib|STAR|Li_|survey|data)", body)
                if dataset_match:
                    return body[: dataset_match.start()]
                return body
    return stem


def _parse_test_stem(stem: str) -> dict[str, Any]:
    match = re.match(r"^test_(tsp|cvrp)_(\d+)_(.*)_(\d{8})_(\d{6})_annealing(True|False)$", stem)
    if not match:
        return {}
    body = match.group(3)
    dataset = None
    config_part = body
    for marker in ("_TSP", "_CVRPlib", "_CVRP", "_STAR", "_Li_"):
        idx = body.find(marker)
        if idx >= 0:
            config_part = body[:idx]
            dataset = body[idx + 1 :]
            break
    timestamp = datetime.strptime(f"{match.group(4)}_{match.group(5)}", "%Y%m%d_%H%M%S")
    return {
        "problem": match.group(1),
        "n_node": int(match.group(2)),
        "config_part": config_part,
        "dataset": dataset,
        "timestamp": timestamp,
        "annealing": match.group(6) == "True",
    }


def _default_flags(values: dict[str, Any]) -> dict[str, bool]:
    problem = values.get("problem") or "tsp"
    defaults = default_config(problem)
    flags: dict[str, bool] = {}
    for key, default in defaults.items():
        value = values.get(key)
        if isinstance(default, float):
            flags[key] = value is not None and math.isclose(float(value), default, rel_tol=1e-9, abs_tol=1e-12)
        else:
            flags[key] = value == default
    flags["is_multihead"] = values.get("is_multihead") is False
    return flags


@lru_cache(maxsize=8)
def default_config(problem: str = "tsp") -> dict[str, Any]:
    path = Path("configs") / problem / "n1000_100.yaml"
    defaults = {
        "n_node": 1000,
        "k_sparse": 32,
        "n_ants": 100,
        "H": 10,
        "mini_H": 100,
        "rho": 0.5,
        "mne": 12,
        "algo": "ppo",
        "lr": 5e-6,
        "edge_feature_set": "compact3",
        "num_heads": 1,
        "head_init": "anchored",
        "head_ant_weights": None,
        "annealing": True,
    }
    if path.exists():
        for line in path.read_text().splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, raw = line.split(":", 1)
            key = key.strip()
            raw = raw.strip().strip("\"'")
            mapped = {"min_new_edges": "mne"}.get(key, key)
            if mapped in defaults:
                defaults[mapped] = _parse_scalar(raw)
    return defaults


@lru_cache(maxsize=256)
def _parse_summary_cached(path: str, mtime: float) -> dict[str, Any]:
    del mtime
    meta: dict[str, str] = {}
    methods: list[list[str]] = []
    size_rows: list[list[str]] = []
    section: str | None = "meta"
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                section = None
                continue
            if row[0] == "Method" and len(row) > 1 and row[1] == "MeanCost":
                section = "methods_header"
                method_header = row
                continue
            if row[0] == "Mean gap by instance size":
                section = "size_title"
                continue
            if section == "meta" and len(row) >= 2:
                meta[row[0]] = row[1]
            elif section == "methods_header":
                methods.append(row)
            elif section == "size_title" and row[0] == "Method":
                section = "size_rows"
                size_header = row
            elif section == "size_rows":
                if len(row) > 1:
                    size_rows.append(row)
    method_df = pd.DataFrame(methods, columns=locals().get("method_header", []))
    size_df = pd.DataFrame(size_rows, columns=locals().get("size_header", []))
    return {"meta": meta, "methods": method_df, "size_gaps": size_df}


@lru_cache(maxsize=256)
def _load_instances_cached(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    df = pd.read_csv(path)
    cost_col = _first_existing(df, ["model_anneal", "model_no_anneal", "base"])
    if not cost_col:
        return df
    ref_col = "opt" if "opt" in df.columns and pd.to_numeric(df["opt"], errors="coerce").notna().any() else "baseline"
    costs = pd.to_numeric(df[cost_col], errors="coerce")
    refs = pd.to_numeric(df.get(ref_col), errors="coerce")
    valid = refs.notna() & (refs.abs() > 1e-12)
    df = df.copy()
    df["primary_cost"] = costs
    df["gap%"] = ((costs - refs) / refs * 100.0).where(valid)
    if "size_group" not in df.columns and "size" in df.columns:
        size = pd.to_numeric(df["size"], errors="coerce")
        df["size_group"] = pd.cut(size, bins=[-math.inf, 999, 9999, math.inf], labels=["<1K", "[1K,10K)", ">=10K"])
    df["bucket"] = df.get("size_group", pd.Series(dtype=str)).map(SIZE_BUCKETS)
    return df


@lru_cache(maxsize=32)
def _load_iters_cached(path: str, mtime: float, usecols: tuple[str, ...], filters: tuple[tuple[str, Any], ...]) -> pd.DataFrame:
    del mtime
    filters_dict = dict(filters)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=list(usecols), chunksize=200_000):
        for key, value in filters_dict.items():
            if key in chunk.columns:
                chunk = chunk[chunk[key] == value]
        chunks.append(chunk)
        if filters_dict and len(pd.concat(chunks, ignore_index=True)) >= 50_000:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=list(usecols))


def _first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    return next((col for col in cols if col in df.columns), None)


def _primary_summary_method(methods: pd.DataFrame) -> dict[str, Any] | None:
    if methods.empty:
        return None
    for name in SUMMARY_METHODS:
        rows = methods[methods["Method"] == name]
        if not rows.empty:
            return rows.iloc[0].to_dict()
    return methods.iloc[0].to_dict()


def _count_csv_rows(path: str | None) -> int:
    if not path or not Path(path).exists():
        return 0
    with open(path, newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _loaded_instance_count(path: str | None) -> int | None:
    if not path or not Path(path).exists():
        return None
    text = Path(path).read_text(errors="replace")
    matches = re.findall(r"Loaded\s+(\d+)\s+instances", text)
    return int(matches[-1]) if matches else None


def _tail_has_interrupt(path: str) -> bool:
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 65536))
        tail = handle.read().decode(errors="replace")
    return any(marker in tail for marker in INTERRUPT_MARKERS)


def _safe_mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if pd.notna(value) else math.nan


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time_value(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(value), re.IGNORECASE)
    return _to_float(match.group(0)) if match else None


def _parse_scalar(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"none", "null", ""}:
        return None
    number = _to_float(raw)
    if number is None:
        return raw
    return int(number) if number.is_integer() and not re.search(r"[.eE]", raw) else number


def _mtime(path: str | Path | None) -> float:
    return Path(path).stat().st_mtime if path and Path(path).exists() else 0.0


def _is_known_nonconfig_token(token: str) -> bool:
    prefixes = (
        "lora",
        "r",
        "alcoef",
        "alent",
        "alt",
        "am",
        "hs",
        "hg",
        "hd",
        "ht",
        "seed",
        "cap",
        "anneal",
        "warmup",
        "L",
        "ls",
    )
    return token in {"static", "nosmooth", "noheu", "nols", "mmas", "deepaco"} or token.startswith(prefixes)
