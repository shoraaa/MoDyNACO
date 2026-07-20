"""Streamlit GUI for browsing DyNACO test logs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from tools.logs_explorer import parser
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.logs_explorer import parser


ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
INDEX_SCHEMA_VERSION = 5


def _dir_signature(logs_dir: Path) -> tuple[tuple[str, float, int], ...]:
    files = list(logs_dir.glob("*.txt")) + list((logs_dir / "csv").glob("*.csv"))
    return tuple(sorted((str(path.relative_to(logs_dir)), path.stat().st_mtime, path.stat().st_size) for path in files))


@st.cache_data(show_spinner="Scanning logs")
def load_index(
    logs_dir: str,
    signature: tuple[tuple[str, float, int], ...],
    schema_version: int,
) -> tuple[list[parser.RunMeta], pd.DataFrame]:
    del signature, schema_version
    runs = parser.discover_runs(logs_dir)
    rows = [parser.run_to_row(run) for run in runs]
    return runs, pd.DataFrame(rows)


def normalize_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill UI columns when Streamlit has an older cached dataframe."""
    defaults = {
        "has_per_times": False,
        "per_time_rows": 0,
        "has_iters": False,
        "has_iter_times": False,
        "iter_time_axis": None,
        "bucket_coverage": "-",
        "gap_by_bucket": "-",
        "small_time": float("nan"),
        "medium_time": float("nan"),
        "large_time": float("nan"),
        "overall_time": float("nan"),
        "small_gap_per_time": float("nan"),
        "medium_gap_per_time": float("nan"),
        "large_gap_per_time": float("nan"),
        "overall_gap_per_time": float("nan"),
    }
    df = df.copy()
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    statuses = st.sidebar.multiselect(
        "Status",
        ["finished", "interrupted", "partial"],
        default=["finished", "interrupted", "partial"],
    )
    problems = st.sidebar.multiselect("Problem", sorted(x for x in df["problem"].dropna().unique()), default=None)
    sizes = st.sidebar.multiselect("Train size", sorted(x for x in df["train_size"].dropna().unique()), default=None)
    multihead = st.sidebar.segmented_control("Multihead", ["all", "yes", "no"], default="all")
    per_times = st.sidebar.segmented_control("Per-time CSV", ["all", "yes", "no"], default="all")
    decoders = st.sidebar.multiselect("Decoder", sorted(x for x in df["decoder_type"].dropna().unique()), default=None)
    routers = st.sidebar.multiselect("Router", sorted(x for x in df["router"].dropna().unique()), default=None)
    search = st.sidebar.text_input("Search stem/dataset")

    filtered = df.copy()
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    if problems:
        filtered = filtered[filtered["problem"].isin(problems)]
    if sizes:
        filtered = filtered[filtered["train_size"].isin(sizes)]
    if multihead == "yes":
        filtered = filtered[filtered["multihead"]]
    elif multihead == "no":
        filtered = filtered[~filtered["multihead"]]
    if per_times == "yes":
        filtered = filtered[filtered["has_per_times"]]
    elif per_times == "no":
        filtered = filtered[~filtered["has_per_times"]]
    if decoders:
        filtered = filtered[filtered["decoder_type"].isin(decoders)]
    if routers:
        filtered = filtered[filtered["router"].isin(routers)]
    if search:
        haystack = filtered["stem"].fillna("") + " " + filtered["dataset"].fillna("")
        filtered = filtered[haystack.str.contains(search, case=False, regex=False)]

    timestamps = pd.to_datetime(filtered["timestamp"], errors="coerce").dropna()
    if not timestamps.empty:
        start, end = st.sidebar.date_input(
            "Date range",
            value=(timestamps.min().date(), timestamps.max().date()),
            min_value=timestamps.min().date(),
            max_value=timestamps.max().date(),
        )
        if start and end:
            ts = pd.to_datetime(filtered["timestamp"], errors="coerce")
            filtered = filtered[(ts.dt.date >= start) & (ts.dt.date <= end)]
    return filtered


def run_table(df: pd.DataFrame) -> None:
    display_cols = [
        "status",
        "problem",
        "train_size",
        "head_config",
        "dataset",
        "timestamp",
        "bucket_coverage",
        "gap_by_bucket",
        "small_gap%",
        "medium_gap%",
        "large_gap%",
        "has_per_times",
        "per_time_rows",
        "has_iters",
        "has_iter_times",
        "iter_time_axis",
        "mean_time",
        "done",
        "total",
        "stem",
    ]
    st.dataframe(
        df[display_cols].sort_values(["timestamp", "stem"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "small_gap%": st.column_config.NumberColumn("<1K gap%", format="%.3f"),
            "medium_gap%": st.column_config.NumberColumn("[1K,10K) gap%", format="%.3f"),
            "large_gap%": st.column_config.NumberColumn(">=10K gap%", format="%.3f"),
            "timestamp": st.column_config.DatetimeColumn("timestamp"),
        },
    )


def detail_view(run: parser.RunMeta) -> None:
    cfg = run.config or parser.parse_config(run.stem)
    st.subheader(run.stem)
    status_cols = st.columns(4)
    status_cols[0].metric("Status", run.status)
    status_cols[1].metric("Instances", f"{run.done_instances}/{run.total_instances or '?'}")
    status_cols[2].metric("Head config", parser.head_label(cfg))
    status_cols[3].metric("Dataset", cfg.dataset or "-")

    if cfg.is_multihead:
        st.info(
            f"Multihead: {cfg.num_heads} heads, decoder={cfg.decoder_type or '-'}, "
            f"router={cfg.router or 'static'}, init={cfg.head_init or '-'}"
        )

    show_all = st.toggle("Show all config", value=False)
    st.dataframe(parser.config_table(cfg, show_all=show_all), hide_index=True, use_container_width=True)

    instances = parser.load_instances(run)
    if instances.empty or "primary_time" not in instances or instances["primary_time"].notna().sum() == 0:
        st.warning(
            "This run's `_instances.csv` does not contain usable per-instance time snapshots "
            "such as `model_anneal_time_I1000`. Gap-per-time plots will use summary time only "
            "where available and cannot be split reliably by bucket."
        )
    perf = parser.aggregate_perf(instances, run)
    perf_df = pd.DataFrame(
        [
            {
                "bucket": bucket,
                "gap%": perf[bucket]["gap%"],
                "mean_time": perf[bucket].get("time"),
                "gap_per_time": perf[bucket].get("gap_per_time"),
                "count": perf[bucket]["count"],
            }
            for bucket in parser.BUCKET_ORDER
        ]
    )
    left, right = st.columns([1, 1])
    with left:
        st.write("Per-size performance")
        st.dataframe(perf_df, hide_index=True, use_container_width=True)
    with right:
        plot_df = perf_df[perf_df["bucket"] != "overall"]
        st.plotly_chart(px.bar(plot_df, x="bucket", y="gap%", text_auto=".2f"), use_container_width=True)

    summary = parser.parse_summary(run.summary_csv)
    if not summary["methods"].empty:
        st.write("Summary method table")
        st.dataframe(summary["methods"], hide_index=True, use_container_width=True)
    if not summary["size_gaps"].empty:
        st.write("Summary size-gap table")
        st.dataframe(summary["size_gaps"], hide_index=True, use_container_width=True)

    convergence_view(run, instances)


def convergence_view(run: parser.RunMeta, instances: pd.DataFrame) -> None:
    if not run.iters_csv:
        st.caption("No iteration CSV for this run.")
        return
    names = []
    if not instances.empty:
        for _, row in instances.head(200).iterrows():
            names.append(f"{int(row['idx'])}: {row.get('name', '')}")
    choice = st.selectbox("Convergence instance", options=names or ["0"], key=f"conv_choice_{run.stem}")
    if st.button("Load convergence", key=f"conv_btn_{run.stem}"):
        idx = choice.split(":", 1)[0] if ":" in choice else choice
        with st.spinner("Reading iteration rows for one instance"):
            iters = parser.load_iters(run, idx)
        if iters.empty:
            st.warning("No iteration rows matched that instance.")
            return
        fig_iter = go.Figure()
        fig_time = go.Figure()
        time_col = _time_axis_column(iters)
        for method, part in iters.groupby("method", dropna=False):
            name = str(method)
            if "best" in part:
                fig_iter.add_trace(go.Scatter(x=part["iter"], y=part["best"], mode="lines", name=f"{name} best"))
                if time_col:
                    fig_time.add_trace(go.Scatter(x=part[time_col], y=part["best"], mode="lines", name=f"{name} best"))
        fig_iter.update_layout(xaxis_title="iteration", yaxis_title="best cost", legend_title="series")
        st.plotly_chart(fig_iter, use_container_width=True)
        if time_col and fig_time.data:
            fig_time.update_layout(xaxis_title=f"{time_col} (s)", yaxis_title="best cost", legend_title="series")
            st.plotly_chart(fig_time, use_container_width=True)


def compare_view(runs_by_stem: dict[str, parser.RunMeta], df: pd.DataFrame) -> None:
    df = enrich_compare_flags(df, runs_by_stem)
    ctrl_cols = st.columns(2)
    only_iters = ctrl_cols[0].checkbox("Only runs with iteration CSV", value=False)
    only_iter_time = ctrl_cols[1].checkbox("Only runs with convergence time axis", value=False)
    options_df = df.copy()
    if only_iters or only_iter_time:
        options_df = options_df[options_df["has_iters"]]
    if only_iter_time:
        options_df = options_df[options_df["has_iter_times"]]

    st.caption(
        f"Compare candidates: {len(options_df)} "
        f"({int(df['has_iters'].sum())} with `_iters.csv`, "
        f"{int(df['has_iter_times'].sum())} with a time axis)"
    )
    options = options_df["stem"].tolist()
    default = options[:2] if len(options) >= 2 else options
    selected = st.multiselect("Runs to compare", options=options, default=default)
    if len(selected) < 2:
        st.info("Select at least two runs. Relax the Compare tab filters if too few runs remain.")
        return

    rows = []
    for stem in selected:
        run = runs_by_stem[stem]
        perf = parser.aggregate_perf(parser.load_instances(run), run)
        rows.append(
            {
                "stem": stem,
                "head_config": parser.head_label(run.config or parser.parse_config(stem)),
                "small_gap%": perf["small"]["gap%"],
                "medium_gap%": perf["medium"]["gap%"],
                "large_gap%": perf["large"]["gap%"],
                "overall_gap%": perf["overall"]["gap%"],
                "small_time": perf["small"]["time"],
                "medium_time": perf["medium"]["time"],
                "large_time": perf["large"]["time"],
                "overall_time": perf["overall"]["time"],
                "small_gap_per_time": perf["small"]["gap_per_time"],
                "medium_gap_per_time": perf["medium"]["gap_per_time"],
                "large_gap_per_time": perf["large"]["gap_per_time"],
                "overall_gap_per_time": perf["overall"]["gap_per_time"],
                "bucket_coverage": parser.run_to_row(run)["bucket_coverage"],
                "mean_time": perf.get("mean_time"),
                "total_time": perf.get("total_time"),
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)

    long_gap = table.melt(
        id_vars=["stem", "head_config"],
        value_vars=["small_gap%", "medium_gap%", "large_gap%", "overall_gap%"],
        var_name="bucket",
        value_name="gap%",
    )
    st.plotly_chart(px.bar(long_gap, x="bucket", y="gap%", color="head_config", barmode="group", hover_data=["stem"]), use_container_width=True)

    if table["mean_time"].notna().any():
        st.plotly_chart(px.bar(table, x="head_config", y="mean_time", hover_data=["stem"]), use_container_width=True)

    gap_time_view(table)
    compare_iters_view(runs_by_stem, selected)


def enrich_compare_flags(df: pd.DataFrame, runs_by_stem: dict[str, parser.RunMeta]) -> pd.DataFrame:
    df = df.copy()
    for col, default in [("has_iters", False), ("has_iter_times", False), ("iter_time_axis", None)]:
        if col not in df.columns:
            df[col] = default
    for idx, row in df.iterrows():
        stem = row.get("stem")
        run = runs_by_stem.get(stem)
        if not run:
            continue
        axis = row.get("iter_time_axis")
        if pd.isna(axis) or axis in ("", None):
            axis = get_iters_time_axis(run)
            df.at[idx, "iter_time_axis"] = axis
        df.at[idx, "has_iters"] = bool(run.iters_csv)
        df.at[idx, "has_iter_times"] = axis is not None
    return df


def get_iters_time_axis(run: parser.RunMeta) -> str | None:
    if hasattr(parser, "iters_time_axis"):
        return parser.iters_time_axis(run)
    if not run.iters_csv:
        return None
    try:
        header = pd.read_csv(run.iters_csv, nrows=0).columns.tolist()
    except Exception:
        return None
    for col in ["elapsed_s", "outer_elapsed_s", "t"]:
        if col in header:
            return col
    return None


def gap_time_view(table: pd.DataFrame) -> None:
    st.subheader("Gap per time")
    missing_timed = table[["small_time", "medium_time", "large_time", "overall_time"]].isna().all(axis=1)
    if missing_timed.any():
        st.warning(
            f"{int(missing_timed.sum())} selected run(s) do not have usable per-instance `*_time_I...` "
            "columns. Their bucket-level gap-per-time values are unavailable."
        )
    time_cols = ["small_time", "medium_time", "large_time", "overall_time"]
    gap_time_cols = ["small_gap_per_time", "medium_gap_per_time", "large_gap_per_time", "overall_gap_per_time"]
    value_cols = [
        "small_gap%",
        "medium_gap%",
        "large_gap%",
        "overall_gap%",
        *time_cols,
        *gap_time_cols,
    ]
    shown_cols = ["head_config", "bucket_coverage", *value_cols, "stem"]
    st.dataframe(table[shown_cols], hide_index=True, use_container_width=True)

    long_gap = table.melt(
        id_vars=["stem", "head_config"],
        value_vars=["small_gap%", "medium_gap%", "large_gap%", "overall_gap%"],
        var_name="bucket",
        value_name="gap%",
    )
    long_time = table.melt(
        id_vars=["stem", "head_config"],
        value_vars=time_cols,
        var_name="bucket",
        value_name="bucket_time",
    )
    long_eff = table.melt(
        id_vars=["stem", "head_config"],
        value_vars=gap_time_cols,
        var_name="bucket",
        value_name="gap_per_time",
    )
    long_gap["bucket"] = long_gap["bucket"].str.replace("_gap%", "", regex=False)
    long_time["bucket"] = long_time["bucket"].str.replace("_time", "", regex=False)
    long_eff["bucket"] = long_eff["bucket"].str.replace("_gap_per_time", "", regex=False)
    merged = long_gap.merge(long_time, on=["stem", "head_config", "bucket"]).merge(
        long_eff, on=["stem", "head_config", "bucket"]
    )
    merged = merged.dropna(subset=["gap%", "bucket_time"])
    if merged.empty:
        st.caption("No per-instance timing columns were available for the selected runs.")
        return

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.scatter(
                merged,
                x="bucket_time",
                y="gap%",
                color="head_config",
                symbol="bucket",
                hover_data=["stem", "bucket", "gap_per_time"],
                labels={"bucket_time": "bucket mean time (s)", "gap%": "gap (%)"},
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.bar(
                merged.dropna(subset=["gap_per_time"]),
                x="bucket",
                y="gap_per_time",
                color="head_config",
                barmode="group",
                hover_data=["stem", "bucket_time", "gap%"],
                labels={"gap_per_time": "gap % per second"},
            ),
            use_container_width=True,
        )


def compare_iters_view(runs_by_stem: dict[str, parser.RunMeta], selected: list[str]) -> None:
    st.divider()
    st.subheader("Per-iteration convergence comparison")
    runs = [runs_by_stem[stem] for stem in selected if runs_by_stem[stem].iters_csv]
    if len(runs) < 2:
        st.caption("Select at least two runs with `_iters.csv` files.")
        return

    mode = st.segmented_control("Convergence scope", ["single instance", "average"], default="single instance")
    methods = st.multiselect("Method filter", ["Model", "Base", "Mix"], default=["Model"])
    if mode == "average":
        compare_average_iters_view(runs, methods)
        return

    bucket_choice = st.segmented_control(
        "Instance bucket",
        ["all", "small (<1K)", "medium ([1K,10K))", "large (>=10K)"],
        default="all",
    )
    bucket_key = {
        "all": None,
        "small (<1K)": "small",
        "medium ([1K,10K))": "medium",
        "large (>=10K)": "large",
    }[bucket_choice]
    instance_options = _shared_instance_options(runs, bucket_key)
    if not instance_options:
        st.caption("No shared instance names were found for that bucket in the selected runs' instance CSVs.")
        return
    choice = st.selectbox("Shared instance", options=instance_options)
    if not st.button("Load per-iteration overlay"):
        return

    idx = choice.split(":", 1)[0]
    fig_iter = go.Figure()
    fig_time = go.Figure()
    with st.spinner("Reading selected iteration traces"):
        for run in runs:
            iters = parser.load_iters(run, idx)
            if iters.empty or "best" not in iters:
                continue
            time_col = _time_axis_column(iters)
            if methods and "method" in iters:
                mask = False
                for method in methods:
                    mask = mask | iters["method"].astype(str).str.contains(method, case=False, na=False)
                iters = iters[mask]
            if iters.empty:
                continue
            label = parser.head_label(run.config or parser.parse_config(run.stem))
            for method_name, part in iters.groupby("method", dropna=False):
                fig_iter.add_trace(
                    go.Scatter(
                        x=part["iter"],
                        y=part["best"],
                        mode="lines",
                        name=f"{label} {method_name}: {run.stem[-19:]}",
                    )
                )
                if time_col:
                    fig_time.add_trace(
                        go.Scatter(
                            x=part[time_col],
                            y=part["best"],
                            mode="lines",
                            name=f"{label} {method_name}: {run.stem[-19:]}",
                        )
                    )
    if fig_iter.data:
        fig_iter.update_layout(xaxis_title="iteration", yaxis_title="best cost", legend_title="run")
        st.plotly_chart(fig_iter, use_container_width=True)
        if fig_time.data:
            fig_time.update_layout(xaxis_title="time (s)", yaxis_title="best cost", legend_title="run")
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.warning("No matching iteration rows were found for that instance/method filter.")


def compare_average_iters_view(runs: list[parser.RunMeta], methods: list[str]) -> None:
    scope = st.segmented_control(
        "Average scope",
        ["all instances", "small (<1K)", "medium ([1K,10K))", "large (>=10K)"],
        default="all instances",
    )
    bucket_key = {
        "all instances": None,
        "small (<1K)": "small",
        "medium ([1K,10K))": "medium",
        "large (>=10K)": "large",
    }[scope]
    if not st.button("Load averaged convergence overlay"):
        return

    fig_iter = go.Figure()
    fig_time = go.Figure()
    with st.spinner("Averaging selected iteration traces"):
        for run in runs:
            indices = _bucket_indices(run, bucket_key)
            if indices is not None and not indices:
                continue
            avg = parser.load_iters_average(run, tuple(indices) if indices is not None else None)
            if avg.empty or "best" not in avg:
                continue
            if methods and "method" in avg:
                mask = False
                for method in methods:
                    mask = mask | avg["method"].astype(str).str.contains(method, case=False, na=False)
                avg = avg[mask]
            if avg.empty:
                continue
            label = parser.head_label(run.config or parser.parse_config(run.stem))
            for method_name, part in avg.groupby("method", dropna=False):
                fig_iter.add_trace(
                    go.Scatter(
                        x=part["iter"],
                        y=part["best"],
                        mode="lines",
                        name=f"{label} {method_name}: {run.stem[-19:]}",
                    )
                )
                if "time" in part and pd.to_numeric(part["time"], errors="coerce").notna().any():
                    fig_time.add_trace(
                        go.Scatter(
                            x=part["time"],
                            y=part["best"],
                            mode="lines",
                            name=f"{label} {method_name}: {run.stem[-19:]}",
                        )
                    )
    if fig_iter.data:
        fig_iter.update_layout(xaxis_title="iteration", yaxis_title=f"average best cost ({scope})", legend_title="run")
        st.plotly_chart(fig_iter, use_container_width=True)
        if fig_time.data:
            fig_time.update_layout(xaxis_title="average time (s)", yaxis_title=f"average best cost ({scope})", legend_title="run")
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.warning("No averaged iteration rows were available for that scope/method filter.")


def _bucket_indices(run: parser.RunMeta, bucket: str | None) -> list[int] | None:
    if bucket is None:
        return None
    df = parser.load_instances(run)
    if df.empty or "idx" not in df or "bucket" not in df:
        return []
    return [int(idx) for idx in df.loc[df["bucket"] == bucket, "idx"].dropna().tolist()]


def _time_axis_column(df: pd.DataFrame) -> str | None:
    for col in ["elapsed_s", "outer_elapsed_s", "t"]:
        if col in df and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def _shared_instance_options(runs: list[parser.RunMeta], bucket: str | None = None) -> list[str]:
    shared: set[tuple[int, str, str]] | None = None
    for run in runs:
        df = parser.load_instances(run)
        if df.empty or "idx" not in df or "name" not in df:
            shared = set()
            break
        if bucket and "bucket" in df:
            df = df[df["bucket"] == bucket]
        if df.empty:
            shared = set()
            break
        cols = ["idx", "name", "bucket"] if "bucket" in df else ["idx", "name"]
        pairs = set()
        for _, row in df[cols].dropna().iterrows():
            label = str(row["bucket"]) if "bucket" in row else "unknown"
            pairs.add((int(row["idx"]), str(row["name"]), label))
        shared = pairs if shared is None else shared & pairs
    if not shared:
        return []
    return [f"{idx}: {name} [{label}]" for idx, name, label in sorted(shared)[:200]]


def main() -> None:
    st.set_page_config(page_title="DyNACO Logs Explorer", layout="wide")
    st.title("DyNACO Logs Explorer")

    logs_dir = Path(st.sidebar.text_input("Logs directory", str(LOGS_DIR))).expanduser()
    if not logs_dir.exists():
        st.error(f"Logs directory not found: {logs_dir}")
        return

    runs, df = load_index(str(logs_dir), _dir_signature(logs_dir), INDEX_SCHEMA_VERSION)
    df = normalize_index_df(df)
    if df.empty:
        st.warning("No runs found.")
        return

    filtered = apply_filters(df)
    st.caption(f"{len(filtered)} of {len(df)} runs shown")
    runs_by_stem = {run.stem: run for run in runs}

    tab_runs, tab_detail, tab_compare = st.tabs(["Runs", "Detail", "Compare"])
    with tab_runs:
        run_table(filtered)
    with tab_detail:
        choices = filtered["stem"].tolist()
        only_timed = st.checkbox("Only runs with per-time CSV logging", value=False)
        if only_timed:
            choices = filtered[filtered["has_per_times"]]["stem"].tolist()
        if choices:
            stem = st.selectbox("Run", options=choices)
            detail_view(runs_by_stem[stem])
        else:
            st.info("No run matches the active filters.")
    with tab_compare:
        compare_view(runs_by_stem, filtered)


if __name__ == "__main__":
    main()
