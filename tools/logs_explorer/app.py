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


def _dir_signature(logs_dir: Path) -> tuple[tuple[str, float, int], ...]:
    files = list(logs_dir.glob("*.txt")) + list((logs_dir / "csv").glob("*.csv"))
    return tuple(sorted((str(path.relative_to(logs_dir)), path.stat().st_mtime, path.stat().st_size) for path in files))


@st.cache_data(show_spinner="Scanning logs")
def load_index(logs_dir: str, signature: tuple[tuple[str, float, int], ...]) -> tuple[list[parser.RunMeta], pd.DataFrame]:
    del signature
    runs = parser.discover_runs(logs_dir)
    rows = [parser.run_to_row(run) for run in runs]
    return runs, pd.DataFrame(rows)


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
        "overall_gap%",
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
            "overall_gap%": st.column_config.NumberColumn("overall gap%", format="%.3f"),
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
    perf = parser.aggregate_perf(instances, run)
    perf_df = pd.DataFrame(
        [
            {"bucket": bucket, "gap%": perf[bucket]["gap%"], "count": perf[bucket]["count"]}
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
        fig = go.Figure()
        for method, part in iters.groupby("method", dropna=False):
            name = str(method)
            if "mean" in part:
                fig.add_trace(go.Scatter(x=part["iter"], y=part["mean"], mode="lines", name=f"{name} mean"))
            if "best" in part:
                fig.add_trace(go.Scatter(x=part["iter"], y=part["best"], mode="lines", name=f"{name} best"))
        fig.update_layout(xaxis_title="iteration", yaxis_title="cost", legend_title="series")
        st.plotly_chart(fig, use_container_width=True)


def compare_view(runs_by_stem: dict[str, parser.RunMeta], df: pd.DataFrame) -> None:
    options = df["stem"].tolist()
    default = options[:2] if len(options) >= 2 else options
    selected = st.multiselect("Runs to compare", options=options, default=default)
    if len(selected) < 2:
        st.info("Select at least two runs.")
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

    shared_iters = [runs_by_stem[stem] for stem in selected if runs_by_stem[stem].iters_csv]
    if len(shared_iters) >= 2 and st.button("Overlay first-instance convergence"):
        fig = go.Figure()
        for run in shared_iters:
            iters = parser.load_iters(run, 0)
            if iters.empty:
                continue
            label = parser.head_label(run.config or parser.parse_config(run.stem))
            best = iters[iters["method"].astype(str).str.contains("Model", na=False)] if "method" in iters else iters
            if not best.empty and "best" in best:
                fig.add_trace(go.Scatter(x=best["iter"], y=best["best"], mode="lines", name=f"{label}: {run.stem[-28:]}"))
        if fig.data:
            fig.update_layout(xaxis_title="iteration", yaxis_title="best cost")
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="DyNACO Logs Explorer", layout="wide")
    st.title("DyNACO Logs Explorer")

    logs_dir = Path(st.sidebar.text_input("Logs directory", str(LOGS_DIR))).expanduser()
    if not logs_dir.exists():
        st.error(f"Logs directory not found: {logs_dir}")
        return

    runs, df = load_index(str(logs_dir), _dir_signature(logs_dir))
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
        if choices:
            stem = st.selectbox("Run", options=choices)
            detail_view(runs_by_stem[stem])
        else:
            st.info("No run matches the active filters.")
    with tab_compare:
        compare_view(runs_by_stem, filtered)


if __name__ == "__main__":
    main()
