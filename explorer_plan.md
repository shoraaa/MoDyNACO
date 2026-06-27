# Plan: Logs Explorer GUI for `logs/`

## Context

DyNACO test runs (`test.py`) emit evaluation artifacts into `logs/` as a flat set of files
per run. There is no convenient way to browse them, tell which runs finished vs. were
interrupted (Ctrl-C / OOM are common), or compare configurations against each other by
performance. Runs are also numerous (98 `.txt` logs, ~140 CSVs) and many are partial.

This plan builds a **Streamlit** app that scans `logs/`, groups files into runs, parses each
run's config (with first-class multihead vs. single-head detection per `train.py`'s naming
convention), classifies finished/unfinished, and lets the user filter, inspect, and compare
runs' performance (gap & time) bucketed into **small (<1K)**, **medium ([1K,10K))**,
**large (>=10K)**, and **overall** — via both tables and plots, including lazy-loaded
per-iteration convergence plots.

## Key facts established during exploration

- **A "run" = files sharing a base stem.** Test logs are named
  `test_{problem}_{trainsize}_{checkpointstem}_{dataset}_{YYYYMMDD}_{HHMMSS}_annealing{bool}.txt`,
  with sibling CSVs in `logs/csv/{stem}_summary.csv`, `{stem}_instances.csv`, `{stem}_iters.csv`.
  Group by the stem (filename minus extension / `_summary`/`_instances`/`_iters` suffix).
- **Authoritative metric source = `_instances.csv`** (written incrementally per-instance, so it
  exists for partial runs too). Columns include `idx,name,size,size_group,opt,baseline,base,
  model_anneal,model_no_anneal,mix_anneal,mix_no_anneal,...,*_I1000/I2000/...` iteration snapshots.
  `size_group` is already one of `<1K` / `[1K,10K)` / `>=10K` (test.py:2778-2780, 2641) — maps
  exactly to small/medium/large.
- **`_summary.csv`** (written near run end, test.py:3218-3229) holds run-level metadata
  (`problem,n_node,checkpoint,dataset,seed`), a per-method table
  (`Method,MeanCost,StdCost,Gap%,StdGap%,GapRef,MeanTime,TotalTime,Best`), and a
  `Mean gap by instance size` table. Use it as a cross-check + source of `MeanTime/TotalTime`.
- **Config source = the checkpoint path** (in `_summary.csv` `checkpoint` row, or the
  `checkpointstem` segment of the filename). It encodes the full config via `train.py`'s
  `build_model_name` (train.py ~3258-3335). Parse it token-by-token; **do not load `.pt`/torch**.
- **Multihead detection** (per train.py): token `_mh{K}` => `num_heads=K` (multihead);
  decoder-type token in `{lora, deep_lora, film, multi_decoder, lowrank, polynet}`; router token
  `_hr{static|ema|learned}` (also legacy `_hrlearned`, `hreta`); head-init `_hi{...}`/`anchored`;
  ant weights `_ha{w-w-w}`. Absence of `_mh` and `num_heads<=1` => single-head. `nockpt` stems =
  baseline-only runs (no model config).
- **Finished vs. unfinished:** `Finished` if `_summary.csv` exists AND `_instances.csv` row count
  == the `Loaded N instances` count parsed from the `.txt`. `Interrupted` if the `.txt` tail
  contains `KeyboardInterrupt` / `Traceback` / `OutOfMemoryError`. Otherwise `Partial/unknown`.
  (Note: the `Wrote summary JSON` marker exists in only 32 journal runs — do NOT rely on it.)
- **Timestamps:** parse `YYYYMMDD_HHMMSS` from the stem; fall back to file mtime.
- Shell here is `zsh`/`python3`, not nushell — ignore the env's `/bin/nu`.

## Deliverables (new files, under `tools/logs_explorer/`)

```
tools/logs_explorer/
├── app.py          # Streamlit UI (entry point)
├── parser.py       # log discovery + config/metric/status parsing (pure, testable)
└── README.md       # how to run
```

**Environment:** use the project's `uv` env (`pyproject.toml` already has `pandas`). Add the two
extra deps with `uv add streamlit plotly` (or as a `[dependency-group] logs-explorer` to keep them
optional), and run via `uv run streamlit run tools/logs_explorer/app.py`. Do **not** use bare
`pip`.

### `parser.py` — pure parsing layer (no Streamlit imports)

- `discover_runs(logs_dir) -> list[RunMeta]`: glob `logs/*.txt` and `logs/csv/*`, group by stem,
  attach available file paths (`txt`, `summary_csv`, `instances_csv`, `iters_csv`).
- `parse_config(checkpoint_or_stem) -> RunConfig`: token-based extractor returning
  `problem, n_node (train size), k_sparse, n_ants, H, mini_H, rho, mne, algo, lr,
  edge_feature_set, is_multihead, num_heads, decoder_type, router, head_init, head_ant_weights,
  extra_flags[], dataset, annealing, timestamp`. Tolerant of legacy/noisy run_tag concatenations
  (best-effort: unknown trailing tokens collected into `extra_flags`).
- `classify_status(run) -> {finished|interrupted|partial}`: per the rule above (parse
  `Loaded N instances` from `.txt`, count `_instances.csv` rows, scan `.txt` tail).
- `load_instances(run) -> pandas.DataFrame`: read `_instances.csv`; pick the primary model column
  (`model_anneal`, else `model_no_anneal`, else `base`); compute per-instance
  `gap% = (cost - opt)/opt*100` (fallback to `baseline` when `opt` empty), keep `size`,
  `size_group`.
- `aggregate_perf(df) -> dict`: mean gap% (and count) per bucket small/medium/large + overall;
  merge `MeanTime/TotalTime` from `_summary.csv` when present.
- `load_iters(run, instance=None) -> DataFrame`: **lazy**, only called from the convergence view;
  read `_iters.csv` with `usecols=[idx,name,method,anneal,iter,t,mean,best,...]`, filtered by
  instance to bound memory on the ~687MB files.
- Use `@functools.lru_cache` / a simple in-process cache keyed by (path, mtime) for run metadata.
- `DEFAULT_CONFIG`: load the baseline from `configs/{problem}/n1000_100.yaml` (problem-specific)
  and treat this shared hyperparameter set as defaults — `n_node, k_sparse, n_ants, H, mini_H,
  rho, min_new_edges(mne), algo, lr, edge_feature_set, num_heads, head_init(anchored),
  head_ant_weights(even), annealing`. `parse_config` flags each field as `is_default` when it
  matches the baseline so the UI can hide it. Reference shared values from `configs/tsp/n1000_100.yaml or n_1000_deep_100.yaml`:
  `n_node=1000, k_sparse=32, n_ants=100, H=10, mini_H=100, rho=0.5, mne=12, algo=ppo, lr=5e-6,
  edge_feature_set=compact3`.

### `app.py` — Streamlit UI

Reuse `parser.py`; cache the run index with `@st.cache_data` keyed on dir mtimes.

1. **Sidebar filters:** finished/unfinished/partial; problem (tsp/cvrp); train size; multihead
   (yes/no) + decoder type + router; date/time range (from parsed timestamp); free-text search on
   stem/dataset.
2. **Run table** (`st.dataframe`): one row per run — status badge, problem, size, head config
   (e.g. `MH×8 polynet/learned` vs `single`), dataset, timestamp, overall gap%, mean time, #
   instances done/total. Sortable; selectable rows feed the compare view.
3. **Run detail** (expander / second tab): extracted config — **by default only fields that
   deviate from the `n1000_100.yaml` baseline are shown** (so the shared hyperparameter set above
   stays hidden), with a "Show all config" toggle to reveal every field; the multihead block is
   highlighted. Then the per-size performance table, the raw `_summary.csv` method table, and a
   **lazy "Convergence" button** that loads `_iters.csv` for a chosen instance and plots
   mean/best cost vs. iteration (plotly), plus a per-bucket gap bar chart.
4. **Compare view:** multiselect ≥2 runs → a combined table (rows = runs, columns = small/medium/
   large/overall gap% and time, recomputed from instances) + grouped bar charts (gap by bucket,
   time by bucket) and, when iters available, overlaid convergence curves for a shared instance.

## Critical files

- **Read-only references (do not modify):** `test.py` (CSV schema at 3218-3229, size buckets at
  2778-2780/2158-2161, instance loop writing `size_group` at 2641), `train.py`
  (`build_model_name` ~3258-3335 for the naming/multihead tokens), `net.py` (decoder-type names).
- **New:** the four files under `tools/logs_explorer/`. No existing repo files change.

## Verification

1. `uv add streamlit plotly` (adds to the project env).
2. Unit-check the parser without a GUI:
   `uv run python -c "from tools.logs_explorer import parser; rs=parser.discover_runs('logs'); print(len(rs)); ..."`
   — assert it finds ~ (count of unique stems), correctly flags the known multihead run
   `..._mh8_polynet_lora_r8_...` as multihead K=8 decoder=polynet, and a `..._best_...` run as
   single-head; spot-check one finished run (summary present, rows==loaded) and one interrupted
   run (txt tail has `KeyboardInterrupt`/OOM).
3. Cross-check `aggregate_perf` for one run against its `_summary.csv` `Mean gap by instance size`
   table (numbers should match within rounding).
4. `uv run streamlit run tools/logs_explorer/app.py` → confirm: filters narrow the table; finished/
   unfinished filter works; selecting a partial run still shows partial per-bucket metrics;
   compare view renders table + bar charts for 2+ runs; convergence view lazily loads a big
   `_iters.csv` for one instance without freezing startup.
