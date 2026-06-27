# DyNACO Logs Explorer

Streamlit app for browsing `logs/` test artifacts without loading checkpoints or torch models.

## Install

Use the project `uv` environment:

```bash
uv add streamlit plotly
```

## Run

From the repository root:

```bash
uv run streamlit run tools/logs_explorer/app.py
```

The app scans `logs/*.txt` plus `logs/csv/*_{summary,instances,iters}.csv`, groups files by run
stem, parses config from the checkpoint-style stem, and keeps `_iters.csv` loading lazy for the
convergence views.

## Parser smoke check

```bash
uv run python -c "from tools.logs_explorer import parser; rs=parser.discover_runs('logs'); print(len(rs)); print(rs[0])"
```
