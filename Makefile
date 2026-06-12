# DyNACO Makefile
# Usage: make train CONFIG=configs/tsp_n1000_ppo.yaml
#        make eval EXP=experiments/tsp_n1000_ppo_20250420
#        make compare

.PHONY: train eval compare test lint clean-old help

# Default target
help:
	@echo "DyNACO Pipeline Commands:"
	@echo ""
	@echo "  make train CONFIG=<path>    Train a new experiment using YAML config"
	@echo "  make eval EXP=<path>       Evaluate a trained experiment"
	@echo "  make compare [PATTERN]     Compare experiments matching glob pattern (default: experiments/*)"
	@echo "  make test                  Run test suite"
	@echo "  make lint                  Run linter (ruff) and formatter check (black)"
	@echo "  make clean-old DAYS=<n>    Clean experiments older than N days (dry-run by default)"
	@echo "  make clean-old-force DAYS=<n>  Actually delete old experiments"
	@echo "  make index                 Rebuild experiments/index.csv"
	@echo "  make plot EXP=<path>       Generate plots for an experiment"
	@echo "  make sweep CONFIG=<path> PARAMS='<spec>'  Run hyperparameter sweep"
	@echo ""
	@echo "Examples:"
	@echo "  make train CONFIG=configs/tsp_n1000_ppo.yaml"
	@echo "  make eval EXP=experiments/tsp_n1000_ppo_20250420"
	@echo "  make compare --pattern 'experiments/tsp_*'"
	@echo "  make clean-old DAYS=30"

# Train a new experiment
train:
	@if [ -z "$(CONFIG)" ]; then echo "Error: CONFIG required. Usage: make train CONFIG=configs/tsp_n1000_ppo.yaml"; exit 1; fi
	uv run scripts/train.py --config $(CONFIG)

# Evaluate an experiment
eval:
	@if [ -z "$(EXP)" ]; then echo "Error: EXP required. Usage: make eval EXP=experiments/tsp_n1000_ppo_20250420"; exit 1; fi
	uv run scripts/evaluate.py --experiment $(EXP)

# Compare experiments (default: all)
compare:
	uv run scripts/compare.py --pattern "$(if $(pattern),$(pattern),experiments/*)"

# Run test suite
test:
	uv run pytest tests/ -v

# Lint and format check
lint:
	uv run ruff check .
	uv run black --check .

# Rebuild experiment index
index:
	uv run scripts/indexExperiments.py --rebuild

# Generate plots for experiment
plot:
	@if [ -z "$(EXP)" ]; then echo "Error: EXP required. Usage: make plot EXP=experiments/tsp_n1000_ppo_20250420"; exit 1; fi
	uv run scripts/plot.py --experiment $(EXP)

# Clean old experiments (dry-run by default)
clean-old:
	uv run scripts/clean.py --older-than $(if $(DAYS),$(DAYS),30) --dry-run

# Actually delete old experiments (use with caution)
clean-old-force:
	uv run scripts/clean.py --older-than $(if $(DAYS),$(DAYS),30) --force

# Hyperparameter sweep
sweep:
	@if [ -z "$(CONFIG)" ]; then echo "Error: CONFIG required. Usage: make sweep CONFIG=configs/tsp_n1000_ppo.yaml PARAMS='lr:0.0001,0.0005;rho:0.1,0.5'"; exit 1; fi
	uv run scripts/sweep.py --config $(CONFIG) --params "$(PARAMS)"

# Archive old checkpoints (manual review before deletion)
archive-checkpoints:
	@echo "Running checkpoint migration dry-run first..."
	uv run scripts/migrate_checkpoints.py --dry-run
	@echo ""
	@echo "Review the report above. If satisfied, run:"
	@echo "  uv run scripts/migrate_checkpoints.py --execute"
