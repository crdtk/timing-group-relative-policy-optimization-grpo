# Timing Group Relative Policy Optimization (GRPO). The target is a CSV file. Its name is made here, from the date, the GPU and the settings
# you name. The name travels to the notebook, the notebook writes the file under that name on Colab, make
# downloads it by the same name, and the download resolves the target. Every run adds a new file.
#   make                                  results/2026-10-30-14-20-T4.csv, the notebook's defaults
#   make SCRIPT=basic BATCH=4             settings you name reach the notebook and the file name
#   make CSV=my-run.csv                   or name the file yourself; the first run asks for a Google sign-in
# A failed run (out of memory, say) keeps its partial rows as <name>.csv.failed. Commit the CSV files.
# The run executes a temporary copy of the notebook, so the output notebook that colab exec writes never lands here.
#   make chart                            redraw docs/stage-breakdown.svg from the CSV files in results/

COLAB   ?= $(if $(shell command -v colab 2>/dev/null),colab,uvx --from google-colab-cli colab)
GPU     ?= T4
TIMEOUT ?= 3600
STAMP   := $(shell date +%Y-%m-%d-%H-%M)
SESSION ?= rlv-$(STAMP)

KNOBS := SCRIPT STEPS ROLLOUTS TOKENS BATCH SEED SKIP_ZERO_ADVANTAGE REPO_URL REPO_REF
SET   := $(foreach k,$(KNOBS),$(if $(filter command line,$(origin $(k))),$(k)))
ENVS  := $(foreach k,$(SET),--env $(k)=$($(k))) --env RUN=$(STAMP) --env NOTEBOOK=$(shell git hash-object rl_velocity.ipynb 2>/dev/null)

space := $(subst ,, )
CSV   ?= $(STAMP)-$(GPU)$(subst $(space),,$(foreach k,$(SET),-$(k)-$($(k)))).csv
.DEFAULT_GOAL := results/$(CSV)

results/%.csv: rl_velocity.ipynb
	@mkdir -p results
	$(COLAB) new -s $(SESSION) --gpu $(GPU)
	@tmp=$$(mktemp -d); trap 'rm -rf $$tmp' EXIT; cp rl_velocity.ipynb $$tmp/; status=0; \
	$(COLAB) exec -s $(SESSION) -f $$tmp/rl_velocity.ipynb --timeout $(TIMEOUT) $(ENVS) --env CSV=$(@F) || status=$$?; \
	$(COLAB) download -s $(SESSION) /content/$(@F) $@.part; \
	$(COLAB) stop -s $(SESSION); \
	if [ $$status = 0 ]; then mv $@.part $@; else mv $@.part $@.failed; fi; \
	exit $$status

.PHONY: chart
chart:
	python3 make_chart.py
