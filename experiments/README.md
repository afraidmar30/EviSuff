# RAES-Eval Experiments

This directory contains the minimal RAES-Bench evaluation loop for SearchClaw.

## 1. Dry-Run Smoke Test

This validates dataset loading, result JSONL writing, trajectory logging,
scoring, and aggregation without calling an LLM or web tools.

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/dry_run.yaml \
  --dataset raes-bench/raes-bench-v12_process/dev_with_gold.jsonl \
  --limit 3 \
  --run-id smoke_dry_run \
  --no-resume

python3.10 experiments/score_raes.py \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --results outputs/runs/smoke_dry_run/results.jsonl \
  --output outputs/scores/smoke_dry_run.scores.jsonl

python3.10 experiments/aggregate_results.py \
  --scores outputs/scores/smoke_dry_run.scores.jsonl \
  --output-json outputs/tables/smoke_dry_run.summary.json \
  --output-md outputs/tables/smoke_dry_run.summary.md
```

## 2. SearchClaw Smoke Test

This requires the SearchClaw runtime dependencies and API configuration.
Use the low-budget smoke config first; it is intended to verify that the
agent, web tools, trajectory logging, scoring, and aggregation are wired
correctly before launching a larger run.

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/searchclaw_smoke.yaml \
  --dataset raes-bench/raes-bench-v12_process/dev_with_gold.jsonl \
  --limit 3 \
  --run-id smoke_searchclaw_smoke_3 \
  --no-resume

python3.10 experiments/score_raes.py \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --results outputs/runs/smoke_searchclaw_smoke_3/results.jsonl \
  --output outputs/scores/smoke_searchclaw_smoke_3.scores.jsonl

python3.10 experiments/aggregate_results.py \
  --scores outputs/scores/smoke_searchclaw_smoke_3.scores.jsonl \
  --output-json outputs/tables/smoke_searchclaw_smoke_3.summary.json \
  --output-md outputs/tables/smoke_searchclaw_smoke_3.summary.md
```

For a fuller SearchClaw run, switch back to the standard config:

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/searchclaw.yaml \
  --dataset raes-bench/raes-bench-v12_process/dev_with_gold.jsonl \
  --limit 20 \
  --run-id searchclaw_20 \
  --no-resume
```

The runner passes only the question and public metadata into the agent.
Gold fields are used only by `score_raes.py`.

## 3. SearchClaw + RAES Gate

This is the main SearchClaw loop plus the layered RAES-Gate controller.
The gate runs before finalization and can force the agent to continue
searching, revise unsupported claims, add uncertainty/conflict disclosure,
or finish cautiously when the gate iteration budget is exhausted. Gate
decisions are logged as `raes_gate` trajectory events.

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/searchclaw_raes_gate.yaml \
  --dataset raes-bench/raes-bench-v12_process/public_questions_stratified20_seed20260520.jsonl \
  --run-id qwen_plus_searchclaw_raes_gate_stratified20 \
  --no-resume

python3.10 experiments/score_raes.py \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --results outputs/runs/qwen_plus_searchclaw_raes_gate_stratified20/results.jsonl \
  --output outputs/scores/qwen_plus_searchclaw_raes_gate_stratified20.scores.jsonl

python3.10 experiments/aggregate_results.py \
  --scores outputs/scores/qwen_plus_searchclaw_raes_gate_stratified20.scores.jsonl \
  --output-json outputs/tables/qwen_plus_searchclaw_raes_gate_stratified20.summary.json \
  --output-md outputs/tables/qwen_plus_searchclaw_raes_gate_stratified20.summary.md
```

### Leave-One-Out RAES-Gate Ablation

The leave-one-out suite runs the SearchClaw baseline, Citation Prompt Only,
Full RAES-Gate, and one removal run for each RAES-Gate module. The
`w/o Stop Controller` condition still logs gate failures but does not block
finalization.

```bash
python3.10 experiments/run_raes_ablation_leave_one_out.py \
  --dataset raes-bench/raes-bench-v12_process/public_questions_stratified20_seed20260520.jsonl \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --prefix qwen_plus_leave_one_out_stratified20
```

## 4. Search + Summarize Baseline

This baseline performs one fixed retrieval pass, fetches the top pages,
and asks the LLM to synthesize from those retrieved sources. It does not
run the SearchClaw multi-turn agent loop, hooks, planning, or follow-up
searches.

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/search_summarize.yaml \
  --dataset raes-bench/raes-bench-v12_process/public_questions_stratified20_seed20260520.jsonl \
  --run-id qwen_plus_search_summarize_stratified20 \
  --no-resume

python3.10 experiments/score_raes.py \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --results outputs/runs/qwen_plus_search_summarize_stratified20/results.jsonl \
  --output outputs/scores/qwen_plus_search_summarize_stratified20.scores.jsonl

python3.10 experiments/aggregate_results.py \
  --scores outputs/scores/qwen_plus_search_summarize_stratified20.scores.jsonl \
  --output-json outputs/tables/qwen_plus_search_summarize_stratified20.summary.json \
  --output-md outputs/tables/qwen_plus_search_summarize_stratified20.summary.md
```

## 5. ReAct Agent Baseline

This baseline uses the same search/fetch tools as SearchClaw, but removes
the formal `research_plan` tool, disables stop hooks, and uses a simple
ReAct-style multi-turn loop. It is a stronger baseline than
Search + Summarize because it can issue follow-up searches.

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/react_agent.yaml \
  --dataset raes-bench/raes-bench-v12_process/public_questions_stratified20_seed20260520.jsonl \
  --run-id qwen_plus_react_agent_stratified20 \
  --no-resume

python3.10 experiments/score_raes.py \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --results outputs/runs/qwen_plus_react_agent_stratified20/results.jsonl \
  --output outputs/scores/qwen_plus_react_agent_stratified20.scores.jsonl

python3.10 experiments/aggregate_results.py \
  --scores outputs/scores/qwen_plus_react_agent_stratified20.scores.jsonl \
  --output-json outputs/tables/qwen_plus_react_agent_stratified20.summary.json \
  --output-md outputs/tables/qwen_plus_react_agent_stratified20.summary.md
```

## 6. ReAct + Citation Baseline

This baseline keeps the ReAct-style loop and adds only the citation
quality stop hook. It still removes `research_plan` and `ask_user`, and
does not use the full SearchClaw planning or completeness gate.

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/react_citation.yaml \
  --dataset raes-bench/raes-bench-v12_process/public_questions_stratified20_seed20260520.jsonl \
  --run-id qwen_plus_react_citation_stratified20 \
  --no-resume

python3.10 experiments/score_raes.py \
  --gold raes-bench/raes-bench-v12_process/gold.jsonl \
  --results outputs/runs/qwen_plus_react_citation_stratified20/results.jsonl \
  --output outputs/scores/qwen_plus_react_citation_stratified20.scores.jsonl

python3.10 experiments/aggregate_results.py \
  --scores outputs/scores/qwen_plus_react_citation_stratified20.scores.jsonl \
  --output-json outputs/tables/qwen_plus_react_citation_stratified20.summary.json \
  --output-md outputs/tables/qwen_plus_react_citation_stratified20.summary.md
```

## 7. Outputs

- `outputs/runs/<run_id>/results.jsonl`
- `outputs/runs/<run_id>/trajectories/*.trajectory.jsonl`
- `outputs/training/<run_id>/sft_messages.jsonl`
- `outputs/training/<run_id>/filter_report.json`
- `outputs/training/<run_id>/dataset_card.md`
- `outputs/scores/*.scores.jsonl`
- `outputs/tables/*.json`
- `outputs/tables/*.md`

`outputs/` is ignored by git.

The runner prints terminal progress by default, including completed/total
samples, status, sample time, citations, remaining count, and ETA. Add
`--no-progress` if you need quiet output.

## 8. Export SFT Training Trajectories

The `searchclaw_raes_gate.yaml` config records full text deltas and up to
20k chars per tool result for trajectory export. To run the full 400-item
RAES v12 public split with the configured `openai/qwen-plus` teacher:

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/searchclaw_raes_gate.yaml \
  --dataset raes-bench/raes-bench-v12_process/public_questions.jsonl \
  --run-id qwen_plus_searchclaw_raes_gate_full400_sft
```

Export filtered SFT JSONL artifacts:

```bash
python3.10 experiments/export_raes_sft.py \
  --run-dir outputs/runs/qwen_plus_searchclaw_raes_gate_full400_sft
```

The exporter writes portable chat-style samples to
`outputs/training/<run_id>/sft_messages.jsonl`. Each row keeps `messages` as
`system/user/final assistant` and stores the agentic search process separately
in `tool_trace`, so tool observations are preserved without being appended to
the final assistant answer.

### RAES-Seed-2K Multi-Rollout Collection

For a larger seed set, run four teacher rollouts over the 400 public RAES
tasks with different temperatures, export each rollout, and merge the accepted
samples:

```bash
python3.10 experiments/run_raes_seed2k.py \
  --dataset raes-bench/raes-bench-v12_process/public_questions.jsonl \
  --config experiments/configs/searchclaw_raes_gate.yaml \
  --prefix qwen_plus_searchclaw_raes_gate_seed2k \
  --rollouts 4 \
  --temperatures 0,0.2,0.4,0.6
```

This creates per-rollout runs under `outputs/runs/<prefix>_rolloutXX_*`,
per-rollout exports under `outputs/training/<prefix>_rolloutXX_*`, and a
merged dataset under `outputs/training/<prefix>/`:

- `sft_messages.jsonl`: accepted full-trajectory SFT samples
- `step_messages.jsonl`: derived process-level samples for tool-call
  prediction, RAES-Gate decisions, and final-answer synthesis
- `merge_report.json`: counts by rollout

For a cheap validation pass:

```bash
python3.10 experiments/run_raes_seed2k.py \
  --limit 3 \
  --prefix seed2k_smoke \
  --rollouts 2 \
  --temperatures 0,0.4 \
  --no-resume
```

## 9. Resuming After API Quota Stops

The RAES runner treats API quota/rate/billing/auth failures as fatal,
resumable stops. If Serper, NewsAPI, or the LLM API reports that quota
or credits are exhausted, the process exits immediately instead of falling
back to a lower-quality backend. Jina supports key rotation via
`JINA_API_KEYS`; it exits only after all configured Jina keys are exhausted
or rejected.

Completed samples are already flushed to:

```bash
outputs/runs/<run_id>/results.jsonl
```

To resume, re-run with the same `--run-id` and omit `--no-resume`:

```bash
python3.10 experiments/run_raes.py \
  --config experiments/configs/searchclaw.yaml \
  --dataset raes-bench/raes-bench-v12_process/public_questions.jsonl \
  --run-id searchclaw_full_400
```

Use `--no-resume` only when you intentionally want to clear and restart
that run id from scratch.
