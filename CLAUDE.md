# Project Context — MultiAgent-Negotiation

## Thesis Goals

MSc thesis at University of Porto. Three main pillars:

1. **Benchmark open-weight LLMs** on NegotiationArena (3 games across 4 model size tiers)
2. **Self-Refine ablation** — does iterative self-critique improve negotiation outcomes?
3. **Future work** — team negotiation and its performance/cost trade-offs

Reference paper: `context/NegotiationArena/2402.05863v1.pdf`
Self-Refine paper: `context/Self_Refine/self_refine_2303.17651v2-1-9.pdf`

---

## Repository Structure

```
MultiAgent-Negotiation/
├── ratbench/                  # Core framework
│   ├── agents/
│   │   ├── agents.py          # Abstract Agent base class
│   │   ├── agent_behaviours.py # SelfRefineAgent, SelfCheckingAgent
│   │   └── hf_agent.py        # HuggingFace agent (all open-weight models)
│   ├── alternating_game.py    # AlternatingGame base class + trace logging
│   ├── game_objects/          # Resource, Valuation, Trade, Goal primitives
│   └── utils.py               # factory_agent() strategy dispatcher
├── games/
│   ├── buy_sell_game/         # BuySell negotiation
│   ├── trading_game/          # Multi-resource trading
│   └── ultimatum/             # Multi-turn ultimatum game
├── runner/
│   └── run_experiment.py      # Main experiment runner
├── configs/
│   └── experiments.yaml       # All experiment definitions (single source of truth)
├── .logs/                     # Experiment results
│   ├── section_one/
│   ├── section_two/
│   └── self_refine/
├── slurm/
│   ├── launch.sh              # SLURM job submitter
│   ├── run.sh                 # Batch script (env + python call)
│   └── servers/               # Server profiles: mia.sh, deucalion.sh
├── kaggle/                    # Kaggle free GPU kernel submission
├── explorer/                  # Streamlit analysis webapp
│   ├── app.py
│   ├── pages/                 # 7 pages (status, section analyses, kaggle)
│   └── analysis/              # Data loaders: section_one.py, section_two.py, self_refine_process.py
├── _notebooks/
│   └── oss/                   # section_one_analysis.ipynb, section_two_analysis.ipynb, self_refine_analysis.ipynb
└── context/                   # Papers, server guides, thesis notes
    ├── NegotiationArena/
    ├── Self_Refine/
    ├── MSc_Thesis/
    └── Servers/               # MIA_Server.md, Deucalion_Server.md, Kaggle_Server.md
```

---

## Games

| Game | Description | Key Parameters |
|------|-------------|----------------|
| **BuySell** | Seller has private value; buyer has private value; negotiate price | `seller_val`, `buyer_val`, `money` |
| **Trading** | Two players swap resources to maximize utility | `p1_resources`, `p2_resources` (X, Y units) |
| **Ultimatum** | Proposer splits a pot; responder accept/rejects | `dollars` |

Default setup: BuySell (seller=40, buyer=60, money=100), Trading (P1: X=25 Y=5, P2: X=5 Y=25), Ultimatum ($100).

---

## Models

Four size tiers defined in `configs/experiments.yaml` → `_shared`:

| Tier | Models | Quantization |
|------|--------|-------------|
| `very_small` | gemma-3-4b-it, Ministral-3-8B, Qwen3.5-9B | 8bit |
| `small` | gemma-3-12b-it, Ministral-3-14B, Qwen3-14B | 8bit |
| `medium` | gemma-3-27b-it, Mistral-Small-3.2-24B, Qwen3.5-27B | 8bit |
| `big` | Llama-3.3-70B, Qwen2.5-72B, Mixtral-8x7B | 4bit |

Most active experiments use `small`. Results exist for very_small, small, and medium.

---

## Experiment Sections

### Section One — Baseline (`section_one`)
- Cross-play (`cross_play: true`), 30 runs per pair
- Ablation: `max_retries: 0` (default) vs `max_retries: 3`
- Config keys: `buysell_section_one`, `trading_section_one`, `ultimatum_section_one` (+ `_retry3` variants)

### Section Two — Personas (`section_two`)
- Self-play only (`cross_play: false`), 30 runs
- P1 always default; P2 gets a persona via `p2_behaviour`
- Three conditions: `default` / `desperate` / `cunning`
- Config keys: `buysell_section_two_personas`, `trading_section_two_personas`, `ultimatum_section_two_personas`

### Self-Refine (`self_refine`)
- Self-play, 30 runs, `max_retries: 2`
- Four strategy conditions per game:
  - `default` × `default`
  - `self_refine` × `self_refine`
  - `default` × `self_refine`
  - `self_refine` × `default`
- Config keys: `trading_self_refine_v1`, `buysell_self_refine_v1`, `ultimatum_self_refine_v1`

---

## Architecture

### Agent Strategy Dispatch
`ratbench/utils.py` → `factory_agent(name, strategy, ...)`:
- `strategy="default"` → `HuggingFaceAgent`
- `strategy="self_refine"` → `SelfRefineHuggingFaceAgent`
- `strategy="self_check"` → `SelfCheckingHuggingFaceAgent`

### Self-Refine Loop (`ratbench/agents/agent_behaviours.py`)
1. Generate initial draft via `super().think()`
2. Repeat `max_refine_iters=2` times:
   - Send 5-axis feedback prompt → get critique
   - Send refine prompt → get rewritten response
3. Restore conversation to pre-refine state; append only final answer
4. Store trace in `self._last_refine_trace`

### Trace Persistence (`ratbench/alternating_game.py`)
`write_game_state()` dumps refine traces to:
```
refine_trace_iter_{iteration}_turn_{turn}.json
```
Contains: `initial_draft`, `iterations[].{feedback, refined}`, `final`

---

## Logs Structure (`.logs/`)

Three top-level sections, each with slightly different nesting.

### Section One
```
.logs/section_one/{experiment}/
    no_retries/   ← max_retries=0
    retry3/       ← max_retries=3
        {size}/                              # very_small | small | medium
            {model_p1}_vs_{model_p2}/
                {setup}/                     # e.g. seller40_buyer60
                    {run_id}/                # epoch timestamp
                        game_state.json
                        interaction.log
```
Example: `.logs/section_one/buysell_section_one/no_retries/small/qwen3-14b_vs_gemma-3-12b-it/seller40_buyer60/1774993134002/`

### Section Two
```
.logs/section_two/{experiment}/
    {size}/
        {model}_vs_{model}_{behaviour_name}/   # e.g. gemma-3-12b-it_vs_gemma-3-12b-it_cunning
            {run_id}/
                game_state.json
                interaction.log
```
Example: `.logs/section_two/ultimatum_section_two_personas/small/qwen3-14b_vs_qwen3-14b_desperate/1774993134002/`

### Self-Refine
```
.logs/self_refine/{experiment}/
    {size}/
        {model}_vs_{model}/
            {setup}_{p1}P1_{p2}P2/           # e.g. seller40_buyer60_self_refineP1_defaultP2
                {run_id}/
                    game_state.json
                    interaction.log
                    refine_trace_iter_{i}_turn_{t}.json   # one per (iteration, turn) if SR active
```
Example: `.logs/self_refine/buysell_self_refine_v1/small/qwen3-14b_vs_qwen3-14b/seller40_buyer60_self_refineP1_self_refineP2/1774993134002/`

### Run-level files
| File | Contents |
|------|----------|
| `game_state.json` | Full serialized game state: all turns, agent conversations, parse info, outcome |
| `interaction.log` | Human-readable alternating dialogue |
| `refine_trace_iter_X_turn_Y.json` | Self-Refine trace for game iteration X, agent turn Y: `{initial_draft, iterations[{feedback, refined}], final}` |

---

## Infrastructure

### HPC (SLURM)
```bash
# Submit jobs for an experiment
SERVER=mia EXPERIMENT=buysell_section_one SIZE=small bash slurm/launch.sh
```
- Server profiles: `slurm/servers/mia.sh`, `slurm/servers/deucalion.sh`
- One sbatch job per (experiment, size) combo
- Supports `--resume --target_runs N` to top up incomplete runs

### Kaggle (free GPUs)
- ~30 GPU-hours/week, 12-hour wall-time per kernel
- Status tracked in `explorer/pages/7_kaggle_runs.py`

### Streamlit Explorer
```bash
streamlit run explorer/app.py
```
Pages: conversation viewer, experiment status, section one/two analysis, model comparison, Self-Refine analysis, Kaggle status.

---

## Key Files Quick Reference

| Purpose | Path |
|---------|------|
| All experiment definitions | `configs/experiments.yaml` |
| Main runner | `runner/run_experiment.py` |
| Self-Refine agent | `ratbench/agents/agent_behaviours.py` |
| HuggingFace agent | `ratbench/agents/hf_agent.py` |
| Game base class + trace logging | `ratbench/alternating_game.py` |
| Strategy dispatcher | `ratbench/utils.py` → `factory_agent()` |
| SLURM launcher | `slurm/launch.sh` |
| Streamlit app | `explorer/app.py` |
| Self-Refine trace loader | `explorer/analysis/self_refine_process.py` |
| Analysis notebooks | `_notebooks/oss/` |
| Evaluation notes | `context/NegotiationArena/notes_on_evaluation_in_negotiation_arena.md` |

---

## Terminology
- **strategy**: experimental condition (default, self_refine, desperate, cunning) — not "arm"
- **size** / **model group**: very_small, small, medium, big
- **section**: section_one (baseline), section_two (personas), self_refine
- **cross-play**: games between different models; **self-play**: model vs itself
