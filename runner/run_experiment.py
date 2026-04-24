#!/usr/bin/env python3
"""
Unified experiment runner for NegotiationArena with open-weight models.

Supports both self-play and cross-play via the `cross_play` config key:
  - false (default): self-play only  (A vs A, B vs B)
  - true:            cross-play only (A vs B, B vs A — all ordered pairs, no self-play)
  - "all":           both            (A vs A, A vs B, B vs A, B vs B)

Supports persona / social_behaviour experiments via config:
  - p1_behaviour / p2_behaviour per setup (default: "")
  - behaviour_name: human-readable label used in log directory names

Usage:
    # Run all models / pairs from config
    python runner/run_experiment.py --config configs/experiments.yaml --experiment buysell_section_one

    # Run only games where a specific model participates (as P1, P2, or both)
    python runner/run_experiment.py --config configs/experiments.yaml --experiment buysell_section_one --model "Qwen/Qwen2.5-7B-Instruct"
"""

import sys
import os
import re
import itertools

sys.path.append(".")

import argparse
import traceback
import yaml
from dotenv import load_dotenv

from ratbench.utils import factory_agent, normalize_model
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.goal import (
    BuyerGoal,
    SellerGoal,
    MaximisationGoal,
    UltimatumGoal,
)
from ratbench.game_objects.valuation import Valuation
from ratbench.constants import *

from games.buy_sell_game.game import BuySellGame
from games.trading_game.game import TradingGame
from games.ultimatum.ultimatum_multi_turn.game import MultiTurnUltimatumGame

load_dotenv(".env")


# ── helpers ───────────────────────────────────────────────────────────
def _safe_name(model) -> str:
    """Turn a model dict or ID string into a filesystem-safe label.

    Appends '_thinking' when ``enable_thinking`` is True so that
    thinking and non-thinking variants get distinct log directories.
    """
    if isinstance(model, dict):
        name = model["id"].split("/")[-1].lower()
        if model.get("enable_thinking"):
            name += "_thinking"
        return name
    return model.split("/")[-1].lower()


def _strategy_tag(setup: dict) -> str | None:
    """Return a per-player strategy tag when the setup specifies strategies.

    Format: ``{p1_strategy}P1_{p2_strategy}P2`` — encodes both strategies and
    their player assignment, so asymmetric (cross-strategy) runs get unique
    log directories. Returns None when no strategy fields are present, in
    which case the caller should fall back to ``behaviour_name``.
    """
    if "p1_strategy" not in setup and "p2_strategy" not in setup:
        return None
    p1 = setup.get("p1_strategy", "default")
    p2 = setup.get("p2_strategy", "default")
    return f"{p1}P1_{p2}P2"


def _build_pairs(models: list, cross_play) -> list:
    """
    Build the list of (model_p1, model_p2) pairs to run.

    Each model is a dict with at least an ``"id"`` key.

    cross_play=False  → self-play only:  [(A,A), (B,B)]
    cross_play=True   → cross-play only: [(A,B), (B,A)] for all A≠B
    cross_play="all"  → full product:    [(A,A), (A,B), (B,A), (B,B)]
    """
    if cross_play == "all":
        return list(itertools.product(models, models))
    elif cross_play:
        return [(a, b) for a, b in itertools.product(models, models) if a != b]
    else:
        return [(m, m) for m in models]


# ── game factories ────────────────────────────────────────────────────
def run_buysell(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0):
    seller_val = setup["seller_val"]
    buyer_val = setup["buyer_val"]
    money = setup.get("money", 100)

    #  Persona / social behaviour support
    p1_behaviour = setup.get("p1_behaviour", "")
    p2_behaviour = setup.get("p2_behaviour", "")
    behaviour_name = setup.get("behaviour_name", "")
    p1_strategy = setup.get("p1_strategy", "default")
    p2_strategy = setup.get("p2_strategy", "default")

    tag = f"seller{seller_val}_buyer{buyer_val}"
    suffix = _strategy_tag(setup) or behaviour_name
    if suffix:
        tag = f"{tag}_{suffix}"
    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_dir = os.path.join(log_base, pair_tag, tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[buysell] Run {i+1}/{num_runs} | {pair_tag} | {tag}")
            a1 = factory_agent(model_p1["id"], agent_name=AGENT_ONE, strategy=p1_strategy, quantization=model_p1["quantization"], model_type=model_p1["model_type"], enable_thinking=model_p1["enable_thinking"], rate_limit_delay=model_p1.get("rate_limit_delay"))
            a2 = factory_agent(model_p2["id"], agent_name=AGENT_TWO, strategy=p2_strategy, quantization=model_p2["quantization"], model_type=model_p2["model_type"], enable_thinking=model_p2["enable_thinking"], rate_limit_delay=model_p2.get("rate_limit_delay"))

            game = BuySellGame(
                players=[a1, a2],
                iterations=iterations,
                max_retries=max_retries,
                resources_support_set=Resources({"X": 0}),
                player_goals=[
                    SellerGoal(cost_of_production=Valuation({"X": seller_val})),
                    BuyerGoal(willingness_to_pay=Valuation({"X": buyer_val})),
                ],
                player_initial_resources=[
                    Resources({"X": 1}),
                    Resources({MONEY_TOKEN: money}),
                ],
                player_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                player_social_behaviour=[p1_behaviour, p2_behaviour],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {pair_tag}/{tag}: {success} ok, {errors} errors")
    return success, errors


def run_trading(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0):
    p1_res = setup["p1_resources"]
    p2_res = setup["p2_resources"]

    #  Persona / social behaviour support
    p1_behaviour = setup.get("p1_behaviour", "")
    p2_behaviour = setup.get("p2_behaviour", "")
    behaviour_name = setup.get("behaviour_name", "")
    p1_strategy = setup.get("p1_strategy", "default")
    p2_strategy = setup.get("p2_strategy", "default")

    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    suffix = _strategy_tag(setup) or behaviour_name
    log_tag = f"{pair_tag}_{suffix}" if suffix else pair_tag
    log_dir = os.path.join(log_base, log_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[trading] Run {i+1}/{num_runs} | {log_tag}")
            r1 = Resources(p1_res)
            r2 = Resources(p2_res)
            a1 = factory_agent(model_p1["id"], agent_name=AGENT_ONE, strategy=p1_strategy, quantization=model_p1["quantization"], model_type=model_p1["model_type"], enable_thinking=model_p1["enable_thinking"], rate_limit_delay=model_p1.get("rate_limit_delay"))
            a2 = factory_agent(model_p2["id"], agent_name=AGENT_TWO, strategy=p2_strategy, quantization=model_p2["quantization"], model_type=model_p2["model_type"], enable_thinking=model_p2["enable_thinking"], rate_limit_delay=model_p2.get("rate_limit_delay"))

            game = TradingGame(
                players=[a1, a2],
                iterations=iterations,
                max_retries=max_retries,
                resources_support_set=Resources({k: 0 for k in p1_res}),
                player_goals=[MaximisationGoal(r1), MaximisationGoal(r2)],
                player_initial_resources=[r1, r2],
                player_social_behaviour=[p1_behaviour, p2_behaviour],
                player_roles=[
                    f"You are {AGENT_ONE}, start by making a proposal.",
                    f"You are {AGENT_TWO}, start by responding to a trade.",
                ],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {log_tag}: {success} ok, {errors} errors")
    return success, errors


def run_ultimatum(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0):
    dollars = setup.get("dollars", 100)

    #  Persona / social behaviour support
    p1_behaviour = setup.get("p1_behaviour", "")
    p2_behaviour = setup.get("p2_behaviour", "")
    behaviour_name = setup.get("behaviour_name", "")
    p1_strategy = setup.get("p1_strategy", "default")
    p2_strategy = setup.get("p2_strategy", "default")

    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    suffix = _strategy_tag(setup) or behaviour_name
    log_tag = f"{pair_tag}_{suffix}" if suffix else pair_tag
    log_dir = os.path.join(log_base, log_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[ultimatum] Run {i+1}/{num_runs} | {log_tag}")
            a1 = factory_agent(model_p1["id"], agent_name=AGENT_ONE, strategy=p1_strategy, quantization=model_p1["quantization"], model_type=model_p1["model_type"], enable_thinking=model_p1["enable_thinking"], rate_limit_delay=model_p1.get("rate_limit_delay"))
            a2 = factory_agent(model_p2["id"], agent_name=AGENT_TWO, strategy=p2_strategy, quantization=model_p2["quantization"], model_type=model_p2["model_type"], enable_thinking=model_p2["enable_thinking"], rate_limit_delay=model_p2.get("rate_limit_delay"))

            game = MultiTurnUltimatumGame(
                players=[a1, a2],
                iterations=iterations,
                max_retries=max_retries,
                resources_support_set=Resources({"Dollars": 0}),
                player_goals=[UltimatumGoal(), UltimatumGoal()],
                player_initial_resources=[
                    Resources({"Dollars": dollars}),
                    Resources({"Dollars": 0}),
                ],
                player_social_behaviour=[p1_behaviour, p2_behaviour],
                player_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {log_tag}: {success} ok, {errors} errors")
    return success, errors


GAME_RUNNERS = {
    "buysell": run_buysell,
    "trading": run_trading,
    "ultimatum": run_ultimatum,
}


# ── log path derivation ───────────────────────────────────────────────
def _derive_log_base(experiment: str, cfg: dict, model_group: str | None) -> str:
    """
    Derive the nested log directory from config fields.

    Uses cfg["section"] to determine the section and model_group for the
    size tier.  CLI --model_group overrides the YAML default.
    """
    section = cfg.get("section", "")
    group = model_group or "default"

    if section == "section_one":
        base_name = re.sub(r"_retry\d+$", "", experiment)
        retries = cfg.get("max_retries", 0)
        condition = f"retry{retries}" if retries else "no_retries"
        return f".logs/{section}/{base_name}/{condition}/{group}"

    if section == "section_two":
        return f".logs/{section}/{experiment}/{group}"

    if section:
        return f".logs/{section}/{experiment}/{group}"

    return f".logs/{experiment}"


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NegotiationArena experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments.yaml",
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment block in the YAML file",
    )
    parser.add_argument(
        "--model_group",
        type=str,
        default=None,
        help="Override the model list by selecting a predefined size group from the _shared config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="(Optional) Filter to pairs involving this model. "
             "In self-play mode, runs only this model. "
             "In cross-play mode, runs all pairs where this model is P1 or P2.",
    )
    parser.add_argument(
        "--num_runs",
        type=str,
        default=None,
        help="Override number of runs from config (useful for quick tests)",
    )
    parser.add_argument(
        "--log_base",
        type=str,
        default=None,
        help="Override base log directory",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        all_configs = yaml.safe_load(f)

    if args.experiment not in all_configs:
        print(f"Experiment '{args.experiment}' not found. Available: {list(all_configs.keys())}")
        sys.exit(1)

    cfg = all_configs[args.experiment]
    game_type = cfg["game"]
    num_runs = int(args.num_runs) if args.num_runs else cfg["num_runs"]
    iterations = cfg["iterations"]
    setups = cfg["setups"]
    cross_play = cfg.get("cross_play", False)
    max_retries = cfg.get("max_retries", 0)
    
    # Extract the correct model list
    if args.model_group:
        # CLI override takes precedence
        model_group_name = args.model_group
        shared_key = f"models_{model_group_name}"
        if "_shared" in all_configs and shared_key in all_configs["_shared"]:
            raw_models = all_configs["_shared"][shared_key]
        else:
            print(f"Model group '{model_group_name}' not found under '_shared' in {args.config}")
            sys.exit(1)
    else:
        models_cfg = cfg.get("models", [])
        if isinstance(models_cfg, str):
            # String key → look up from _shared
            model_group_name = models_cfg
            shared_key = f"models_{model_group_name}"
            if "_shared" in all_configs and shared_key in all_configs["_shared"]:
                raw_models = all_configs["_shared"][shared_key]
            else:
                print(f"Model group '{model_group_name}' not found under '_shared' in {args.config}")
                sys.exit(1)
        else:
            # Inline list (legacy/backward compat)
            model_group_name = None
            raw_models = models_cfg

    models = [normalize_model(m) for m in raw_models]
    log_base = args.log_base or _derive_log_base(args.experiment, cfg, model_group_name)

    runner = GAME_RUNNERS.get(game_type)
    if runner is None:
        print(f"Unknown game type: {game_type}. Available: {list(GAME_RUNNERS.keys())}")
        sys.exit(1)

    # Build model pairs
    pairs = _build_pairs(models, cross_play)

    # Filter by --model if provided
    if args.model:
        pairs = [(p1, p2) for p1, p2 in pairs if args.model in (p1["id"], p2["id"])]
        if not pairs:
            print(f"No pairs found involving model '{args.model}'. "
                  f"Available models: {[m['id'] for m in models]}")
            sys.exit(1)

    # Count behaviour setups for summary
    behaviour_setups = [s for s in setups if s.get("behaviour_name")]
    default_setups = [s for s in setups if not s.get("behaviour_name")]

    # Summary
    print(f"{'='*60}")
    print(f"Experiment : {args.experiment}")
    print(f"Game       : {game_type}")
    print(f"Cross-play : {cross_play}")
    print(f"Max retries: {max_retries}")
    print(f"Model Grp  : {model_group_name or 'inline list'}")
    print(f"Pairs      : {len(pairs)}")
    for p1, p2 in pairs:
        label = "self-play" if p1["id"] == p2["id"] else "cross-play"
        print(f"  {_safe_name(p1)} vs {_safe_name(p2)}  ({label})")
    print(f"Setups     : {len(setups)} ({len(default_setups)} default, {len(behaviour_setups)} with personas)")
    for s in setups:
        bname = s.get("behaviour_name", "default")
        print(f"  - {bname}: {s}")
    print(f"Runs/combo : {num_runs}")
    print(f"Total games: {len(pairs) * len(setups) * num_runs}")
    print(f"{'='*60}\n")

    # Run all combos — evict unused models between pairs to save VRAM
    from ratbench.agents.hf_agent import evict_unused_models
    total_success, total_errors = 0, 0
    for model_p1, model_p2 in pairs:
        keep = {(model_p1["id"], model_p1["quantization"]),
                (model_p2["id"], model_p2["quantization"])}
        evict_unused_models(keep)
        for setup in setups:
            s, e = runner(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=max_retries)
            total_success += s
            total_errors += e

    print(f"\n{'='*50}")
    print(f"DONE: {total_success} succeeded, {total_errors} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()