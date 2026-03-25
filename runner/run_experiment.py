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
def _safe_name(model_id: str) -> str:
    """Turn 'Qwen/Qwen2.5-7B-Instruct' into 'qwen2.5-7b-instruct'."""
    return model_id.split("/")[-1].lower()


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
        return [(a, b) for a, b in itertools.product(models, models) if a["id"] != b["id"]]
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

    tag = f"seller{seller_val}_buyer{buyer_val}"
    if behaviour_name:
        tag = f"{tag}_{behaviour_name}"
    pair_tag = f"{_safe_name(model_p1['id'])}_vs_{_safe_name(model_p2['id'])}"
    log_dir = os.path.join(log_base, pair_tag, tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[buysell] Run {i+1}/{num_runs} | {pair_tag} | {tag}")
            a1 = factory_agent(model_p1["id"], agent_name=AGENT_ONE, quantization=model_p1["quantization"])
            a2 = factory_agent(model_p2["id"], agent_name=AGENT_TWO, quantization=model_p2["quantization"])

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

    pair_tag = f"{_safe_name(model_p1['id'])}_vs_{_safe_name(model_p2['id'])}"
    log_tag = pair_tag
    if behaviour_name:
        log_tag = f"{pair_tag}_{behaviour_name}"
    log_dir = os.path.join(log_base, log_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[trading] Run {i+1}/{num_runs} | {log_tag}")
            r1 = Resources(p1_res)
            r2 = Resources(p2_res)
            a1 = factory_agent(model_p1["id"], agent_name=AGENT_ONE, quantization=model_p1["quantization"])
            a2 = factory_agent(model_p2["id"], agent_name=AGENT_TWO, quantization=model_p2["quantization"])

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

    pair_tag = f"{_safe_name(model_p1['id'])}_vs_{_safe_name(model_p2['id'])}"
    log_tag = pair_tag
    if behaviour_name:
        log_tag = f"{pair_tag}_{behaviour_name}"
    log_dir = os.path.join(log_base, log_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[ultimatum] Run {i+1}/{num_runs} | {log_tag}")
            a1 = factory_agent(model_p1["id"], agent_name=AGENT_ONE, quantization=model_p1["quantization"])
            a2 = factory_agent(model_p2["id"], agent_name=AGENT_TWO, quantization=model_p2["quantization"])

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
    models = [normalize_model(m) for m in cfg["models"]]
    log_base = args.log_base or f".logs/{args.experiment}"

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
    print(f"Pairs      : {len(pairs)}")
    for p1, p2 in pairs:
        label = "self-play" if p1["id"] == p2["id"] else "cross-play"
        print(f"  {_safe_name(p1['id'])} vs {_safe_name(p2['id'])}  ({label})")
    print(f"Setups     : {len(setups)} ({len(default_setups)} default, {len(behaviour_setups)} with personas)")
    for s in setups:
        bname = s.get("behaviour_name", "default")
        print(f"  - {bname}: {s}")
    print(f"Runs/combo : {num_runs}")
    print(f"Total games: {len(pairs) * len(setups) * num_runs}")
    print(f"{'='*60}\n")

    # Run all combos
    total_success, total_errors = 0, 0
    for model_p1, model_p2 in pairs:
        for setup in setups:
            s, e = runner(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=max_retries)
            total_success += s
            total_errors += e

    print(f"\n{'='*50}")
    print(f"DONE: {total_success} succeeded, {total_errors} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()