#!/usr/bin/env python3
"""
Unified experiment runner for NegotiationArena with open-weight models.

Supports both self-play and cross-play via the `cross_play` config key:
  - false (default): self-play only  (A vs A, B vs B)
  - true:            cross-play only (A vs B, B vs A — all ordered pairs, no self-play)
  - "all":           both            (A vs A, A vs B, B vs A, B vs B)

Usage:
    # Run all models / pairs from config
    python runner/run_experiment.py --config configs/experiments.yaml --experiment buysell_section_one

    # Run only games where a specific model participates (as P1, P2, or both)
    python runner/run_experiment.py --config configs/experiments.yaml --experiment buysell_section_one_cross --model "Qwen/Qwen2.5-7B-Instruct"
"""

import sys
import os
import itertools

sys.path.append(".")

import argparse
import traceback
import yaml
from dotenv import load_dotenv

from ratbench.agents.hf_agent import HuggingFaceAgent
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
def run_buysell(model_p1, model_p2, setup, num_runs, iterations, log_base):
    seller_val = setup["seller_val"]
    buyer_val = setup["buyer_val"]
    money = setup.get("money", 100)
    tag = f"seller{seller_val}_buyer{buyer_val}"
    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_dir = os.path.join(log_base, pair_tag, tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[buysell] Run {i+1}/{num_runs} | {pair_tag} | {tag}")
            a1 = HuggingFaceAgent(agent_name=AGENT_ONE, model_id=model_p1)
            a2 = HuggingFaceAgent(agent_name=AGENT_TWO, model_id=model_p2)

            game = BuySellGame(
                players=[a1, a2],
                iterations=iterations,
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
                player_social_behaviour=["", ""],
                log_dir=log_dir,
            )
            game.run()
            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()

    print(f"  ✓ {pair_tag}/{tag}: {success} ok, {errors} errors")
    return success, errors


def run_trading(model_p1, model_p2, setup, num_runs, iterations, log_base):
    p1_res = setup["p1_resources"]
    p2_res = setup["p2_resources"]
    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_dir = os.path.join(log_base, pair_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[trading] Run {i+1}/{num_runs} | {pair_tag}")
            r1 = Resources(p1_res)
            r2 = Resources(p2_res)
            a1 = HuggingFaceAgent(agent_name=AGENT_ONE, model_id=model_p1)
            a2 = HuggingFaceAgent(agent_name=AGENT_TWO, model_id=model_p2)

            game = TradingGame(
                players=[a1, a2],
                iterations=iterations,
                resources_support_set=Resources({k: 0 for k in p1_res}),
                player_goals=[MaximisationGoal(r1), MaximisationGoal(r2)],
                player_initial_resources=[r1, r2],
                player_social_behaviour=["", ""],
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

    print(f"  ✓ {pair_tag}: {success} ok, {errors} errors")
    return success, errors


def run_ultimatum(model_p1, model_p2, setup, num_runs, iterations, log_base):
    dollars = setup.get("dollars", 100)
    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_dir = os.path.join(log_base, pair_tag)

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[ultimatum] Run {i+1}/{num_runs} | {pair_tag}")
            a1 = HuggingFaceAgent(agent_name=AGENT_ONE, model_id=model_p1)
            a2 = HuggingFaceAgent(agent_name=AGENT_TWO, model_id=model_p2)

            game = MultiTurnUltimatumGame(
                players=[a1, a2],
                iterations=iterations,
                resources_support_set=Resources({"Dollars": 0}),
                player_goals=[UltimatumGoal(), UltimatumGoal()],
                player_initial_resources=[
                    Resources({"Dollars": dollars}),
                    Resources({"Dollars": 0}),
                ],
                player_social_behaviour=["", ""],
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

    print(f"  ✓ {pair_tag}: {success} ok, {errors} errors")
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
        type=int,
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
    num_runs = args.num_runs or cfg["num_runs"]
    iterations = cfg["iterations"]
    setups = cfg["setups"]
    cross_play = cfg.get("cross_play", False)
    models = cfg["models"]
    log_base = args.log_base or f".logs/{args.experiment}"

    runner = GAME_RUNNERS.get(game_type)
    if runner is None:
        print(f"Unknown game type: {game_type}. Available: {list(GAME_RUNNERS.keys())}")
        sys.exit(1)

    # Build model pairs
    pairs = _build_pairs(models, cross_play)

    # Filter by --model if provided
    if args.model:
        pairs = [(p1, p2) for p1, p2 in pairs if args.model in (p1, p2)]
        if not pairs:
            print(f"No pairs found involving model '{args.model}'. "
                  f"Available models: {models}")
            sys.exit(1)

    # Summary
    print(f"{'='*60}")
    print(f"Experiment : {args.experiment}")
    print(f"Game       : {game_type}")
    print(f"Cross-play : {cross_play}")
    print(f"Pairs      : {len(pairs)}")
    for p1, p2 in pairs:
        label = "self-play" if p1 == p2 else "cross-play"
        print(f"  {_safe_name(p1)} vs {_safe_name(p2)}  ({label})")
    print(f"Setups     : {len(setups)}")
    print(f"Runs/combo : {num_runs}")
    print(f"Total games: {len(pairs) * len(setups) * num_runs}")
    print(f"{'='*60}\n")

    # Run all combos
    total_success, total_errors = 0, 0
    for model_p1, model_p2 in pairs:
        for setup in setups:
            s, e = runner(model_p1, model_p2, setup, num_runs, iterations, log_base)
            total_success += s
            total_errors += e

    print(f"\n{'='*50}")
    print(f"DONE: {total_success} succeeded, {total_errors} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()