#!/usr/bin/env python3
"""
Resume interrupted NegotiationArena experiments.

For each (model_pair x setup) combo, counts how many games already
completed successfully in the log directory, and only runs the
remaining ones.

Usage:
    # Dry run — just see what's done and what's remaining:
    python resume_experiments.py --config configs/experiments.yaml --experiment trading_section_two_personas_medium --dry_run

    # Actually resume:
    python resume_experiments.py --config configs/experiments.yaml --experiment trading_section_two_personas_medium
"""

import sys
import os
import json
import itertools
import argparse
import traceback

sys.path.append(".")

import yaml
from dotenv import load_dotenv

from ratbench.utils import factory_agent
from ratbench.game_objects.resource import Resources
from ratbench.game_objects.goal import (
    BuyerGoal, SellerGoal, MaximisationGoal, UltimatumGoal,
)
from ratbench.game_objects.valuation import Valuation
from ratbench.constants import *

from games.buy_sell_game.game import BuySellGame
from games.trading_game.game import TradingGame
from games.ultimatum.ultimatum_multi_turn.game import MultiTurnUltimatumGame

load_dotenv(".env")


# ── helpers ───────────────────────────────────────────────────────────
def _safe_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _build_pairs(models: list, cross_play) -> list:
    if cross_play == "all":
        return list(itertools.product(models, models))
    elif cross_play:
        return [(a, b) for a, b in itertools.product(models, models) if a != b]
    else:
        return [(m, m) for m in models]


def count_completed_games(log_dir: str) -> int:
    """Count games with a valid END state in a log directory."""
    if not os.path.isdir(log_dir):
        return 0

    completed = 0
    for entry in os.listdir(log_dir):
        game_state_path = os.path.join(log_dir, entry, "game_state.json")
        if not os.path.isfile(game_state_path):
            continue
        try:
            with open(game_state_path) as f:
                data = json.load(f)
            last = data["game_state"][-1]
            if last.get("current_iteration") == "END":
                completed += 1
        except Exception:
            pass  # broken / incomplete game file - don't count it

    return completed


def _get_log_dir(game_type, log_base, pair_tag, setup):
    """Reconstruct the log directory path for a given combo, matching run_experiment.py logic."""
    behaviour_name = setup.get("behaviour_name", "")

    if game_type == "buysell":
        tag = f"seller{setup['seller_val']}_buyer{setup['buyer_val']}"
        if behaviour_name:
            tag = f"{tag}_{behaviour_name}"
        return os.path.join(log_base, pair_tag, tag)
    else:
        log_tag = pair_tag
        if behaviour_name:
            log_tag = f"{pair_tag}_{behaviour_name}"
        return os.path.join(log_base, log_tag)


# ── game factories (same as run_experiment.py, but with resume logic) ─
def run_buysell(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0):
    seller_val = setup["seller_val"]
    buyer_val = setup["buyer_val"]
    money = setup.get("money", 100)
    p1_behaviour = setup.get("p1_behaviour", "")
    p2_behaviour = setup.get("p2_behaviour", "")
    behaviour_name = setup.get("behaviour_name", "")

    tag = f"seller{seller_val}_buyer{buyer_val}"
    if behaviour_name:
        tag = f"{tag}_{behaviour_name}"
    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_dir = os.path.join(log_base, pair_tag, tag)

    already_done = count_completed_games(log_dir)
    remaining = max(0, num_runs - already_done)

    if remaining == 0:
        print(f"  [buysell] {pair_tag}/{tag}: {already_done}/{num_runs} done — SKIP")
        return 0, 0

    print(f"  [buysell] {pair_tag}/{tag}: {already_done}/{num_runs} done — running {remaining} more")

    success, errors = 0, 0
    for i in range(remaining):
        try:
            print(f"    Run {i+1}/{remaining} (total will be {already_done + i + 1}/{num_runs})")
            a1 = factory_agent(model_p1, agent_name=AGENT_ONE)
            a2 = factory_agent(model_p2, agent_name=AGENT_TWO)

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

    return success, errors


def run_trading(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0):
    p1_res = setup["p1_resources"]
    p2_res = setup["p2_resources"]
    p1_behaviour = setup.get("p1_behaviour", "")
    p2_behaviour = setup.get("p2_behaviour", "")
    behaviour_name = setup.get("behaviour_name", "")

    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_tag = pair_tag
    if behaviour_name:
        log_tag = f"{pair_tag}_{behaviour_name}"
    log_dir = os.path.join(log_base, log_tag)

    already_done = count_completed_games(log_dir)
    remaining = max(0, num_runs - already_done)

    if remaining == 0:
        print(f"  [trading] {log_tag}: {already_done}/{num_runs} done — SKIP")
        return 0, 0

    print(f"  [trading] {log_tag}: {already_done}/{num_runs} done — running {remaining} more")

    success, errors = 0, 0
    for i in range(remaining):
        try:
            print(f"    Run {i+1}/{remaining} (total will be {already_done + i + 1}/{num_runs})")
            r1 = Resources(p1_res)
            r2 = Resources(p2_res)
            a1 = factory_agent(model_p1, agent_name=AGENT_ONE)
            a2 = factory_agent(model_p2, agent_name=AGENT_TWO)

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

    return success, errors


def run_ultimatum(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0):
    dollars = setup.get("dollars", 100)
    p1_behaviour = setup.get("p1_behaviour", "")
    p2_behaviour = setup.get("p2_behaviour", "")
    behaviour_name = setup.get("behaviour_name", "")

    pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
    log_tag = pair_tag
    if behaviour_name:
        log_tag = f"{pair_tag}_{behaviour_name}"
    log_dir = os.path.join(log_base, log_tag)

    already_done = count_completed_games(log_dir)
    remaining = max(0, num_runs - already_done)

    if remaining == 0:
        print(f"  [ultimatum] {log_tag}: {already_done}/{num_runs} done — SKIP")
        return 0, 0

    print(f"  [ultimatum] {log_tag}: {already_done}/{num_runs} done — running {remaining} more")

    success, errors = 0, 0
    for i in range(remaining):
        try:
            print(f"    Run {i+1}/{remaining} (total will be {already_done + i + 1}/{num_runs})")
            a1 = factory_agent(model_p1, agent_name=AGENT_ONE)
            a2 = factory_agent(model_p2, agent_name=AGENT_TWO)

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

    return success, errors


GAME_RUNNERS = {
    "buysell": run_buysell,
    "trading": run_trading,
    "ultimatum": run_ultimatum,
}


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Resume interrupted NegotiationArena experiments")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to pairs involving this model")
    parser.add_argument("--log_base", type=str, default=None,
                        help="Override base log directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only show what would be run, don't execute")
    args = parser.parse_args()

    with open(args.config) as f:
        all_configs = yaml.safe_load(f)

    if args.experiment not in all_configs:
        print(f"Experiment '{args.experiment}' not found. Available: {list(all_configs.keys())}")
        sys.exit(1)

    cfg = all_configs[args.experiment]
    game_type = cfg["game"]
    num_runs = cfg["num_runs"]
    iterations = cfg["iterations"]
    setups = cfg["setups"]
    cross_play = cfg.get("cross_play", False)
    max_retries = cfg.get("max_retries", 0)
    models = cfg["models"]
    log_base = args.log_base or f".logs/{args.experiment}"

    runner = GAME_RUNNERS.get(game_type)
    if runner is None:
        print(f"Unknown game type: {game_type}")
        sys.exit(1)

    pairs = _build_pairs(models, cross_play)
    if args.model:
        pairs = [(p1, p2) for p1, p2 in pairs if args.model in (p1, p2)]

    # ── Scan phase: report what's done and what's remaining ──
    print(f"\n{'='*60}")
    print(f"RESUME SCAN: {args.experiment}")
    print(f"Game: {game_type} | Pairs: {len(pairs)} | Setups: {len(setups)} | Target runs/combo: {num_runs}")
    print(f"Log base: {log_base}")
    print(f"{'='*60}\n")

    total_target = 0
    total_done = 0
    total_remaining = 0

    for model_p1, model_p2 in pairs:
        pair_tag = f"{_safe_name(model_p1)}_vs_{_safe_name(model_p2)}"
        for setup in setups:
            log_dir = _get_log_dir(game_type, log_base, pair_tag, setup)
            behaviour_name = setup.get("behaviour_name", "")

            done = count_completed_games(log_dir)
            remaining = max(0, num_runs - done)
            total_target += num_runs
            total_done += done
            total_remaining += remaining

            status = "DONE" if remaining == 0 else f"-> {remaining} remaining"
            label = f"{pair_tag}" + (f"/{behaviour_name}" if behaviour_name else "")
            print(f"  {label:60s}  {done:3d}/{num_runs}  {status}")

    print(f"\nSummary: {total_done}/{total_target} games completed, {total_remaining} remaining")

    if args.dry_run:
        print("\n[DRY RUN] No games will be executed.")
        return

    if total_remaining == 0:
        print("\nAll games already completed! Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"RUNNING {total_remaining} remaining games...")
    print(f"{'='*60}\n")

    # ── Run phase ──
    total_success, total_errors = 0, 0
    for model_p1, model_p2 in pairs:
        for setup in setups:
            s, e = runner(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=max_retries)
            total_success += s
            total_errors += e

    print(f"\n{'='*50}")
    print(f"RESUME DONE: {total_success} new games succeeded, {total_errors} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()