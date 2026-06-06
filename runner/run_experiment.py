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

from ratbench.utils import factory_agent, normalize_model, build_party
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
    """Turn a model dict / ID string / team party into a filesystem-safe label.

    Appends '_thinking' when ``enable_thinking`` is True so that
    thinking and non-thinking variants get distinct log directories.
    A team party (``{"team": {...}}``) becomes e.g. ``team_gemma-3-12b-it_x3``
    (homogeneous) or ``team_gemma-3-12b-it+ministral-3-14b+qwen3-14b``
    (heterogeneous).
    """
    if isinstance(model, dict) and "team" in model:
        members = model["team"]["members"]
        shorts = [m["id"].split("/")[-1].lower() for m in members]
        if len(set(shorts)) == 1:
            return f"team_{shorts[0]}_x{len(shorts)}"
        return "team_" + "+".join(shorts)
    if isinstance(model, dict):
        name = model["id"].split("/")[-1].lower()
        if model.get("enable_thinking"):
            name += "_thinking"
        return name
    return model.split("/")[-1].lower()


def _party_model_keys(spec) -> set:
    """Return the set of ``(model_id, quantization)`` keys a party loads.

    A team contributes one key per member; a single model contributes one.
    Used to build the VRAM ``keep`` set so heterogeneous members are not
    evicted mid-sweep.
    """
    if isinstance(spec, dict) and "team" in spec:
        return {(m["id"], m.get("quantization")) for m in spec["team"]["members"]}
    return {(spec["id"], spec["quantization"])}


def _resolve_model_list(opponents, all_configs) -> list:
    """Resolve a team experiment's ``opponents`` (tier key or inline list)."""
    if isinstance(opponents, str):
        shared_key = f"models_{opponents}"
        raw = all_configs.get("_shared", {}).get(shared_key)
        if raw is None:
            print(f"Opponents group '{opponents}' not found under '_shared'.")
            sys.exit(1)
    elif isinstance(opponents, list):
        raw = opponents
    else:
        print("A team experiment requires an 'opponents' tier key or list.")
        sys.exit(1)
    return [normalize_model(m) for m in raw]


def _count_existing_games(log_dir: str) -> int:
    """Count completed games (subdirs containing game_state.json) in log_dir."""
    if not os.path.isdir(log_dir):
        return 0
    return sum(
        1 for e in os.scandir(log_dir)
        if e.is_dir() and os.path.exists(os.path.join(e.path, "game_state.json"))
    )


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
def run_buysell(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0, target_runs=None):
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

    if target_runs is not None:
        existing = _count_existing_games(log_dir)
        remaining = max(0, target_runs - existing)
        if remaining == 0:
            print(f"  → {pair_tag}/{tag}: {existing}/{target_runs} done, skipping")
            return 0, 0
        print(f"  → {pair_tag}/{tag}: {existing}/{target_runs} done, running {remaining} more")
        num_runs = remaining

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[buysell] Run {i+1}/{num_runs} | {pair_tag} | {tag}")
            a1 = build_party(model_p1, AGENT_ONE, p1_strategy)
            a2 = build_party(model_p2, AGENT_TWO, p2_strategy)

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


def run_trading(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0, target_runs=None):
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

    if target_runs is not None:
        existing = _count_existing_games(log_dir)
        remaining = max(0, target_runs - existing)
        if remaining == 0:
            print(f"  → {log_tag}: {existing}/{target_runs} done, skipping")
            return 0, 0
        print(f"  → {log_tag}: {existing}/{target_runs} done, running {remaining} more")
        num_runs = remaining

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[trading] Run {i+1}/{num_runs} | {log_tag}")
            r1 = Resources(p1_res)
            r2 = Resources(p2_res)
            a1 = build_party(model_p1, AGENT_ONE, p1_strategy)
            a2 = build_party(model_p2, AGENT_TWO, p2_strategy)

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


def run_ultimatum(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=0, target_runs=None):
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

    if target_runs is not None:
        existing = _count_existing_games(log_dir)
        remaining = max(0, target_runs - existing)
        if remaining == 0:
            print(f"  → {log_tag}: {existing}/{target_runs} done, skipping")
            return 0, 0
        print(f"  → {log_tag}: {existing}/{target_runs} done, running {remaining} more")
        num_runs = remaining

    success, errors = 0, 0
    for i in range(num_runs):
        try:
            print(f"[ultimatum] Run {i+1}/{num_runs} | {log_tag}")
            a1 = build_party(model_p1, AGENT_ONE, p1_strategy)
            a2 = build_party(model_p2, AGENT_TWO, p2_strategy)

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combinations that already have enough games; top up to --target_runs",
    )
    parser.add_argument(
        "--target_runs",
        type=int,
        default=None,
        help="Target games per combination when --resume is active (default: num_runs from config)",
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
    target_runs = (args.target_runs if args.target_runs is not None else num_runs) if args.resume else None
    
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

    runner = GAME_RUNNERS.get(game_type)
    if runner is None:
        print(f"Unknown game type: {game_type}. Available: {list(GAME_RUNNERS.keys())}")
        sys.exit(1)

    # Build pairs — a team experiment is a fixed team party versus each opponent;
    # `team_slot` (p1 default / p2) chooses which slot the team occupies so we can
    # study it as opener (P1) or responder (P2). Otherwise the usual self/cross-play
    # product over the model tier.
    team_cfg = cfg.get("team")
    if team_cfg:
        opponents = _resolve_model_list(cfg.get("opponents"), all_configs)
        party = {"team": team_cfg}
        team_slot = str(cfg.get("team_slot", "p1")).lower()
        if team_slot == "p2":
            pairs = [(opp, party) for opp in opponents]   # team responds in P2
        else:
            pairs = [(party, opp) for opp in opponents]    # team opens in P1
        if isinstance(cfg.get("opponents"), str):
            model_group_name = cfg["opponents"]
    else:
        pairs = _build_pairs(models, cross_play)

    log_base = args.log_base or _derive_log_base(args.experiment, cfg, model_group_name)

    # Filter by --model if provided (matches either slot; a team party has no id)
    if args.model:
        pairs = [(p1, p2) for p1, p2 in pairs
                 if args.model in (p1.get("id"), p2.get("id"))]
        if not pairs:
            print(f"No pairs found involving model '{args.model}'.")
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
        if "team" in p1:
            label = "team-vs-opponent"
        elif p1["id"] == p2["id"]:
            label = "self-play"
        else:
            label = "cross-play"
        print(f"  {_safe_name(p1)} vs {_safe_name(p2)}  ({label})")
    print(f"Setups     : {len(setups)} ({len(default_setups)} default, {len(behaviour_setups)} with personas)")
    for s in setups:
        bname = s.get("behaviour_name", "default")
        print(f"  - {bname}: {s}")
    print(f"Runs/combo : {num_runs}{f' (resume → target {target_runs})' if target_runs is not None else ''}")
    print(f"Total games: {len(pairs) * len(setups) * num_runs} max")
    print(f"{'='*60}\n")

    # Run all combos — evict unused models between pairs to save VRAM
    from ratbench.agents.hf_agent import evict_unused_models
    evict_hf_cache = cfg.get("evict_hf_cache", False)
    total_success, total_errors = 0, 0
    for model_p1, model_p2 in pairs:
        keep = _party_model_keys(model_p1) | _party_model_keys(model_p2)
        evict_unused_models(keep, evict_disk=evict_hf_cache)
        for setup in setups:
            s, e = runner(model_p1, model_p2, setup, num_runs, iterations, log_base, max_retries=max_retries, target_runs=target_runs)
            total_success += s
            total_errors += e

    print(f"\n{'='*50}")
    print(f"DONE: {total_success} succeeded, {total_errors} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()