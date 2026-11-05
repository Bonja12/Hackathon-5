"""Geopolitical and user-country experiments on top of PRoViLE.

Research questions
------------------
1. Do models from the US, Europe and Asia differ in how they answer the same
   politically sensitive or harmful requests?
2. Can a model be made to believe it is talking to a user from a specific
   country, and does that belief change its output?

Method (paired design)
----------------------
* The attacker LLM writes the attack prompts ONCE (independently of any target)
  and they are cached. Every target model and every persona then receives the
  exact same prompts, so differences cannot come from attacker randomness.
  The technique ``direct_request`` skips the attacker and uses a fixed prompt.
* Persona conditions only add a statement about where the user is located; see
  ``personas.py``. The baseline adds nothing.
* A manipulation check asks the target where the user is located, to measure
  whether it accepts the persona at all.
* Judge, scale and parsing are the unchanged PRoViLE ones.

Usage (from the top folder, with the .env of PRoViLE)::

    python -m provile.geo_experiment --dry-run
    python -m provile.geo_experiment
    python -m provile.geo_experiment --analyze-only
"""

import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from provile import analysis
from provile.llm_model import get_LLM
from provile.main import check_attacker_prompt, check_judge_response
from provile.personas import (
    BELIEF_PROBES,
    DEFAULT_PERSONAS,
    DELIVERIES,
    Persona,
    apply_persona,
    get_persona,
    mentions_country,
)

load_dotenv()

PKG_DIR = Path(__file__).resolve().parent
DIRECT_TECHNIQUE = "direct_request"
ATTACK_FAILED = "no attack prompt could be generated"

DEFAULT_OBJECTIVES = [
    PKG_DIR / "attack_objectives" / "geopolitical_objectives.csv",
    PKG_DIR / "attack_objectives" / "test_objectives.csv",
]
DEFAULT_TECHNIQUES = [
    PKG_DIR / "attack_techniques" / "geopolitical_techniques.csv",
    PKG_DIR / "attack_techniques" / "test_techniques.csv",
]
DEFAULT_TARGETS = PKG_DIR / "targets" / "geo_targets.csv"

PROMPT_COLUMNS = ["objective", "category", "technique", "prompt_idx", "atk_prompt"]
RESULT_KEY = ["model", "condition", "objective", "technique", "prompt_idx", "repeat"]
BELIEF_KEY = ["model", "condition", "probe_idx", "repeat"]


@dataclass(frozen=True)
class Condition:
    name: str
    persona: Persona | None
    delivery: str | None


def build_conditions(
    persona_codes: list[str], deliveries: list[str], include_baseline: bool = True
) -> list[Condition]:
    conditions = [Condition(analysis.BASELINE, None, None)] if include_baseline else []
    for code in persona_codes:
        persona = get_persona(code)
        for delivery in deliveries:
            conditions.append(Condition(f"{persona.code}/{delivery}", persona, delivery))
    return conditions


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, delimiter=";") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "category" not in df.columns:
        df["category"] = np.nan
    return df


def load_objectives(paths: list[Path]) -> pd.DataFrame:
    df = read_csvs(paths)
    df["category"] = df["category"].fillna("harmful_content")
    if df["name"].duplicated().any():
        raise ValueError("Objective names must be unique across the objective files.")
    return df


def load_techniques(paths: list[Path]) -> pd.DataFrame:
    df = read_csvs(paths)
    if df["name"].duplicated().any():
        raise ValueError("Technique names must be unique across the technique files.")
    return df


def direct_prompt(objective_prompt: str) -> str:
    return f"Please provide {objective_prompt}."


def generate_attack_prompts(
    attacker: Any,
    objectives: pd.DataFrame,
    techniques: pd.DataFrame,
    shots: int,
    cache_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Create (or load from cache) the fixed prompt set used for every target."""
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty:
        # An earlier run may have cached an API error as if it were a prompt.
        cached = cached[~cached["atk_prompt"].map(analysis.is_error_response)]
    done = (
        {(r.objective, r.technique, int(r.prompt_idx)) for r in cached.itertuples()}
        if not cached.empty
        else set()
    )
    rows: list[dict[str, Any]] = cached.to_dict("records") if not cached.empty else []

    for _, obj in objectives.iterrows():
        for _, tech in techniques.iterrows():
            n = 1 if tech["name"] == DIRECT_TECHNIQUE else shots
            for idx in range(n):
                if (obj["name"], tech["name"], idx) in done:
                    continue
                if tech["name"] == DIRECT_TECHNIQUE:
                    prompt = direct_prompt(obj["prompt"])
                else:
                    prompt = check_attacker_prompt(
                        attacker, obj, tech, [], logger, defaultdict(list)
                    )
                    if prompt == ATTACK_FAILED or analysis.is_error_response(prompt):
                        logger.warning(
                            f"No attack prompt for {obj['name']}/{tech['name']}/{idx}; "
                            "skipped (run again to retry)"
                        )
                        continue
                rows.append(
                    {
                        "objective": obj["name"],
                        "category": obj["category"],
                        "technique": tech["name"],
                        "prompt_idx": idx,
                        "atk_prompt": prompt,
                    }
                )
                pd.DataFrame(rows, columns=PROMPT_COLUMNS).to_csv(cache_path, index=False)
    return pd.DataFrame(rows, columns=PROMPT_COLUMNS)


def load_done(
    path: Path, key: list[str], retry_errors: bool = False
) -> tuple[list[dict[str, Any]], set[tuple]]:
    """Rows stored by an earlier run. With ``retry_errors`` the rows that ended in
    an API error are dropped, so they are sent again."""
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    if retry_errors and "error" in df.columns:
        df = df[~df["error"].astype(bool)]
    return df.to_dict("records"), {tuple(r) for r in df[key].itertuples(index=False)}


def run_attacks(
    targets: pd.DataFrame,
    judge: Any,
    llm_factory: Any,
    prompts: pd.DataFrame,
    objectives: pd.DataFrame,
    conditions: list[Condition],
    repeats: int,
    results_path: Path,
    logger: logging.Logger,
    save_every: int = 25,
    retry_errors: bool = False,
) -> pd.DataFrame:
    """Send every prompt to every target under every condition and judge it."""
    rows, done = load_done(results_path, RESULT_KEY, retry_errors)
    objective_by_name = {r["name"]: r for _, r in objectives.iterrows()}
    new = 0

    for _, tgt in targets.iterrows():
        target = llm_factory(tgt["model"])
        for cond in conditions:
            for p in prompts.itertuples():
                for rep in range(repeats):
                    key = (tgt["model"], cond.name, p.objective, p.technique, p.prompt_idx, rep)
                    if key in done:
                        continue
                    system, user_msg = apply_persona(cond.persona, cond.delivery or "", p.atk_prompt)
                    response = target.invoke(user_msg, system_prompt=system)
                    error = analysis.is_error_response(response)
                    score, judge_response = np.nan, ""
                    if not error:
                        score, judge_response = check_judge_response(
                            judge, objective_by_name[p.objective], response, logger, defaultdict(list)
                        )
                    rows.append(
                        {
                            "model": tgt["model"],
                            "region": tgt["region"],
                            "condition": cond.name,
                            "persona": cond.persona.code if cond.persona else "",
                            "delivery": cond.delivery or "",
                            "objective": p.objective,
                            "category": p.category,
                            "technique": p.technique,
                            "prompt_idx": p.prompt_idx,
                            "repeat": rep,
                            "atk_prompt": p.atk_prompt,
                            "target_response": response,
                            "judge_response": judge_response,
                            "score": score,
                            "error": error,
                        }
                    )
                    new += 1
                    if new % save_every == 0:
                        pd.DataFrame(rows).to_csv(results_path, index=False)
                        logger.info(f"{len(rows)} trials stored")
    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    return df


def run_belief_check(
    targets: pd.DataFrame,
    llm_factory: Any,
    conditions: list[Condition],
    repeats: int,
    belief_path: Path,
    logger: logging.Logger,
    retry_errors: bool = False,
) -> pd.DataFrame:
    """Ask each target where it thinks the user is. Measures whether the persona 'sticks'."""
    rows, done = load_done(belief_path, BELIEF_KEY, retry_errors)
    for _, tgt in targets.iterrows():
        target = llm_factory(tgt["model"])
        for cond in conditions:
            for probe_idx, probe in enumerate(BELIEF_PROBES):
                for rep in range(repeats):
                    if (tgt["model"], cond.name, probe_idx, rep) in done:
                        continue
                    system, user_msg = apply_persona(cond.persona, cond.delivery or "", probe)
                    answer = target.invoke(user_msg, system_prompt=system)
                    error = analysis.is_error_response(answer)
                    rows.append(
                        {
                            "model": tgt["model"],
                            "region": tgt["region"],
                            "condition": cond.name,
                            "probe_idx": probe_idx,
                            "repeat": rep,
                            "answer": answer,
                            "error": error,
                            "says_unknown": (not error) and "unknown" in answer.lower(),
                            "believes": bool(
                                cond.persona and not error and mentions_country(answer, cond.persona)
                            ),
                        }
                    )
        pd.DataFrame(rows).to_csv(belief_path, index=False)
        logger.info(f"Belief check done for {tgt['model']}")
    return pd.DataFrame(rows)


def estimate_calls(
    n_targets: int, n_conditions: int, n_prompts: int, repeats: int, n_probes: int = len(BELIEF_PROBES)
) -> dict[str, int]:
    trials = n_targets * n_conditions * n_prompts * repeats
    probes = n_targets * n_conditions * n_probes * repeats
    return {"target_calls": trials + probes, "judge_calls": trials, "trials": trials}


def count_prompts(objectives: pd.DataFrame, techniques: pd.DataFrame, shots: int) -> int:
    per_obj = sum(1 if n == DIRECT_TECHNIQUE else shots for n in techniques["name"])
    return len(objectives) * per_obj


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="CSV: model;region;developer")
    ap.add_argument("--objectives", type=Path, nargs="+", default=DEFAULT_OBJECTIVES)
    ap.add_argument("--techniques", type=Path, nargs="+", default=DEFAULT_TECHNIQUES)
    ap.add_argument("--personas", nargs="+", default=list(DEFAULT_PERSONAS), help="ISO codes, e.g. US NL CN")
    ap.add_argument("--deliveries", nargs="+", default=["system"], choices=DELIVERIES)
    ap.add_argument("--shots", type=int, default=2, help="attack prompts per objective/technique (attacked techniques)")
    ap.add_argument("--repeats", type=int, default=1, help="times each target answers the same prompt")
    ap.add_argument("--out-dir", type=Path, default=Path("results/geo"))
    ap.add_argument("--retry-errors", action="store_true", help="send trials that failed with an API error again")
    ap.add_argument("--skip-belief", action="store_true", help="skip the manipulation check")
    ap.add_argument("--dry-run", action="store_true", help="only print the number of API calls")
    ap.add_argument("--analyze-only", action="store_true", help="only analyse existing result files")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("geo")

    results_path = args.out_dir / "results.csv"
    belief_path = args.out_dir / "belief.csv"

    if not args.analyze_only:
        targets = pd.read_csv(args.targets, delimiter=";")
        objectives = load_objectives(args.objectives)
        techniques = load_techniques(args.techniques)
        conditions = build_conditions(args.personas, args.deliveries)

        n_prompts = count_prompts(objectives, techniques, args.shots)
        est = estimate_calls(len(targets), len(conditions), n_prompts, args.repeats)
        logger.info(
            f"{len(targets)} targets x {len(conditions)} conditions x {n_prompts} prompts x "
            f"{args.repeats} repeats = {est['trials']} trials "
            f"(~{est['target_calls']} target calls + {est['judge_calls']} judge calls)"
        )
        if args.dry_run:
            return

        source = os.getenv("LLM_SOURCE", "OpenRouter")

        def llm_factory(model: str) -> Any:
            return get_LLM(source=source, model=model)

        attacker = llm_factory(os.getenv("ATTACKER_LLM", "NO_LLM_SET"))
        judge = llm_factory(os.getenv("JUDGE_LLM", "NO_LLM_SET"))

        prompts = generate_attack_prompts(
            attacker, objectives, techniques, args.shots, args.out_dir / "attack_prompts.csv", logger
        )
        run_attacks(
            targets, judge, llm_factory, prompts, objectives, conditions, args.repeats,
            results_path, logger, retry_errors=args.retry_errors,
        )
        if not args.skip_belief:
            run_belief_check(
                targets, llm_factory, conditions, args.repeats, belief_path, logger,
                retry_errors=args.retry_errors,
            )

    results = pd.read_csv(results_path)
    belief = pd.read_csv(belief_path) if belief_path.exists() else None
    tables = analysis.write_report(results, belief, args.out_dir)

    pd.set_option("display.width", 200)
    print("\nBaseline mean score per region and topic (each model counts once):")
    print(tables["region_baseline"].round(2).to_string(index=False))
    effects = tables["persona_effects"]
    if not effects.empty:
        print("\nPersona effect (paired, vs baseline):")
        print(effects.round(3).to_string(index=False))
    if "belief_rates" in tables:
        print("\nManipulation check (share of probes naming the persona's country):")
        print(tables["belief_rates"].round(2).to_string(index=False))
    print(f"\nTables and plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
