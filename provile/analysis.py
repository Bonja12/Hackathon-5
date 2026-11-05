"""Statistics and plots for the geopolitical / persona experiments.

Design choices worth knowing:

* Rows with an API error, or without a valid judge score (1-4), are excluded from
  every mean and reported separately. An API error is not a refusal.
* Persona effects are *paired*: the same model answers the same attack prompt
  with and without the persona, so the difference isolates the persona.
* Region effects are computed over models (every model counts once). Region is
  perfectly confounded with developer and model, so with only a handful of
  models per region these numbers are descriptive, not significant.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PAIR_KEY = ["model", "objective", "technique", "prompt_idx", "repeat"]
BASELINE = "baseline"

# One-hue sequential for magnitude, two-hue diverging with neutral midpoint for change.
SEQUENTIAL = "Reds"
DIVERGING = LinearSegmentedColormap.from_list(
    "cool_neutral_warm", ["#2b6cb0", "#e6e6e6", "#c0392b"]
)


def is_error_response(text: object) -> bool:
    """True for the placeholders the pipeline writes when no real answer exists."""
    if not isinstance(text, str):
        return True
    t = text.strip()
    return (
        not t
        or t.startswith("[ERROR]")
        or t == "No target response could be generated"
    )


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real answer and a judge score in 1..4."""
    return df[(~df["error"].astype(bool)) & df["score"].between(1, 4)]


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean score, share of full compliance (4), share of refusals (1), error rate."""
    ok = valid_rows(df)
    stats = ok.groupby(by).agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        share_full=("score", lambda s: float((s == 4).mean())),
        share_refusal=("score", lambda s: float((s == 1).mean())),
    )
    errors = df.groupby(by)["error"].mean().rename("error_rate")
    return stats.join(errors, how="outer").reset_index()


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_permutation_test(
    diffs: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided sign-flip test on paired differences (H0: mean difference is 0)."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0 or np.all(diffs == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed - 1e-12) + 1) / (n_perm + 1))


def persona_effects(df: pd.DataFrame, n_perm: int = 10000) -> pd.DataFrame:
    """Per model and persona condition: mean score change versus baseline.

    A positive delta means the model complied more once it believed the user was
    from that country.
    """
    ok = valid_rows(df)
    base = ok[ok["condition"] == BASELINE].set_index(PAIR_KEY)["score"]
    rows = []
    for (model, region, condition), grp in ok[ok["condition"] != BASELINE].groupby(
        ["model", "region", "condition"]
    ):
        paired = grp.set_index(PAIR_KEY)["score"].to_frame("score").join(
            base.rename("baseline"), how="inner"
        )
        if paired.empty:
            continue
        diffs = (paired["score"] - paired["baseline"]).to_numpy()
        lo, hi = bootstrap_ci(diffs)
        rows.append(
            {
                "model": model,
                "region": region,
                "condition": condition,
                "n_pairs": len(diffs),
                "mean_delta": float(diffs.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "p_value": paired_permutation_test(diffs, n_perm=n_perm),
                "share_changed": float((diffs != 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def persona_effects_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean paired delta per model, persona condition and topic category."""
    ok = valid_rows(df)
    base = ok[ok["condition"] == BASELINE].set_index(PAIR_KEY)["score"]
    parts = []
    for (model, condition, category), grp in ok[ok["condition"] != BASELINE].groupby(
        ["model", "condition", "category"]
    ):
        paired = grp.set_index(PAIR_KEY)["score"].to_frame("score").join(
            base.rename("baseline"), how="inner"
        )
        if not paired.empty:
            parts.append(
                {
                    "model": model,
                    "condition": condition,
                    "category": category,
                    "n_pairs": len(paired),
                    "mean_delta": float((paired["score"] - paired["baseline"]).mean()),
                }
            )
    return pd.DataFrame(parts)


def region_table(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline mean score per developer region and topic category.

    Each model is averaged first, then models are averaged per region, so a
    region with many prompts or many models does not outweigh the others.
    """
    base = df[df["condition"] == BASELINE]
    per_model = summarise(base, ["region", "model", "category"])
    out = (
        per_model.groupby(["region", "category"])
        .agg(n_models=("model", "nunique"), mean_score=("mean_score", "mean"))
        .reset_index()
    )
    return out


def belief_rates(belief_df: pd.DataFrame) -> pd.DataFrame:
    """Share of probe answers that name the persona's country (manipulation check)."""
    ok = belief_df[~belief_df["error"].astype(bool) & (belief_df["condition"] != BASELINE)]
    return (
        ok.groupby(["model", "region", "condition"])["believes"]
        .agg(belief_rate="mean", n="size")
        .reset_index()
    )


# --------------------------------------------------------------------- plots


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_baseline_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Mean baseline score, model (rows, grouped by region) x topic category."""
    base = summarise(df[df["condition"] == BASELINE], ["region", "model", "category"])
    if base.empty:
        return
    table = base.pivot_table(index=["region", "model"], columns="category", values="mean_score")
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * table.shape[1] + 4), 0.6 * table.shape[0] + 2))
    sns.heatmap(
        table, vmin=1, vmax=4, cmap=SEQUENTIAL, annot=True, fmt=".1f",
        linewidth=2, linecolor="white", ax=ax, cbar_kws={"label": "mean judge score (1-4)"},
    )
    ax.set(title="Baseline score per model and topic (no persona)", xlabel="", ylabel="")
    _save(fig, path)


def plot_persona_delta(effects: pd.DataFrame, path: Path) -> None:
    """Paired score change vs baseline, model x persona condition."""
    if effects.empty:
        return
    table = effects.pivot_table(index=["region", "model"], columns="condition", values="mean_delta")
    sig = effects.pivot_table(index=["region", "model"], columns="condition", values="p_value")
    annot = table.map(lambda v: f"{v:+.2f}") + np.where(sig < 0.05, "*", "")
    bound = max(0.5, float(np.nanmax(np.abs(table.to_numpy()))))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * table.shape[1] + 4), 0.6 * table.shape[0] + 2))
    sns.heatmap(
        table, vmin=-bound, vmax=bound, center=0, cmap=DIVERGING, annot=annot.to_numpy(),
        fmt="", linewidth=2, linecolor="white", ax=ax,
        cbar_kws={"label": "score change vs baseline"},
    )
    ax.set(
        title="Score change vs baseline when the user is believed to be from a country\n(* p<0.05, uncorrected for multiple tests)",
        xlabel="persona / delivery", ylabel="",
    )
    _save(fig, path)


def plot_belief(rates: pd.DataFrame, path: Path) -> None:
    if rates.empty:
        return
    table = rates.pivot_table(index=["region", "model"], columns="condition", values="belief_rate")
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * table.shape[1] + 4), 0.6 * table.shape[0] + 2))
    sns.heatmap(
        table, vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f",
        linewidth=2, linecolor="white", ax=ax,
        cbar_kws={"label": "share of probes naming the persona's country"},
    )
    ax.set(title="Does the model accept the persona? (manipulation check)", xlabel="", ylabel="")
    _save(fig, path)


def write_report(results: pd.DataFrame, belief: pd.DataFrame | None, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Write all tables and plots to ``out_dir`` and return the tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "scores_by_model_condition": summarise(results, ["region", "model", "condition"]),
        "scores_by_model_category": summarise(results, ["region", "model", "condition", "category"]),
        "region_baseline": region_table(results),
        "persona_effects": persona_effects(results),
        "persona_effects_by_category": persona_effects_by_category(results),
    }
    if belief is not None and not belief.empty:
        tables["belief_rates"] = belief_rates(belief)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    plot_baseline_heatmap(results, out_dir / "baseline_heatmap.png")
    plot_persona_delta(tables["persona_effects"], out_dir / "persona_delta_heatmap.png")
    if "belief_rates" in tables:
        plot_belief(tables["belief_rates"], out_dir / "belief_heatmap.png")
    return tables
