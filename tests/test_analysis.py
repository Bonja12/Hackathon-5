import numpy as np
import pandas as pd

from provile import analysis


def make_df(rows):
    cols = ["model", "region", "condition", "objective", "category", "technique", "prompt_idx", "repeat", "score", "error"]
    return pd.DataFrame(rows, columns=cols)


def test_is_error_response():
    assert analysis.is_error_response("[ERROR] NO RESPONSE CREATED")
    assert analysis.is_error_response("   ")
    assert analysis.is_error_response(None)
    assert analysis.is_error_response("No target response could be generated")
    assert not analysis.is_error_response("I can't help with that.")


def test_summarise_excludes_errors_and_reports_error_rate():
    df = make_df(
        [
            ("m", "US", "baseline", "o", "c", "t", 0, 0, 4, False),
            ("m", "US", "baseline", "o", "c", "t", 1, 0, 1, False),
            ("m", "US", "baseline", "o", "c", "t", 2, 0, np.nan, True),
        ]
    )
    out = analysis.summarise(df, ["model"]).iloc[0]
    assert out["n"] == 2
    assert out["mean_score"] == 2.5
    assert out["share_full"] == 0.5 and out["share_refusal"] == 0.5
    assert abs(out["error_rate"] - 1 / 3) < 1e-9


def test_judge_failure_score_zero_is_excluded():
    df = make_df([("m", "US", "baseline", "o", "c", "t", 0, 0, 0, False)])
    assert analysis.valid_rows(df).empty


def test_permutation_test_detects_consistent_shift_and_ignores_noise():
    assert analysis.paired_permutation_test(np.full(20, -2.0), seed=1) < 0.001
    assert analysis.paired_permutation_test(np.zeros(10)) == 1.0
    noise = np.array([1, -1] * 10, dtype=float)
    assert analysis.paired_permutation_test(noise, seed=1) > 0.9


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = analysis.bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert lo <= 3.0 <= hi


def test_persona_effects_are_paired_against_baseline():
    rows = []
    for i in range(12):
        rows.append(("m", "US", "baseline", "o", "c", "t", i, 0, 4, False))
        rows.append(("m", "US", "CN/system", "o", "c", "t", i, 0, 1, False))
    eff = analysis.persona_effects(make_df(rows)).iloc[0]
    assert eff["mean_delta"] == -3
    assert eff["n_pairs"] == 12
    assert eff["p_value"] < 0.01


def test_persona_effects_skip_unpaired_rows():
    rows = [
        ("m", "US", "baseline", "o", "c", "t", 0, 0, 4, False),
        ("m", "US", "CN/system", "o", "c", "t", 99, 0, 1, False),
    ]
    assert analysis.persona_effects(make_df(rows)).empty


def test_region_table_counts_each_model_once():
    rows = []
    for _ in range(10):  # model a has many rows, model b only one
        rows.append(("a", "Asia", "baseline", "o", "c", "t", len(rows), 0, 4, False))
    rows.append(("b", "Asia", "baseline", "o", "c", "t", 99, 0, 2, False))
    out = analysis.region_table(make_df(rows)).iloc[0]
    assert out["n_models"] == 2
    assert out["mean_score"] == 3.0


def test_belief_rates_ignore_baseline_and_errors():
    belief = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "region": ["US"] * 4,
            "condition": ["baseline", "CN/system", "CN/system", "CN/system"],
            "believes": [False, True, False, True],
            "error": [False, False, False, True],
        }
    )
    out = analysis.belief_rates(belief).iloc[0]
    assert out["n"] == 2 and out["belief_rate"] == 0.5


def test_write_report_creates_tables_and_plots(tmp_path):
    rows = []
    for i in range(6):
        for model, region in (("a", "US"), ("b", "Asia")):
            rows.append((model, region, "baseline", "o", "china_critical", "t", i, 0, 4, False))
            rows.append((model, region, "CN/system", "o", "china_critical", "t", i, 0, 2, False))
    belief = pd.DataFrame(
        {"model": ["a", "b"], "region": ["US", "Asia"], "condition": ["CN/system"] * 2,
         "believes": [True, True], "error": [False, False]}
    )
    analysis.write_report(make_df(rows), belief, tmp_path)
    for name in ("persona_effects.csv", "region_baseline.csv", "baseline_heatmap.png",
                 "persona_delta_heatmap.png", "belief_heatmap.png"):
        assert (tmp_path / name).exists(), name
