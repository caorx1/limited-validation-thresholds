"""Check V2 source and configuration summaries from saved task summaries."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
METRICS = [
    "macro_f1", "gain", "negative_rate", "drop_gt_1pp_rate", "D",
    "positive_f1", "precision", "recall", "predicted_positive_rate",
]
SOURCE_KEYS = ["source", "budget", "arm", "method"]
CONFIG_KEYS = ["configuration", "budget", "method"]


def compare(rebuilt, expected, keys, metrics):
    a = rebuilt.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if not (a.index.is_unique and b.index.is_unique and a.index.equals(b.index)):
        raise ValueError(f"Missing or duplicate keys for {keys}")
    delta = (a[metrics] - b[metrics]).abs().to_numpy()
    if not np.isfinite(delta).all() or delta.max() >= 1e-12:
        raise ValueError(f"Numerical mismatch for {keys}")
    return int(delta.size), float(delta.max())


def run():
    tasks = pd.read_csv(TABLES / "checked_v2_task_metrics.csv")
    sizes = tasks.groupby(SOURCE_KEYS).size()
    if len(sizes) != 32 or not (sizes == 9).all():
        raise ValueError("Expected nine model–partition summaries per source, budget, arm and rule")
    sources = tasks.groupby(SOURCE_KEYS, as_index=False)[METRICS].mean()
    source_ref = pd.read_csv(TABLES / "checked_v2_source_metrics.csv")
    source_count, source_error = compare(sources, source_ref, SOURCE_KEYS, METRICS)

    primary = pd.read_csv(TABLES / "all_strategy_source_metrics.csv").rename(columns={
        "strategy": "method", "mean_macro_f1": "macro_f1", "mean_delta_vs_s0": "gain",
        "negative_draw_fraction": "negative_rate", "drop_gt_01_fraction": "drop_gt_1pp_rate",
        "decision_instability": "D",
    })
    common = METRICS[:5]
    rows = []
    changes = {
        "original": [], "PageValues_only": ["online_shoppers"],
        "Customer_Value_only": ["iranian_churn"], "both_fields": ["online_shoppers", "iranian_churn"],
    }
    for configuration, changed in changes.items():
        for budget in [100, 500]:
            for method in ["S0", "S1", "S2", "S3"]:
                unchanged = primary.loc[
                    (primary.budget == budget) & (primary.method == method) & ~primary.source.isin(changed),
                    ["source", *common],
                ]
                included = sources.loc[
                    (sources.budget == budget) & (sources.method == method)
                    & (sources.arm == "include") & sources.source.isin(changed), ["source", *common],
                ]
                merged = pd.concat([unchanged, included], ignore_index=True)
                if len(merged) != 5 or merged.source.nunique() != 5:
                    raise ValueError("Expected five distinct, equally weighted sources")
                rows.append(dict(configuration=configuration, budget=budget, method=method,
                                 **merged[common].mean().to_dict()))
    configurations = pd.DataFrame(rows)
    config_ref = pd.read_csv(TABLES / "five_source_information_set_sensitivity.csv")
    config_count, config_error = compare(configurations, config_ref, CONFIG_KEYS, common)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    sources.to_csv(out / "rebuilt_v2_source_metrics.csv", index=False)
    configurations.to_csv(out / "rebuilt_information_set_sensitivity.csv", index=False)
    report = {
        "operation": "reaggregate saved V2 task summaries",
        "task_rows": len(tasks),
        "source_checks": source_count, "source_max_abs_error": source_error,
        "configuration_checks": config_count, "configuration_max_abs_error": config_error,
        "new_model_fits": 0, "new_threshold_searches": 0,
    }
    (out / "v2_rebuild_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
