

python - <<'PY'
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

raw = pd.read_csv("results/defense_raw/raw_requests_0_vs_12.csv")
by = pd.read_csv("results/defense_raw/raw_combined_by_split.csv")

for ds in ["llama", "qwen", "combined"]:
    d = raw.copy() if ds == "combined" else raw[raw["dataset"] == ds].copy()
    d = d.reset_index(drop=True)

    y = (d["offload_gb"] == 12).astype(int).values
    groups = d["run_concurrency"].values

    mapping = []
    for split_id, (_, test_idx) in enumerate(LeaveOneGroupOut().split(d, y, groups)):
        heldout = sorted(d.iloc[test_idx]["run_concurrency"].unique())
        mapping.append({
            "dataset": ds,
            "split_id": split_id,
            "heldout_concurrency": ",".join(map(str, heldout)),
        })

    mapping = pd.DataFrame(mapping)

    filt = by[
        (by["dataset"] == ds) &
        (by["cv"] == "loco") &
        (by["defense"] == "none") &
        (by["attacker_type"] == "adaptive") &
        (by["classifier"] == "LogReg") &
        (by["featureset"] == "all_primary")
    ].copy()

    out = filt.merge(mapping, on=["dataset", "split_id"], how="left")

    print("\n==", ds, "==")
    print(out[[
        "split_id",
        "heldout_concurrency",
        "balanced_accuracy",
        "auroc",
        "accuracy",
        "n_low_test",
        "n_high_test",
    ]].sort_values("split_id").to_string(index=False))
PY
