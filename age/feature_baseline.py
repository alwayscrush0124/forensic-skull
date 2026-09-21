"""簡單特徵 baseline：顱骨大小（骨體素數、bbox 三軸）與骨密度代理（骨體素平均 HU）做 ridge regression 預測年齡。

目的：看 CNN 的 MAE 有多少是簡單特徵就能拿到的。fold 切法與 train_age_cv.py 完全相同（同 seed、同年齡分箱）。
用法：python feature_baseline.py --data ../preprocessed --out results/feature_baseline
"""
import argparse, csv, json, os
import numpy as np
import nibabel as nib
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_age_cv import age_bins, bootstrap_mae, BONE_HU

FEATS = ["bone_voxels", "bbox_x", "bbox_y", "bbox_z", "bone_mean_hu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../preprocessed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(os.path.join(args.data, "labels_anon.csv"))) if not r["exclude"]]
    y = np.array([float(r["age"]) for r in rows]); feats = []
    for r in rows:
        d = np.asanyarray(nib.load(os.path.join(args.data, f"{r['case_id']}.nii.gz")).dataobj)
        bone = d > BONE_HU; idx = np.where(bone)
        feats.append([bone.sum()] + [i.max() - i.min() + 1 for i in idx] + [d[bone].mean()])   # 骨體素數、bbox x/y/z、骨平均 HU
    X = np.array(feats, float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed); bins = age_bins(y)
    results = {}
    for name, cols in [("size_only", [0, 1, 2, 3]), ("bone_hu_only", [4]), ("all_5", [0, 1, 2, 3, 4])]:
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, bins):
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr][:, cols], y[tr])
            oof[va] = reg.predict(X[va][:, cols])
        results[name] = {"oof_mae": float(np.abs(oof - y).mean()), "oof_mae_ci95": bootstrap_mae(y, oof),
                         "pearson_r": float(np.corrcoef(y, oof)[0, 1])}
        print(f"{name:14s} MAE {results[name]['oof_mae']:5.2f}  CI {results[name]['oof_mae_ci95'][0]:.2f}–{results[name]['oof_mae_ci95'][1]:.2f}  r {results[name]['pearson_r']:+.3f}")
    base = np.zeros(len(y))
    for tr, va in skf.split(X, bins):
        base[va] = y[tr].mean()
    results["mean_baseline_mae"] = float(np.abs(base - y).mean())
    results["feature_corr_with_age"] = {k: float(np.corrcoef(X[:, i], y)[0, 1]) for i, k in enumerate(FEATS)}
    print(f"mean baseline  MAE {results['mean_baseline_mae']:5.2f}")
    print("corr with age:", {k: round(v, 3) for k, v in results["feature_corr_with_age"].items()})
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
