"""尺寸 baseline：只用顱骨大小（骨體素數、骨 bbox 三軸長度）做 logistic regression。

目的：檢查 CNN 的 AUC 有多少只是「頭大 = 男」。fold 切法與 train_cv.py 完全相同（同 seed、同順序）。
用法：python size_baseline.py --data preprocessed --out results/size_baseline
"""
import argparse, csv, json, os
import numpy as np
import nibabel as nib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from train_cv import bootstrap_auc

BONE_HU = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(os.path.join(args.data, "labels_anon.csv"))) if not r["exclude"]]
    y = np.array([int(r["gender"]) for r in rows]); feats = []
    for r in rows:
        d = np.asanyarray(nib.load(os.path.join(args.data, f"{r['case_id']}.nii.gz")).dataobj)
        bone = d > BONE_HU; idx = np.where(bone)
        feats.append([bone.sum()] + [i.max() - i.min() + 1 for i in idx])      # 骨體素數、bbox x/y/z
    X = np.array(feats, float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results = {}
    for name, cols in [("bone_voxels_only", [0]), ("bbox_xyz_only", [1, 2, 3]), ("all_4", [0, 1, 2, 3])]:
        oof = np.zeros(len(y))
        for tr, va in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr][:, cols], y[tr])
            oof[va] = clf.predict_proba(X[va][:, cols])[:, 1]
        results[name] = {"oof_auc": float(roc_auc_score(y, oof)), "oof_auc_ci95": bootstrap_auc(y, oof),
                         "acc": float(((oof > 0.5) == y).mean())}
        print(f"{name:18s} AUC {results[name]['oof_auc']:.3f}  CI {results[name]['oof_auc_ci95'][0]:.3f}–{results[name]['oof_auc_ci95'][1]:.3f}  acc {results[name]['acc']:.3f}")
    results["feature_means_male_female"] = {k: [float(X[y == 0, i].mean()), float(X[y == 1, i].mean())]
                                            for i, k in enumerate(["bone_voxels", "bbox_x", "bbox_y", "bbox_z"])}
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
