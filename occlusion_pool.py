"""把多折的 occlusion 結果合併：九宮格質量比例（合併 + 各折）、群體平均圖。
用法：python occlusion_pool.py --folds 0 1 2 --out results/occlusion_pooled
"""
import argparse, json, os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_cv import load_cases

YN, ZN, ZCUT = ["posterior", "middle", "anterior"], ["base", "mid", "vault"], [0, 13, 27, 40]


def grid_fractions(m):
    m = np.clip(m, 0, None); tot = m.sum()
    return {f"{yn}/{zn}": round(float(m[:, a * 80:(a + 1) * 80, ZCUT[b]:ZCUT[b + 1]].sum() / tot), 3)
            for a, yn in enumerate(YN) for b, zn in enumerate(ZN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed")
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    ids, X, y = load_cases(args.data, os.path.join(args.data, "labels_anon.csv"), smoke=False, bone_only=True)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X, y))
    maps, yv, correct, fold_of, case_ids = [], [], [], [], []
    for k in args.folds:
        s = json.load(open(f"results/occlusion_fold{k}/occ_summary.json"))
        m = np.load(f"results/occlusion_fold{k}/occ_maps.npy").astype(np.float32)
        assert [c["case_id"] for c in s["cases"]] == [ids[i] for i in splits[k][1]], f"fold {k} 名單不符"
        maps.append(m); yv += [c["gender"] for c in s["cases"]]; fold_of += [k] * len(s["cases"]); case_ids += [c["case_id"] for c in s["cases"]]
        correct += [(c["p_female"] > 0.5) == (c["gender"] == 1) for c in s["cases"]]
    maps, yv, correct, fold_of = np.concatenate(maps), np.array(yv), np.array(correct), np.array(fold_of)
    va_all = np.concatenate([splits[k][1] for k in args.folds])

    summary = {"folds": args.folds, "n": int(len(yv)), "n_correct": int(correct.sum())}
    groups = {"female_correct": (yv == 1) & correct, "male_correct": (yv == 0) & correct}
    for g, sel in groups.items():
        summary[g] = {"n": int(sel.sum()), "pooled": grid_fractions(maps[sel].mean(0)),
                      "per_fold": {int(k): grid_fractions(maps[sel & (fold_of == k)].mean(0)) for k in args.folds}}
    json.dump(summary, open(os.path.join(args.out, "occ_pooled_summary.json"), "w"), indent=2)

    for g in groups:
        print(f"\n{g}  n={summary[g]['n']}  (pooled | per fold {args.folds})")
        print(f"{'':10s} {'posterior':>22s} {'middle':>22s} {'anterior':>22s}")
        for zn in ["vault", "mid", "base"]:
            cells = []
            for yn in YN:
                key = f"{yn}/{zn}"; pf = " ".join(f"{summary[g]['per_fold'][k][key]:.2f}" for k in args.folds)
                cells.append(f"{summary[g]['pooled'][key]:.3f} ({pf})")
            print(f"{zn:10s} " + " ".join(f"{c:>22s}" for c in cells))

    win = lambda v: np.clip((v - 300) / 1000, 0, 1)
    zs = [4, 12, 20, 28, 36]
    fig, ax = plt.subplots(2, len(zs) + 1, figsize=(2.6 * (len(zs) + 1), 5.6))
    for row, (g, sel) in enumerate(groups.items()):
        m, mb = maps[sel].mean(0), win(X[va_all[sel]].astype(np.float32)).mean(0); vmax = np.abs(m).max()
        for col, z in enumerate(zs):
            a = ax[row, col]; a.imshow(mb[:, :, z].T, cmap="gray", origin="lower")
            a.imshow(m[:, :, z].T, cmap="RdBu_r", alpha=0.5, origin="lower", vmin=-vmax, vmax=vmax); a.axis("off")
            if row == 0: a.set_title(f"z={z}", fontsize=8)
        a = ax[row, -1]; a.imshow(mb[120].T, cmap="gray", origin="lower", aspect=5)
        a.imshow(m[120].T, cmap="RdBu_r", alpha=0.5, origin="lower", aspect=5, vmin=-vmax, vmax=vmax); a.axis("off")
        if row == 0: a.set_title("mid-sagittal (ant→right)", fontsize=8)
        ax[row, 0].text(-40, 120, f"{g}\nn={int(sel.sum())}  folds {args.folds}", fontsize=8, rotation=90, va="center")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "occ_pooled_mean.png"), dpi=110); plt.close(fig)
    print("\nsaved:", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
