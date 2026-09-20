"""Occlusion sensitivity：把一塊顱骨換成空氣，看 logit 變多少。用 explain_fold0/model.pt 對同一批 held-out 病例做。

importance = 遮掉該區後「真實類別的證據」減少多少：
  女性病例 = −Δlogit（遮掉後更不像女 → 正值）；男性病例 = +Δlogit（遮掉後更像女 → 正值）
輸出：results/occlusion_fold0/ occ_maps.npy、occ_summary.json、occ_mean.png、occ_examples.png
用法：python occlusion.py --data preprocessed --model results/explain_fold0/model.pt --out results/occlusion_fold0
"""
import argparse, json, os, time
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from monai.networks.nets import DenseNet121
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_cv import load_cases, to_input, WINDOW_BONE

CUBE = (24, 24, 8)      # voxel = 24×24×40 mm
STRIDE = (12, 12, 4)
MIN_BONE = 50           # 遮罩內骨體素少於此就跳過（空氣區不用遮）


def occlude_case(model, vol, device, bs=16):
    x0 = to_input(vol[None], device, False, WINDOW_BONE)
    with torch.no_grad():
        base = model(x0).item()
    bone = vol > 300
    pos = [(ix, iy, iz) for ix in range(0, 240 - CUBE[0] + 1, STRIDE[0]) for iy in range(0, 240 - CUBE[1] + 1, STRIDE[1])
           for iz in range(0, 40 - CUBE[2] + 1, STRIDE[2]) if bone[ix:ix + CUBE[0], iy:iy + CUBE[1], iz:iz + CUBE[2]].sum() >= MIN_BONE]
    deltas = np.zeros(len(pos), np.float32)
    with torch.no_grad():
        for b in range(0, len(pos), bs):
            chunk = pos[b:b + bs]; xb = x0.repeat(len(chunk), 1, 1, 1, 1)
            for k, (ix, iy, iz) in enumerate(chunk):
                xb[k, :, ix:ix + CUBE[0], iy:iy + CUBE[1], iz:iz + CUBE[2]] = 0.0        # 換成空氣
            deltas[b:b + len(chunk)] = model(xb).squeeze(1).float().cpu().numpy() - base
    return base, pos, deltas


def to_volume(pos, vals):
    acc, cnt = np.zeros((240, 240, 40), np.float32), np.zeros((240, 240, 40), np.float32)
    for (ix, iy, iz), v in zip(pos, vals):
        acc[ix:ix + CUBE[0], iy:iy + CUBE[1], iz:iz + CUBE[2]] += v; cnt[ix:ix + CUBE[0], iy:iy + CUBE[1], iz:iz + CUBE[2]] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed")
    ap.add_argument("--model", default="results/explain_fold0/model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 例（測試用）")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device)); model.eval()

    ids, X, y = load_cases(args.data, os.path.join(args.data, "labels_anon.csv"), smoke=False, bone_only=True)
    _, va = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X, y)))
    if args.limit: va = va[:args.limit]
    maps, rows = np.zeros((len(va), 240, 240, 40), np.float32), []
    for j, i in enumerate(va):
        t0 = time.time(); base, pos, d = occlude_case(model, X[i], device)
        imp = -d if y[i] == 1 else d
        maps[j] = to_volume(pos, imp)
        k = int(np.argmax(imp)); cx, cy, cz = [p + c // 2 for p, c in zip(pos[k], CUBE)]
        rows.append({"case_id": ids[i], "gender": int(y[i]), "base_logit": round(base, 3), "p_female": round(float(1 / (1 + np.exp(-base))), 3),
                     "n_cubes": len(pos), "max_importance": round(float(imp.max()), 3), "top_cube_center_xyz": [cx, cy, cz],
                     "importance_sum_pos": round(float(imp[imp > 0].sum()), 2)})
        print(f"[{j + 1}/{len(va)}] {ids[i]} {'F' if y[i] else 'M'} p(F)={rows[-1]['p_female']:.2f} cubes={len(pos)} "
              f"max imp={imp.max():.2f} at {(cx, cy, cz)} | {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(args.out, "occ_maps.npy"), maps.astype(np.float16))

    # --- 摘要：正 importance 的質量在 y（後/中/前）× z（底/中/頂）九宮格的比例 ---
    ynames, znames = ["posterior", "middle", "anterior"], ["base", "mid", "vault"]
    summary = {"cube_mm": [CUBE[0], CUBE[1], CUBE[2] * 5], "n_val": int(len(va)), "cases": rows}
    yv, sel_groups = np.array([r["gender"] for r in rows]), {}
    correct = np.array([(r["p_female"] > 0.5) == (r["gender"] == 1) for r in rows])
    for grp, sel in [("female_correct", (yv == 1) & correct), ("male_correct", (yv == 0) & correct)]:
        m = np.clip(maps[sel].mean(0), 0, None); tot = m.sum(); grid = {}
        for a, yn in enumerate(ynames):
            for b, zn in enumerate(znames):
                ysl, zsl = slice(a * 80, (a + 1) * 80), slice([0, 13, 27][b], [13, 27, 40][b])
                grid[f"{yn}/{zn}"] = round(float(m[:, ysl, zsl].sum() / tot), 3)
        lat = m[:80].sum() + m[160:].sum()
        summary[grp] = {"n": int(sel.sum()), "mass_fraction_y_z": grid, "lateral_thirds_x": round(float(lat / tot), 3)}
        sel_groups[grp] = sel
    json.dump(summary, open(os.path.join(args.out, "occ_summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))

    # --- 圖 1：群體平均（判對的女 / 男），五個 z + 正中矢狀；紅 = 遮掉會失去真實類別證據 ---
    win = lambda v: np.clip((v - 300) / 1000, 0, 1)
    zs = [4, 12, 20, 28, 36]
    fig, ax = plt.subplots(2, len(zs) + 1, figsize=(2.6 * (len(zs) + 1), 5.6))
    for row, (grp, sel) in enumerate(sel_groups.items()):
        m, mb = maps[sel].mean(0), win(X[va[sel]].astype(np.float32)).mean(0); vmax = np.abs(m).max()
        for col, z in enumerate(zs):
            a = ax[row, col]; a.imshow(mb[:, :, z].T, cmap="gray", origin="lower")
            a.imshow(m[:, :, z].T, cmap="RdBu_r", alpha=0.5, origin="lower", vmin=-vmax, vmax=vmax); a.axis("off")
            if row == 0: a.set_title(f"z={z}", fontsize=8)
        a = ax[row, -1]; a.imshow(mb[120].T, cmap="gray", origin="lower", aspect=5)
        a.imshow(m[120].T, cmap="RdBu_r", alpha=0.5, origin="lower", aspect=5, vmin=-vmax, vmax=vmax); a.axis("off")
        if row == 0: a.set_title("mid-sagittal (ant→right)", fontsize=8)
        ax[row, 0].text(-40, 120, f"{grp}\nn={int(sel.sum())}", fontsize=8, rotation=90, va="center")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "occ_mean.png"), dpi=110); plt.close(fig)

    # --- 圖 2：個案（最有信心判對的女 3、男 3），三個 z ---
    conf = np.array([r["p_female"] if r["gender"] else 1 - r["p_female"] for r in rows]) * correct
    pick = [j for j in np.argsort(-conf) if yv[j] == 1][:3] + [j for j in np.argsort(-conf) if yv[j] == 0][:3]
    levels = [10, 20, 30]
    fig, ax = plt.subplots(len(levels), len(pick), figsize=(2.6 * len(pick), 2.6 * len(levels)))
    for col, j in enumerate(pick):
        vol, m = X[va[j]], maps[j]; vmax = np.abs(m).max()
        for row, z in enumerate(levels):
            a = ax[row, col]; a.imshow(win(vol[:, :, z]).T, cmap="gray", origin="lower")
            a.imshow(m[:, :, z].T, cmap="RdBu_r", alpha=0.5, origin="lower", vmin=-vmax, vmax=vmax); a.axis("off")
            if row == 0: a.set_title(f"{rows[j]['case_id']} {'F' if yv[j] else 'M'} p(F)={rows[j]['p_female']:.2f}", fontsize=8)
            if col == 0: a.text(-30, 120, f"z={z}", fontsize=8, rotation=90, va="center")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "occ_examples.png"), dpi=110); plt.close(fig)
    print("saved:", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
