"""可解釋性：重訓純骨版 fold 0，對其 held-out 病例做 Grad-CAM，看模型看顱骨的哪裡。

- 女性病例：Grad-CAM 對 +logit（推向「女」的區域）；男性病例：對 −logit（推向「男」的區域）
- 層：features.denseblock2（30×30×5，約 8 mm 面內、40 mm 縱向）；denseblock4 只有 7×7×1，z 方向沒有解析度
輸出：results/explain_fold0/ model.pt、cam_examples.png、cam_mean.png、cam_summary.json
用法：python explain.py --data preprocessed --out results/explain_fold0 [--epochs 30]
"""
import argparse, json, os, random
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from monai.visualize import GradCAM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_cv import load_cases, run_fold, to_input, WINDOW_BONE

LAYER = "features.denseblock2"


class Neg(nn.Module):                      # 男性用：對 −logit 做 CAM
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return -self.m(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.window = WINDOW_BONE; args.cooldown = 0
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    ids, X, y = load_cases(args.data, os.path.join(args.data, "labels_anon.csv"), smoke=False, bone_only=True)
    tr, va = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X, y)))
    history = []
    p, model = run_fold(0, tr, va, X, y, args, device, history)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    model.eval()

    # --- Grad-CAM，每例一張 [240,240,40]，值域 [0,1] ---
    cam_f, cam_m = GradCAM(nn_module=model, target_layers=LAYER), GradCAM(nn_module=Neg(model), target_layers="m." + LAYER)
    cams = np.zeros((len(va),) + X.shape[1:], np.float32)
    for j, i in enumerate(va):
        x = to_input(X[i:i + 1], device, False, args.window)
        c = (cam_f if y[i] == 1 else cam_m)(x=x, class_idx=0)
        cams[j] = c[0, 0].float().cpu().numpy()
    np.save(os.path.join(args.out, "cams_fold0_val.npy"), cams.astype(np.float16))

    # --- 摘要：CAM 質量在各解剖區的比例（RAS：x 右、y 前、z 上）---
    correct = (p > 0.5) == y[va]
    def mass_frac(c, region):
        return float(c[region].sum() / c.sum())
    xx, yy, zz = np.meshgrid(np.arange(240), np.arange(240), np.arange(40), indexing="ij")
    regions = {"anterior_half": yy >= 120, "superior_half": zz >= 20, "right_half": xx >= 120,
               "z_top_third": zz >= 27, "z_mid_third": (zz >= 13) & (zz < 27), "z_bottom_third": zz < 13}
    summary = {"layer": LAYER, "n_val": int(len(va)), "n_correct": int(correct.sum()),
               "val_auc_last_epoch": history[-1]["val_auc"], "val_acc_last_epoch": history[-1]["val_acc"]}
    for grp, sel in [("female_correct", (y[va] == 1) & correct), ("male_correct", (y[va] == 0) & correct)]:
        mean_cam = cams[sel].mean(0)
        summary[grp] = {"n": int(sel.sum()), **{k: round(mass_frac(mean_cam, r), 3) for k, r in regions.items()}}
    json.dump(summary, open(os.path.join(args.out, "cam_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # --- 圖 1：個案。最有信心且判對的女 3、男 3，各三個 axial 高度 ---
    win = lambda v: np.clip((v - 300) / 1000, 0, 1)
    conf = np.where(y[va] == 1, p, 1 - p) * correct
    pick = [j for j in np.argsort(-conf) if y[va][j] == 1][:3] + [j for j in np.argsort(-conf) if y[va][j] == 0][:3]
    levels = [10, 20, 30]
    fig, ax = plt.subplots(len(levels), len(pick), figsize=(2.6 * len(pick), 2.6 * len(levels)))
    for col, j in enumerate(pick):
        vol = X[va[j]]
        for row, z in enumerate(levels):
            a = ax[row, col]; a.imshow(win(vol[:, :, z]).T, cmap="gray", origin="lower")
            a.imshow(cams[j][:, :, z].T, cmap="jet", alpha=0.4, origin="lower", vmin=0, vmax=1); a.axis("off")
            if row == 0: a.set_title(f"{ids[va[j]]}  {'F' if y[va[j]] else 'M'}  p(F)={p[j]:.2f}", fontsize=8)
            if col == 0: a.text(-30, 120, f"z={z}", fontsize=8, rotation=90, va="center")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "cam_examples.png"), dpi=110); plt.close(fig)

    # --- 圖 2：群體平均。判對的女／男各一列，五個 z 切面 + 正中矢狀 ---
    zs = [4, 12, 20, 28, 36]
    fig, ax = plt.subplots(2, len(zs) + 1, figsize=(2.6 * (len(zs) + 1), 5.6))
    for row, (grp, sel) in enumerate([("female (evidence for F)", (y[va] == 1) & correct), ("male (evidence for M)", (y[va] == 0) & correct)]):
        mean_cam, mean_bone = cams[sel].mean(0), win(X[va[sel]].astype(np.float32)).mean(0)
        for col, z in enumerate(zs):
            a = ax[row, col]; a.imshow(mean_bone[:, :, z].T, cmap="gray", origin="lower")
            a.imshow(mean_cam[:, :, z].T, cmap="jet", alpha=0.45, origin="lower", vmin=0, vmax=mean_cam.max()); a.axis("off")
            a.set_title(f"z={z}" if row == 0 else "", fontsize=8)
        a = ax[row, -1]; a.imshow(mean_bone[120, :, :].T, cmap="gray", origin="lower", aspect=5)
        a.imshow(mean_cam[120, :, :].T, cmap="jet", alpha=0.45, origin="lower", aspect=5, vmin=0, vmax=mean_cam.max()); a.axis("off")
        a.set_title("mid-sagittal" if row == 0 else "", fontsize=8)
        ax[row, 0].text(-40, 120, f"{grp}\nn={int(sel.sum())}", fontsize=8, rotation=90, va="center")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "cam_mean.png"), dpi=110); plt.close(fig)
    print("saved:", os.listdir(args.out))


if __name__ == "__main__":
    main()
