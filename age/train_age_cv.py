"""顱骨 CT 年齡迴歸：5-fold CV，DenseNet121-3D，L1 loss。

和 ../train_cv.py（性別分類）用同一套前處理輸出、同一個網路、同一組訓練設定；差別只在：
- 標籤 gender → age（float），目標值以「訓練折」的 mean/SD 標準化後再回推
- BCE → L1 loss；輸出不過 sigmoid
- 分折用年齡五分位分箱做 stratified（避免某折沒有年輕人）
- 指標：MAE / RMSE / Pearson r + bootstrap 95% CI；baseline = 猜訓練折平均年齡；另附各年齡層 MAE 與偏差
用法：python train_age_cv.py --data ../preprocessed --out results/age_fullhead [--epochs 30] [--bone_only] [--smoke]
--bone_only：只保留 >300 HU 的最大連通元件，window 改 [300,1300]。注意這會把顱內鈣化（松果體、頸動脈虹吸部等）
            一併去掉，那可能是年齡的重要線索；預設用全頭部窗 [-500,1300]。
"""
import argparse, csv, json, os, random, time
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from monai.networks.nets import DenseNet121
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW = (-500.0, 1300.0)   # HU 骨窗 W1800 / L400
WINDOW_BONE = (300.0, 1300.0)
BONE_HU = 300
SHIFT = (8, 8, 2)           # 增強：隨機平移 voxel 上限（x, y, z）
AGE_GROUPS = [18, 40, 50, 60, 70, 80, 100]   # 分層報告用；18–39 合併，因為 40 歲以下只有 12 例


def keep_largest_bone(d):
    lab, _ = ndi.label(d > BONE_HU)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return np.where(lab == sizes.argmax(), d, np.int16(-1024))


def load_cases(data_dir, labels_csv, smoke, bone_only=False):
    rows = [r for r in csv.DictReader(open(labels_csv)) if not r["exclude"]]
    if smoke:
        rows = rows[:12]
    ids = [r["case_id"] for r in rows]
    y = np.array([float(r["age"]) for r in rows])
    sex = np.array([int(r["gender"]) for r in rows])
    X = np.stack([np.asanyarray(nib.load(os.path.join(data_dir, f"{c}.nii.gz")).dataobj) for c in ids])
    if bone_only:
        X = np.stack([keep_largest_bone(x) for x in X])
    assert X.dtype == np.int16, X.dtype
    return ids, X, y, sex


def age_bins(y, n=5):
    """年齡分位數分箱，只用來做 stratified 分折。"""
    return np.digitize(y, np.percentile(y, np.linspace(0, 100, n + 1)[1:-1]))


def to_input(x_int16, device, augment, window):
    x = torch.from_numpy(x_int16).to(device).float().unsqueeze(1)          # [B,1,X,Y,Z]
    x = (x.clamp(*window) - window[0]) / (window[1] - window[0])
    if augment:
        if random.random() < 0.5:
            x = x.flip(2)                                                    # 左右翻轉
        x = torch.roll(x, shifts=[random.randint(-s, s) for s in SHIFT], dims=(2, 3, 4))
    return x


def evaluate(model, X, device, bs, window):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(to_input(X[i:i + bs], device, False, window)).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


def run_fold(k, tr, va, X, y, args, device, history):
    torch.manual_seed(args.seed + k); random.seed(args.seed + k)
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.L1Loss()
    mu, sd = y[tr].mean(), y[tr].std()                                       # 目標標準化只看訓練折
    yt = torch.from_numpy((y - mu) / sd).float()
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); perm = np.random.permutation(tr); tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = to_input(X[idx], device, True, args.window), yt[idx].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb).squeeze(1).float(), yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        p = evaluate(model, X[va], device, args.bs, args.window) * sd + mu    # 回推成歲數
        mae = float(np.abs(p - y[va]).mean())
        history.append({"fold": k, "epoch": ep, "train_loss": tot / len(tr), "val_mae": mae})
        print(f"fold {k} ep {ep:2d}/{args.epochs} | loss {tot / len(tr):.4f} | val MAE {mae:5.2f} | {time.time() - t0:.0f}s", flush=True)
    return p                                                                 # 最後一個 epoch 的 OOF 預測


def bootstrap_mae(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    maes = [np.abs(y[i] - p[i]).mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))


def by_age_group(y, p):
    out = {}
    for lo, hi in zip(AGE_GROUPS[:-1], AGE_GROUPS[1:]):
        m = (y >= lo) & (y < hi)
        if m.any():
            out[f"{lo}-{hi - 1}"] = {"n": int(m.sum()), "mae": float(np.abs(p[m] - y[m]).mean()), "bias": float((p[m] - y[m]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../preprocessed")
    ap.add_argument("--labels", default=None, help="預設 <data>/labels_anon.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bone_only", action="store_true", help="只留顱骨（最大連通骨元件），window [300,1300]")
    ap.add_argument("--smoke", action="store_true", help="12 例、2 fold、2 epoch，只驗證程式能跑")
    args = ap.parse_args()
    args.window = WINDOW_BONE if args.bone_only else WINDOW
    if args.smoke:
        args.folds, args.epochs = 2, 2
    labels = args.labels or os.path.join(args.data, "labels_anon.csv")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    ids, X, y, sex = load_cases(args.data, labels, args.smoke, args.bone_only)
    print(f"device {device} | bone_only={args.bone_only} window={args.window} | n={len(y)} age {y.min():.0f}–{y.max():.0f} "
          f"mean {y.mean():.1f} sd {y.std():.1f} | X {X.shape} {X.dtype}", flush=True)

    oof = np.full(len(y), np.nan); base = np.full(len(y), np.nan); fold_of = np.zeros(len(y), int); history = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(X, age_bins(y, 2 if args.smoke else 5))):
        oof[va] = run_fold(k, tr, va, X, y, args, device, history); base[va] = y[tr].mean(); fold_of[va] = k
    assert not np.isnan(oof).any()

    err = oof - y
    fold_maes = [float(np.abs(err[fold_of == k]).mean()) for k in range(args.folds)]
    metrics = {
        "n": int(len(y)), "folds": args.folds, "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
        "window_hu": args.window, "bone_only": args.bone_only, "model": "DenseNet121-3D (MONAI)", "device": device.type,
        "age_mean": float(y.mean()), "age_sd": float(y.std()),
        "oof_mae": float(np.abs(err).mean()), "oof_mae_ci95": bootstrap_mae(y, oof),
        "oof_rmse": float(np.sqrt((err ** 2).mean())), "pearson_r": float(np.corrcoef(y, oof)[0, 1]), "mean_bias": float(err.mean()),
        "fold_mae_mean": float(np.mean(fold_maes)), "fold_mae_sd": float(np.std(fold_maes)), "fold_maes": fold_maes,
        "mean_baseline_mae": float(np.abs(base - y).mean()),
        "by_age_group": by_age_group(y, oof),
    }
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(args.out, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id", "gender", "age", "fold", "pred_age"])
        w.writerows(zip(ids, sex, y.astype(int), fold_of, np.round(oof, 1)))
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].scatter(y, oof, s=14); ax[0].plot([15, 100], [15, 100], "k--", lw=0.8)
    ax[0].set(title=f"OOF pred vs true  MAE={metrics['oof_mae']:.1f} (95% CI {metrics['oof_mae_ci95'][0]:.1f}–{metrics['oof_mae_ci95'][1]:.1f})  r={metrics['pearson_r']:.2f}",
              xlabel="true age", ylabel="predicted age", xlim=(15, 100), ylim=(15, 100))
    for k in range(args.folds):
        h = [r for r in history if r["fold"] == k]
        ax[1].plot([r["epoch"] for r in h], [r["val_mae"] for r in h], label=f"fold {k}")
    ax[1].axhline(metrics["mean_baseline_mae"], color="k", ls="--", lw=0.8, label="guess mean")
    ax[1].set(title="val MAE per epoch", xlabel="epoch", ylabel="MAE (years)"); ax[1].legend(fontsize=8)
    g = metrics["by_age_group"]; names = list(g)
    ax[2].bar(names, [g[n]["mae"] for n in names])
    for i, n in enumerate(names):
        ax[2].text(i, g[n]["mae"] + 0.3, f"bias {g[n]['bias']:+.1f}\nn={g[n]['n']}", ha="center", fontsize=8)
    ax[2].set(title="MAE by age group (bias = pred − true)", xlabel="true age", ylabel="MAE (years)")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "summary.png"), dpi=120)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
