"""v2 訓練／評估：5-fold stratified CV，顱骨性別分類。

讀 preprocessed/<case_id>.nii.gz（int16 HU，同 spacing 同 shape）+ labels_anon.csv。
- 固定 HU 骨窗，不做逐例 min-max（v1 的問題）
- 每個 fold 跑固定 epoch 數，不用 val 挑 epoch（避免用 val 選模型灌水）
- 結果 = 全部 out-of-fold 預測合併算 AUC + bootstrap 95% CI，旁邊放 majority baseline
用法：python train_cv.py --data preprocessed --out results/run1 [--epochs 30] [--bone_only] [--smoke]
--bone_only：只保留 >300 HU 的最大連通元件（顱骨），window 改 [300,1300]；頭髮、軟組織、頭架、耳環歸零
"""
import argparse, csv, json, os, random, time
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from monai.networks.nets import DenseNet121
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW = (-500.0, 1300.0)   # HU 骨窗 W1800 / L400
WINDOW_BONE = (300.0, 1300.0)
BONE_HU = 300
SHIFT = (8, 8, 2)           # 增強：隨機平移 voxel 上限（x, y, z）


def keep_largest_bone(d):
    lab, _ = ndi.label(d > BONE_HU)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return np.where(lab == sizes.argmax(), d, np.int16(-1024))


def load_cases(data_dir, labels_csv, smoke, bone_only=False):
    rows = [r for r in csv.DictReader(open(labels_csv)) if not r["exclude"]]
    if smoke:
        rows = rows[:12]
    ids = [r["case_id"] for r in rows]
    y = np.array([int(r["gender"]) for r in rows])
    X = np.stack([np.asanyarray(nib.load(os.path.join(data_dir, f"{c}.nii.gz")).dataobj) for c in ids])
    if bone_only:
        X = np.stack([keep_largest_bone(x) for x in X])
    assert X.dtype == np.int16, X.dtype
    return ids, X, y


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
            out.append(torch.sigmoid(model(to_input(X[i:i + bs], device, False, window))).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


def run_fold(k, tr, va, X, y, args, device, history):
    torch.manual_seed(args.seed + k); random.seed(args.seed + k)
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.from_numpy(y).float()
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); perm = np.random.permutation(tr); tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = to_input(X[idx], device, True, args.window), yt[idx].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb).squeeze(1).float(), yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        p = evaluate(model, X[va], device, args.bs, args.window)
        auc = roc_auc_score(y[va], p) if len(set(y[va])) == 2 else float("nan")
        history.append({"fold": k, "epoch": ep, "train_loss": tot / len(tr), "val_auc": auc,
                        "val_acc": float(((p > 0.5) == y[va]).mean())})
        print(f"fold {k} ep {ep:2d}/{args.epochs} | loss {tot / len(tr):.4f} | val AUC {auc:.3f} "
              f"acc {history[-1]['val_acc']:.3f} | {time.time() - t0:.0f}s", flush=True)
    return p, model                                                          # 最後一個 epoch 的 OOF 預測


def bootstrap_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed")
    ap.add_argument("--labels", default=None, help="預設 <data>/labels_anon.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bone_only", action="store_true", help="只留顱骨（最大連通骨元件），window [300,1300]")
    ap.add_argument("--cooldown", type=int, default=0, help="每個 fold 結束後休息幾秒（降溫用）")
    ap.add_argument("--smoke", action="store_true", help="12 例、2 fold、2 epoch，只驗證程式能跑")
    args = ap.parse_args()
    args.window = WINDOW_BONE if args.bone_only else WINDOW
    if args.smoke:
        args.folds, args.epochs = 2, 2
    labels = args.labels or os.path.join(args.data, "labels_anon.csv")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    ids, X, y = load_cases(args.data, labels, args.smoke, args.bone_only)
    print(f"device {device} | bone_only={args.bone_only} window={args.window} | n={len(y)} (female={y.sum()}, male={len(y) - y.sum()}) | X {X.shape} {X.dtype}", flush=True)

    oof = np.full(len(y), np.nan); fold_of = np.zeros(len(y), int); history = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        oof[va], _ = run_fold(k, tr, va, X, y, args, device, history); fold_of[va] = k
        if args.cooldown and k < args.folds - 1:
            print(f"cooldown {args.cooldown}s", flush=True); time.sleep(args.cooldown)   # 每折之間讓機器降溫
    assert not np.isnan(oof).any()

    pred = (oof > 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fold_aucs = [roc_auc_score(y[fold_of == k], oof[fold_of == k]) for k in range(args.folds)]
    metrics = {
        "n": int(len(y)), "n_female": int(y.sum()), "folds": args.folds, "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
        "window_hu": args.window, "bone_only": args.bone_only, "model": "DenseNet121-3D (MONAI)", "device": device.type,
        "oof_auc": float(roc_auc_score(y, oof)), "oof_auc_ci95": bootstrap_auc(y, oof),
        "fold_auc_mean": float(np.mean(fold_aucs)), "fold_auc_sd": float(np.std(fold_aucs)), "fold_aucs": fold_aucs,
        "acc": float((pred == y).mean()), "sens_female": float(cm[1, 1] / cm[1].sum()), "spec_male": float(cm[0, 0] / cm[0].sum()),
        "majority_baseline_acc": float(max(y.mean(), 1 - y.mean())), "confusion_matrix_rows_true_male_female": cm.tolist(),
    }
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(args.out, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id", "gender", "fold", "prob_female"])
        w.writerows(zip(ids, y, fold_of, np.round(oof, 4)))
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fpr, tpr, _ = roc_curve(y, oof)
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title=f"OOF ROC  AUC={metrics['oof_auc']:.3f} (95% CI {metrics['oof_auc_ci95'][0]:.3f}–{metrics['oof_auc_ci95'][1]:.3f})",
              xlabel="1 - specificity", ylabel="sensitivity (female)")
    for k in range(args.folds):
        h = [r for r in history if r["fold"] == k]
        ax[1].plot([r["epoch"] for r in h], [r["val_auc"] for r in h], label=f"fold {k}")
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8); ax[1].set(title="val AUC per epoch", xlabel="epoch", ylim=(0.2, 1)); ax[1].legend(fontsize=8)
    ax[2].imshow(cm, cmap="Blues"); ax[2].set(title=f"OOF confusion  acc={metrics['acc']:.3f}  (majority={metrics['majority_baseline_acc']:.3f})",
                                              xticks=[0, 1], yticks=[0, 1], xticklabels=["pred M", "pred F"], yticklabels=["true M", "true F"])
    for (i, j), v in np.ndenumerate(cm):
        ax[2].text(j, i, str(v), ha="center", va="center", fontsize=16, color="white" if v > cm.max() / 2 else "black")
    plt.tight_layout(); fig.savefig(os.path.join(args.out, "summary.png"), dpi=120)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
