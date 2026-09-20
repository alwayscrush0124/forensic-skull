"""Sanity check：把標籤隨機打亂後訓練一折。沒有洩漏的話 val AUC 應在 0.5 附近。
用法：python sanity_shuffle.py --data preprocessed --epochs 10
"""
import argparse, os, random
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from train_cv import load_cases, run_fold, WINDOW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="preprocessed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.window = WINDOW; args.cooldown = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    ids, X, y = load_cases(args.data, os.path.join(args.data, "labels_anon.csv"), smoke=False)
    y_shuf = np.random.permutation(y)
    print(f"標籤打亂：與原標籤一致比例 {np.mean(y_shuf == y):.2f}（純隨機約 0.5）", flush=True)
    tr, va = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X, y_shuf)))
    history = []
    run_fold(0, tr, va, X, y_shuf, args, device, history)
    print("val AUC per epoch:", [round(h["val_auc"], 3) for h in history])


if __name__ == "__main__":
    main()
