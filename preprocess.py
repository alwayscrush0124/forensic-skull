"""前處理 v2：只做幾何，不做強度正規化（window 留給 Dataset）。

每例：讀 NIfTI → 轉 RAS → 重取樣到固定 spacing（世界座標，處理軸序不一）
     → 以骨組織 bbox 中心裁切/補零到固定 shape → 存 int16 NIfTI（保留 HU，可用 Slicer 看）
輸出：preprocessed/<case_id>.nii.gz、preprocessed/manifest.csv、preprocessed/qa_contact_sheet.png
失敗即中止，不靜默跳過（v1 的教訓）。
"""
import csv, os, sys
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOX = (1.0, 1.0, 5.0)        # mm；z 保留原始厚度，不造假切片
TARGET = (240, 240, 40)      # voxels；骨 bbox p95 ≈ 229×223×36，max 250×238×38
BONE_HU = 300
AIR_HU = -1024

SRC = "nifti_data"
LABELS = "labels_cleaned.csv"
OUT = "preprocessed"


def center_crop_pad(data, center, target):
    out = np.full(target, AIR_HU, dtype=data.dtype)
    for_src, for_out = [], []
    for c, t, s in zip(center, target, data.shape):
        lo = int(round(c - t / 2))
        src_lo, src_hi = max(lo, 0), min(lo + t, s)
        out_lo = src_lo - lo
        for_src.append(slice(src_lo, src_hi))
        for_out.append(slice(out_lo, out_lo + (src_hi - src_lo)))
    out[tuple(for_out)] = data[tuple(for_src)]
    return out


def process_one(path):
    img = nib.as_closest_canonical(nib.load(path))
    orig_zooms = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])
    res = resample_to_output(img, voxel_sizes=VOX, order=1, cval=AIR_HU)
    data = np.asanyarray(res.dataobj).astype(np.float32)
    bone = data > BONE_HU
    if not bone.any():
        raise RuntimeError(f"no bone voxels (> {BONE_HU} HU) in {path}")
    idx = np.where(bone)
    bbox = [(int(i.min()), int(i.max()) + 1) for i in idx]
    center = [(lo + hi) / 2 for lo, hi in bbox]
    cropped = center_crop_pad(data, center, TARGET)
    cropped = np.clip(cropped, AIR_HU, 3000).astype(np.int16)
    affine = res.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ [int(round(c - t / 2)) for c, t in zip(center, TARGET)]
    return nib.Nifti1Image(cropped, affine), {
        "orig_shape": img.shape, "orig_zooms": orig_zooms,
        "bbox_size": [hi - lo for lo, hi in bbox],
        "clipped": any(hi - lo > t for (lo, hi), t in zip(bbox, TARGET)),
        "note": "orig z-spacing on axis 0" if orig_zooms[0] > 2 else "",
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(LABELS)))
    manifest, thumbs = [], []
    for i, r in enumerate(rows, 1):
        case_id = f"skull_{i:03d}"
        img, info = process_one(os.path.join(SRC, r["filename"] + ".nii.gz"))
        nib.save(img, os.path.join(OUT, case_id + ".nii.gz"))
        manifest.append({"case_id": case_id, "filename": r["filename"], "gender": r["gender"],
                         "age": r["age"], "adult": int(float(r["age"]) >= 18), **info})
        d = np.asanyarray(img.dataobj)
        thumbs.append((case_id, d[:, :, d.shape[2] // 2].T, d[:, d.shape[1] // 2, :].T))
        print(f"[{i:3d}/{len(rows)}] {case_id}  {info['orig_shape']} {info['orig_zooms']} -> {img.shape}"
              f"{'  CLIPPED' if info['clipped'] else ''}{'  ' + info['note'] if info['note'] else ''}")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys()); w.writeheader(); w.writerows(manifest)

    n = len(thumbs); cols = 12; rws = -(-n // cols)
    fig, axes = plt.subplots(rws * 2, cols, figsize=(cols * 1.6, rws * 2 * 1.6))
    for ax in axes.flat: ax.axis("off")
    for k, (cid, ax_sl, cor_sl) in enumerate(thumbs):
        rr, cc = divmod(k, cols)
        axes[rr * 2, cc].imshow(ax_sl, cmap="gray", vmin=-200, vmax=1500, origin="lower")
        axes[rr * 2, cc].set_title(cid, fontsize=6)
        axes[rr * 2 + 1, cc].imshow(cor_sl, cmap="gray", vmin=-200, vmax=1500, origin="lower", aspect=VOX[2] / VOX[0])
    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "qa_contact_sheet.png"), dpi=110)
    print(f"\ndone: {n} cases -> {OUT}/  (manifest.csv, qa_contact_sheet.png)")


if __name__ == "__main__":
    main()
