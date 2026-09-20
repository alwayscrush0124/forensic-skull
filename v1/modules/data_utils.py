import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import pandas as pd
import os

class SkullDataset(Dataset):
    # 這裡必須要有 target_shape
    def __init__(self, nifti_dir, label_csv, target_shape=(512, 512, 32)):
        self.nifti_dir = nifti_dir
        self.labels = pd.read_csv(label_csv)
        self.target_shape = target_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 確保第一欄是檔名，第二欄是標籤
        img_id = str(self.labels.iloc[idx, 0])
        label = self.labels.iloc[idx, 1]
        
        img_path = os.path.join(self.nifti_dir, f"{img_id}.nii.gz")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.nifti_dir, f"{img_id.lower()}.nii.gz")

        try:
            img = nib.load(img_path).get_fdata()
            img = np.clip(img, 0, 2000) / 2000.0
            img = self.resize_3d(img, self.target_shape)
        except Exception as e:
            # 發生錯誤時回傳零張量，避免訓練中斷
            img = np.zeros(self.target_shape)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        return img_tensor, torch.tensor(label).long()

    def resize_3d(self, data, target_shape):
        factors = [t / s for t, s in zip(target_shape, data.shape)]
        return ndimage.zoom(data, factors, order=1)
