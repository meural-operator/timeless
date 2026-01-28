import os, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset

class BCDataset(Dataset):
    def __init__(self, root, seq_len=20, split="train"):
        self.seq_len = seq_len
        cases = sorted(glob.glob(os.path.join(root, "case*")))
        split_idx = int(0.9 * len(cases))
        self.cases = cases[:split_idx] if split == "train" else cases[split_idx:]

        self.samples = []
        for c in self.cases:
            u = np.load(os.path.join(c, "u.npy"))
            v = np.load(os.path.join(c, "v.npy"))
            data = np.stack([u, v], axis=1)

            meta = json.load(open(os.path.join(c, "case.json")))
            cond = np.array([
                meta["Re"],
                meta["radius"],
                meta["inlet_velocity"],
                meta["bc_type"]
            ], dtype=np.float32)

            for t in range(0, data.shape[0] - seq_len - 1, 5):
                self.samples.append((data[t], data[t+1:t+1+seq_len], cond))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y, c = self.samples[i]
        return (
            torch.tensor(x).float(),
            torch.tensor(y).float(),
            torch.tensor(c).float()
        )
