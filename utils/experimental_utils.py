import os, csv, json, torch, random, numpy as np
from datetime import datetime

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_dir(root, name, seed):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(root, f"{name}_{ts}_seed{seed}")
    os.makedirs(exp_dir)
    for sub in ["logs", "checkpoints"]:
        os.makedirs(os.path.join(exp_dir, sub))
    return exp_dir

class CSVLogger:
    def __init__(self, path, header):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

def save_checkpoint(path, model, optimizer, epoch):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
