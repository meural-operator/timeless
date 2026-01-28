import os, json, sys, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.dataset_bc import BCDataset
from utils.experiment_utils import *
from models.deeponet import DeepONet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(cfg_path):
    cfg = load_config(cfg_path)
    seed_everything(cfg["experiment"]["seed"])

    exp_dir = create_experiment_dir(
        cfg["experiment"]["root_dir"],
        cfg["experiment"]["name"],
        cfg["experiment"]["seed"]
    )

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    logger = CSVLogger(os.path.join(exp_dir, "logs/train.csv"),
                       ["epoch", "train_mse"])

    dataset = BCDataset(cfg["dataset"]["root"], cfg["dataset"]["seq_len"])
    loader = DataLoader(dataset,
                        cfg["optimization"]["batch_size"],
                        shuffle=True)

    u0, _, cond = dataset[0]
    H, W = u0.shape[-2:]
    branch_dim = u0.numel() + cond.numel()

    x = torch.linspace(0,1,H)
    y = torch.linspace(0,1,W)
    grid = torch.stack(torch.meshgrid(x,y,indexing="ij"),
                       dim=-1).view(-1,2).to(DEVICE)

    model = DeepONet(
        branch_dim=branch_dim,
        trunk_dim=2,
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"]
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["optimization"]["learning_rate"])

    ckpt_path = os.path.join(exp_dir, "checkpoints/last.pth")
    start_epoch = load_checkpoint(ckpt_path, model, opt) + 1 \
                  if os.path.exists(ckpt_path) else 0

    for epoch in range(start_epoch, cfg["optimization"]["epochs"]):
        loss_acc = 0.0
        for u0, y, c in loader:
            B = u0.shape[0]
            branch = torch.cat([u0.view(B,-1), c], dim=1).to(DEVICE)
            trunk = grid.unsqueeze(0).expand(B,-1,-1)

            target = y[:,0].to(DEVICE).view(B,2,-1).permute(0,2,1)
            pred = model(branch, trunk)

            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_acc += loss.item()

        loss_acc /= len(loader)
        logger.log([epoch, loss_acc])

        save_checkpoint(ckpt_path, model, opt, epoch)
        if epoch % cfg["checkpoint"]["save_every"] == 0:
            save_checkpoint(
                os.path.join(exp_dir, f"checkpoints/epoch_{epoch:03d}.pth"),
                model, opt, epoch
            )

        print(f"[DeepONet] Epoch {epoch:03d} | MSE {loss_acc:.3e}")

if __name__ == "__main__":
    main(sys.argv[1])
