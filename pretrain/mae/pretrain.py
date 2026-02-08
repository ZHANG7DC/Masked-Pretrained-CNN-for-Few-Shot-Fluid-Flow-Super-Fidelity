import os
from datetime import datetime
import argparse
import json
import yaml
import random
from vit import MAE
from dataloader import PDEDataset, AFDataset, WFDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from vit import ViT, MAE
def train(epoch: int,
          channels: list,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          batch: torch.tensor,
          device: torch.device = torch.device("cuda:0")) -> None:
    """
    Train the model for one epoch using gradient accumulation.
    
    Args:
        args (dict): Dictionary of training parameters.
        epoch (int): Current epoch number.
        model (torch.nn.Module): The MAE model.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        loader (DataLoader): Data loader for training data.
        device (torch.device): Device to run training on.
        channels (list): LF channels used for pretraining
    """
    model.train()
    LF = batch[:,channels].float().to(device)
    optimizer.zero_grad()
    loss = model(LF)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()
    
def main():
    parser = argparse.ArgumentParser(description="Training script for MAE")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    cmd_args = parser.parse_args()

    # Read configuration from the provided YAML file path
    with open(cmd_args.config, "r") as f:
        config = yaml.safe_load(f)
    print(config)
    # Set seeds for reproducibility
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    channels = config['channels']
    # Create dataset and dataloader
    if config["pde"] == 'burgers':
        dataset = PDEDataset(config["data_path"], config["names"])
    elif config["pde"] == 'airfoil':
        dataset = AFDataset(config['data_path'])
    elif config['pde'] == 'windfarm':
        dataset = WFDataset(config["data_path"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    # Build model (replace DummyMAE with your actual MAE model implementation)
    encoder = ViT(image_size=config["image_size"], image_channels=config['image_channels'], patch_size=config['patch_size'], dim=config['encoder_dim'], depth=config['encoder_depth'], heads=config['encoder_heads'], mlp_dim=config['encoder_mlp_dim'], channels=len(channels))
    model = MAE(encoder=encoder, decoder_dim=config['decoder_dim']).float()
    model.to(device)

    # Optionally, load pretrained weights if available
    if config["pretrained_path"] != "none":
        model.load_state_dict(torch.load(config["pretrained_path"]))

    # Create optimizer (using AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_lr"],
        betas=(config["beta1"], config["beta2"])
    )

    # Set up OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["num_epochs"],
        pct_start=config["pct_start"],
        div_factor=config["div_factor"],
        final_div_factor=config["final_div_factor"]
    )

    # Create directory for saving model checkpoints if it doesn't exist
    os.makedirs(config["save_path"], exist_ok=True)
    # Training loop over epochs
    batch = next(iter(train_loader))
    pbar = tqdm(range(1, config["num_epochs"] + 1), dynamic_ncols=True)
    for epoch in iter(pbar):
        loss = train(epoch, channels, model, optimizer, scheduler, batch, device)
        checkpoint_path = os.path.join(config["save_path"], f"mae_epoch_{epoch}.pth")
        if epoch%1000 == 0:
            torch.save(model.state_dict(), checkpoint_path)
        pbar.set_description(f'loss: {loss}')
if __name__ == '__main__':
    main()