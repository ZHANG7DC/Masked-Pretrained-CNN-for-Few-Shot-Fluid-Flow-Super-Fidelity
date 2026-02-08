import os
from datetime import datetime
import argparse
import json
import yaml
import random
from vit import MAE
from burgers_loader import PDEDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from vit import ViT, MAE, Decoder

def train(epoch: int,
          channels: list,
          encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          batch: dict,
          device: torch.device = torch.device("cuda:0")) -> None:
    """
    Train the model for one epoch using gradient accumulation.
    
    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The MAE model.
        optimizer (torch.optim.Optimizer): Optimizer.
        loader (DataLoader): Data loader for training data.
        device (torch.device): Device to run training on.
    """
    encoder.train()
    decoder.train()
    running_loss = 0.0

    LF = batch['LF'][:,channels].float().to(device)
    HF = batch['HF'].float().to(device)
    param = batch['param'].float().to(device)
    optimizer.zero_grad()
    out = encoder(LF,param,'mae_tokens')
    dout = decoder(out['mae_tokens'])
    recon_HF = encoder.reconstruct_sim(LF,dout,torch.arange(dout.shape[1]).unsqueeze(0).to(device))
    loss = F.mse_loss(recon_HF[:,decoder.channels-1], HF)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Finetuning script for MAE decoder")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    cmd_args = parser.parse_args()

    # Read configuration from YAML file
    with open(cmd_args.config, "r") as f:
        config = yaml.safe_load(f)
    print(config)

    # Set seeds for reproducibility
    #torch.manual_seed(config["seed"])
    device = torch.device(config["device"])

    # Create dataset and dataloader
    dataset = PDEDataset(config["data_path"], config["names"], config["indices_path"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    channels = config['channels']
                         
    # Build model (initialize encoder and decoder)
    vit = ViT(
        image_size=config["image_size"],
        channels = len(channels),
        param_dim=config['param_dim'],
        patch_size=config['patch_size'],
        dim=config['encoder_dim'],
        depth=config['encoder_depth'],
        heads=config['encoder_heads'],
        mlp_dim=config['encoder_mlp_dim']
    )
    encoder = MAE(encoder=vit, decoder_dim=config['decoder_dim'], masking_ratio=0.0).float()
    encoder.to(device)

    # Optionally load pretrained weights if available
    if config["pretrained_path"] != 'none':
        pretrain_weights = torch.load(config["pretrained_path"])                     
        encoder.load_state_dict(pretrain_weights)

    for param in encoder.parameters():
        param.requires_grad = False
    decoder = Decoder(num_patches=vit.pos_embedding.shape[1]*len(channels), pixel_values_per_patch=vit.sim_to_patch_embedding[0][1].weight.shape[-1], decoder_dim=config['decoder_dim'], channels=len(channels))
    decoder.to(device)
    if config["pretrained_path"] != 'none':
        decoder.load_state_dict(pretrain_weights, strict=False)
    trainable_params = list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainable_params),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"])
    )

    # Create directory for saving model checkpoints if it doesn't exist
    os.makedirs(config["save_path"], exist_ok=True)
    # Training loop over epochs
    batch = next(iter(train_loader))
    pbar = tqdm(range(1, config["num_epochs"] + 1), dynamic_ncols=True)
    for epoch in iter(pbar):
        avg_loss = train(epoch, channels, encoder, decoder, optimizer, batch, device)
        checkpoint_path = os.path.join(config["save_path"], f"{epoch}.pth")
        if epoch % 1000 == 0:
            torch.save({'encoder':encoder.state_dict(),'decoder':decoder.state_dict()}, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        pbar.set_description(f'epoch {epoch} loss {avg_loss}')
if __name__ == '__main__':
    main()
