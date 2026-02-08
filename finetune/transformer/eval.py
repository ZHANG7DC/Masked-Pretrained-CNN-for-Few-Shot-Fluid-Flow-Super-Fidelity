import os
from datetime import datetime
import argparse
import json
import yaml
import random
from vit import MAE, ViT, Decoder
from burgers_loader import PDEDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate(encoder: torch.nn.Module,
             decoder: torch.nn.Module,
             loader: DataLoader,
             channels: list,
             device: torch.device = torch.device("cuda:0")) -> np.ndarray:
    """
    Evaluate the model and collect MSE losses per batch.
    
    Args:
        encoder (torch.nn.Module): The MAE encoder.
        decoder (torch.nn.Module): The decoder module.
        loader (DataLoader): Data loader for evaluation data.
        channels (list): List of channel indices to use.
        device (torch.device): Device for computation.
        
    Returns:
        mse_array (np.ndarray): Array of MSE losses (one per batch).
    """
    encoder.eval()
    decoder.eval()
    mse_losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            LF = batch['LF'][:, channels].float().to(device)
            HF = batch['HF'].float().to(device)
            param = batch['param'].float().to(device)
            
            # Forward pass through encoder and decoder.
            out = encoder(LF, param, 'mae_tokens')
            dout = decoder(out['mae_tokens'])
            
            # Reconstruct the high-frequency image.
            recon_HF = encoder.reconstruct_sim(
                LF,
                dout,
                torch.arange(dout.shape[1]).unsqueeze(0).to(device)
            )
            
            # Compute the MSE loss.
            loss = F.mse_loss(recon_HF[:, decoder.channels - 1], HF)**0.5
            mse_losses.append(loss.item())
            
    mse_array = np.array(mse_losses)
    return mse_array

def main():
    parser = argparse.ArgumentParser(description="Evaluation script for MAE decoder")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    cmd_args = parser.parse_args()

    # Read configuration from YAML file.
    with open(cmd_args.config, "r") as f:
        config = yaml.safe_load(f)
    print(config)

    # Set seed and device.
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])

    # Create evaluation dataset and dataloader.
    dataset = PDEDataset(config["data_path"], config["names"])
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    channels = config['channels']

    # Build encoder (ViT) and wrap with MAE.
    vit = ViT(
        image_size=config["image_size"],
        channels=len(channels),
        param_dim=config['param_dim'],
        patch_size=config['patch_size'],
        dim=config['encoder_dim'],
        depth=config['encoder_depth'],
        heads=config['encoder_heads'],
        mlp_dim=config['encoder_mlp_dim']
    )
    encoder = MAE(encoder=vit, decoder_dim=config['decoder_dim'], masking_ratio=0.0).float()
    encoder.to(device)

    # Load pretrained weights.
    encoder_weights = torch.load(config["encoder_pretrained_path"], map_location=device)['encoder']
    encoder.load_state_dict(encoder_weights)

    # Build and load the decoder.
    decoder = Decoder(
        num_patches=vit.pos_embedding.shape[1] * len(channels),
        pixel_values_per_patch=vit.sim_to_patch_embedding[0][1].weight.shape[-1],
        decoder_dim=config['decoder_dim'],
        channels=len(channels)
    )
    decoder.to(device)
    decoder_weights = torch.load(config["decoder_pretrained_path"], map_location=device)['decoder']
    decoder.load_state_dict(decoder_weights)

    # Evaluate the model.
    mse_array = evaluate(encoder, decoder, eval_loader, channels, device)
    print("Evaluation MSE array:")
    print(mse_array)

    # Save the MSE array.
    os.makedirs(config["save_path"], exist_ok=True)
    save_file = os.path.join(config["save_path"], "eval_rmse.npy")
    np.save(save_file, mse_array)
    print(np.mean(mse_array))

if __name__ == '__main__':
    main()
