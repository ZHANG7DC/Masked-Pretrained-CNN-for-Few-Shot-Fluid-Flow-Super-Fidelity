import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from unet import UNet
from dataset import BurgersDataset
import argparse
# --- Load Config ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
args = parser.parse_args()

# --- Load Config ---
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# --- Device ---
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# --- Dataset & Dataloader ---
dataset = BurgersDataset(config["L_path"], config["H_path"], config['training_instance_file'])
train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
data = next(iter(train_loader))
# --- Model ---
model = UNet(config["in_chans"], config["in_chans"])
model = model.to(device)

# --- Load Pretrained Weights (if any) ---
if config.get("pretrained_path"):
    state_dict = torch.load(config["pretrained_path"], map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded pretrained weights from {config['pretrained_path']}")
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    
# --- Loss and Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=config["learning_rate"])

# --- Training Loop ---
pbar = tqdm(range(config["epochs"]), dynamic_ncols=True)
SAVE_DIR = config["save_path"]
os.makedirs(SAVE_DIR, exist_ok=True)
inputs = data["LF"].to(device).float()
targets = data["HF"].to(device).float()
for epoch in pbar:
    model.train()
    running_loss = 0.0

    optimizer.zero_grad()
    outputs = model(inputs)[:,config["in_chans"]//2]
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() ** 0.5

    avg_loss = running_loss / (epoch + 1)

    # Save model
    if epoch % config["save_interval"] == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{epoch}.pth"))

    # Logging
    pbar.set_description(f"avg loss: {avg_loss:.6f}")
