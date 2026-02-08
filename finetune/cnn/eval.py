import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet import UNet
from dataset import BurgersDataset

def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device,
             save_outputs: bool = False,
             output_dir: str = None) -> np.ndarray:
    """
    Evaluate a UNet model and optionally save predictions.

    Returns:
        np.ndarray: Array of RMSE (sqrt(MSE)) values per batch.
    """
    model.eval()
    mse_losses = []
    os.makedirs(output_dir, exist_ok=True) if save_outputs and output_dir else None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating", dynamic_ncols=True)):
            inputs = batch['LF'].float().to(device)
            targets = batch['HF'].float().to(device)
            names = batch['Name']
            outputs = model(inputs)[:, 1]

            # Compute RMSE for each channel in the batch
            loss = F.mse_loss(outputs[0], targets[0], reduction='mean').item()**0.5
            mse_losses.append(loss)

            if save_outputs and output_dir:
                output_np = outputs.cpu().numpy()
                for j, name in enumerate(names):
                    sample_output = output_np[j]
                    save_name = os.path.splitext(os.path.basename(name))[0] + "_pred.npy"
                    np.save(os.path.join(output_dir, save_name), sample_output)

    return np.array(mse_losses)

def main():
    parser = argparse.ArgumentParser(description="UNet Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("seed", 42))

    save_outputs = config.get("save_predictions", False)
    pred_output_dir = os.path.join(config["save_path"], "predictions") if save_outputs else None
    
    # Dataset & Dataloader
    dataset = BurgersDataset(config["L_path"], config["H_path"])
    eval_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # Load model
    model = UNet(config["in_chans"], config["in_chans"])
    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    model.eval()
    model.to(device)

    # Run evaluation
    mse_array = evaluate(model, eval_loader, device, save_outputs, pred_output_dir)
    print("Average RMSE:", np.mean(mse_array,axis=0))

    # Save results
    os.makedirs(config["save_path"], exist_ok=True)
    save_path = os.path.join(config["save_path"], "eval_rmse.npy")
    np.save(save_path, mse_array)


if __name__ == "__main__":
    main()
