import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import cv2
import numpy as np
import lpips
from DISTS_pytorch import DISTS

from bettermodel import ModularSRNet, PCASRDataset

def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        print(f"  [ERROR] Checkpoint not found at: {path}")
        return False
        
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle both direct state_dict saves and dictionary saves
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    clean_weights = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_weights, strict=False)
    model.eval()
    return True

# ==================== METRICS ====================
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def calculate_lpips(img1, img2, lpips_model, device):
    """Calculate LPIPS (lower is better)"""
    with torch.no_grad():
        # Convert numpy arrays to tensors [0, 1]
        img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        img1_tensor = img1_tensor.to(device) #done automatically by eval loop
        img2_tensor = img2_tensor.to(device)

        # LPIPS expects inputs in range [-1, 1]
        img1_norm = img1_tensor * 2.0 - 1.0
        img2_norm = img2_tensor * 2.0 - 1.0

        lpips_value = lpips_model(img1_norm, img2_norm)
        return lpips_value.mean().item()

def calculate_dists(img1, img2, dists_model, device):
    """Calculate DISTS (lower is better)"""
    with torch.no_grad():
        # Convert numpy arrays to tensors [0, 1]
        img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # DISTS expects inputs in range [0, 1]
        dists_value = dists_model(img1_tensor, img2_tensor)
        return dists_value.mean().item()

def calc_diff_metrics():
    ROOT_DATA_DIR = r"C:\Users\jumpj\Documents\summer2026"
    OUTPUT_DIR = r"C:\Users\jumpj\Documents\summer2026\setupcheck"
    BATCH_SIZE = 16
    ALL_PCA_LEVELS = [15, 20, 30, 40, 50]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Test Set Evaluation on {device}...")
    
    #for later metrics
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    dists_model = DISTS().to(device).eval()

    METRICS_CONFIG = {
        "PSNR": {"func": calculate_psnr, "fmt": "{:.2f} dB"},
        "SSIM": {"func": calculate_ssim, "fmt": "{:.4f}"},
        "LPIPS": {"func": lambda img1, img2: calculate_lpips(img1, img2, lpips_model, device), "fmt": "{:.4f}"},
        "DISTS": {"func": lambda img1, img2: calculate_dists(img1, img2, dists_model, device), "fmt": "{:.4f}"}
    }

    #dataloaders
    test_loaders = {}
    for pca in ALL_PCA_LEVELS:
        try:
            dataset = PCASRDataset(ROOT_DATA_DIR, split="test", pca_levels=[pca], augment=False)
            test_loaders[pca] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
            print(f"Loaded Test Set for PCA {pca}: {len(dataset)} images")
        except ValueError as e:
            print(f"Skipping PCA {pca}: {e}")

    if not test_loaders:
        print("ERROR: No test datasets loaded. Please check your data directories.")
        return

    #config
    base_dir = r"C:\Users\jumpj\Documents\summer2026"
    configs = [
        {
            "name": "Flat ResNet (DS 0)", 
            "down_steps": 0, 
            "checkpoint_path": os.path.join(base_dir, "setupcheck", "downsample_tests", "downsample_0", "model_best.pth")
            #"checkpoint_path": os.path.join(base_dir, "checkpoints", "model_best.pth")
        },
        {
            "name": "Encoder-Decoder (DS 1)", 
            "down_steps": 1, 
            "checkpoint_path": os.path.join(base_dir, "setupcheck", "downsample_tests", "downsample_1", "model_best.pth")
            #"checkpoint_path": os.path.join(base_dir, "encoderdecoder", "model_best.pth") 
        },
        {
            "name": "Heavy Downsample (DS 2)", 
            "down_steps": 2, 
            "checkpoint_path": os.path.join(base_dir, "setupcheck", "downsample_tests", "downsample_2", "model_best.pth")
        },
        {
            "name": "Extreme Downsample (DS 3)", 
            "down_steps": 3, 
            "checkpoint_path": os.path.join(base_dir, "setupcheck", "downsample_tests", "downsample_3", "model_best.pth")
        }
    ]

    #output start
    output_filepath = os.path.join(OUTPUT_DIR, "newmetrics.txt")
    
    header = "\n" + "="*110 + "\n"
    header += f"{'Model':<25} | " + " | ".join([f"PCA {p}".ljust(12) for p in ALL_PCA_LEVELS]) + " | Overall Avg\n"
    header += "-"*110 + "\n"
    
    print(header, end="")
    with open(output_filepath, "w") as f:
        f.write(header)

    #helper function to print and save rows
    def log_row(text):
        print(text, end="")
        with open(output_filepath, "a") as f:
            f.write(text)

    #baselines
    log_row(f"{'Baseline (No Model)':<25} |\n")
    
    baseline_results = {m: {} for m in METRICS_CONFIG}
    baseline_overall = {m: 0.0 for m in METRICS_CONFIG}
    baseline_total_batches = 0

    with torch.no_grad():
        for pca, loader in test_loaders.items():
            pca_accumulators = {m: 0.0 for m in METRICS_CONFIG}
            
            for lr_imgs, hr_imgs in loader:
                lr_imgs, hr_imgs = lr_imgs.to(device, non_blocking=True), hr_imgs.to(device, non_blocking=True)
                
                #for functs (ex to numpy or astype)
                lr_np = lr_imgs.detach().cpu().numpy()
                hr_np = hr_imgs.detach().cpu().numpy()
                
                batch_size = lr_np.shape[0]
                batch_metrics = {m: 0.0 for m in METRICS_CONFIG}
                
                for i in range(batch_size):
                    #as is numpy need to change format
                    lr_single = (np.transpose(lr_np[i], (1, 2, 0)) * 255.0).clip(0, 255) #need make from 0-1 to 0-255 to match functions 
                    hr_single = (np.transpose(hr_np[i], (1, 2, 0)) * 255.0).clip(0, 255)
                    
                    for m_name, m_info in METRICS_CONFIG.items():
                        batch_metrics[m_name] += m_info["func"](lr_single, hr_single)
                
                for m_name in METRICS_CONFIG:
                    pca_accumulators[m_name] += (batch_metrics[m_name] / batch_size)
            
            for m_name in METRICS_CONFIG:
                avg_val = pca_accumulators[m_name] / len(loader)
                baseline_results[m_name][pca] = avg_val
                baseline_overall[m_name] += pca_accumulators[m_name]
                
            baseline_total_batches += len(loader)

    for m_name, m_info in METRICS_CONFIG.items():
        overall_avg = baseline_overall[m_name] / baseline_total_batches if baseline_total_batches > 0 else 0.0
        row = f"  - {m_name:<21} | "
        
        row_vals = [m_info["fmt"].format(baseline_results[m_name][p]).ljust(12) if p in baseline_results[m_name] else "N/A".ljust(12) for p in ALL_PCA_LEVELS]
        row += " | ".join(row_vals)
        
        row += f" | {m_info['fmt'].format(overall_avg)}\n"
        log_row(row)

    log_row("-" * 110 + "\n")

    #eval
    for cfg in configs:
        model = ModularSRNet(ch=128, num_blocks=16, downsample_steps=cfg['down_steps']).to(device)
        
        if not load_checkpoint(model, cfg['checkpoint_path'], device):
            continue

        log_row(f"{cfg['name']:<25} |\n")
        
        model_results = {m: {} for m in METRICS_CONFIG}
        model_overall = {m: 0.0 for m in METRICS_CONFIG}
        total_batches = 0

        with torch.no_grad():
            for pca, loader in test_loaders.items():
                pca_accumulators = {m: 0.0 for m in METRICS_CONFIG}
                
                for lr_imgs, hr_imgs in loader:
                    lr_imgs, hr_imgs = lr_imgs.to(device, non_blocking=True), hr_imgs.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        preds = model(lr_imgs)
                    
                    preds_np = preds.detach().cpu().numpy()
                    hr_np = hr_imgs.detach().cpu().numpy()
                    
                    batch_size = preds_np.shape[0]
                    batch_metrics = {m: 0.0 for m in METRICS_CONFIG}
                    
                    for i in range(batch_size):
                        pred_single = (np.transpose(preds_np[i], (1, 2, 0)) * 255.0).clip(0, 255)
                        hr_single = (np.transpose(hr_np[i], (1, 2, 0)) * 255.0).clip(0, 255)                        
                        
                        for m_name, m_info in METRICS_CONFIG.items():
                            batch_metrics[m_name] += m_info["func"](pred_single, hr_single)
                    
                    for m_name in METRICS_CONFIG:
                        pca_accumulators[m_name] += (batch_metrics[m_name] / batch_size)

                for m_name in METRICS_CONFIG:
                    avg_val = pca_accumulators[m_name] / len(loader)
                    model_results[m_name][pca] = avg_val
                    model_overall[m_name] += pca_accumulators[m_name]
                    
                total_batches += len(loader)

        for m_name, m_info in METRICS_CONFIG.items():
            overall_avg = model_overall[m_name] / total_batches if total_batches > 0 else 0.0
            row = f"  - {m_name:<21} | "
            
            row_vals = [m_info["fmt"].format(model_results[m_name][p]).ljust(12) if p in model_results[m_name] else "N/A".ljust(12) for p in ALL_PCA_LEVELS]
            row += " | ".join(row_vals)
            
            row += f" | {m_info['fmt'].format(overall_avg)}\n"
            log_row(row)

    log_row("-" * 110 + "\n")
            
    del model #for storage
    torch.cuda.empty_cache()

    log_row("="*110 + "\n")
    print(f"Test set results saved to: {output_filepath}")

if __name__ == "__main__":
    calc_diff_metrics()