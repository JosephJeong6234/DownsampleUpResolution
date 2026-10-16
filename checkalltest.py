import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Import dependencies directly from your main file
from bettermodel import ModularSRNet, PCASRDataset

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

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

def evaluate_test_set():
    ROOT_DATA_DIR = r"C:\Users\jumpj\Documents\summer2026"
    OUTPUT_DIR = r"C:\Users\jumpj\Documents\summer2026\setupcheck"
    BATCH_SIZE = 16
    ALL_PCA_LEVELS = [15, 20, 30, 40, 50]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Test Set Evaluation on {device}...")

    # 1. Initialize Test DataLoaders per PCA level
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

    # 2. Define the Model Configurations to Test
    # IMPORTANT: Verify these checkpoint paths match exactly where your models are saved.
    base_dir = r"C:/Users/jumpj/Documents/summer2026/"
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

    # 3. Output Formatting Setup
    output_filepath = os.path.join(OUTPUT_DIR, "new_test_set_results.txt")
    header = "\n" + "="*80 + "\n"
    header += f"{'Model Configuration':<25} | " + " | ".join([f"PCA {p:<2}" for p in ALL_PCA_LEVELS]) + " | Overall Avg\n"
    header += "-"*80 + "\n"
    
    print(header, end="")
    with open(output_filepath, "w") as f:
        f.write(header)

    baseline_results = {}
    baseline_psnr_sum = 0.0
    baseline_total_batches = 0

    with torch.no_grad():
        for pca, loader in test_loaders.items():
            pca_psnr = 0.0
            for lr_imgs, hr_imgs in loader:
                # 1. Detach from GPU and convert to NumPy
                lr_np = lr_imgs.detach().cpu().numpy()
                hr_np = hr_imgs.detach().cpu().numpy()
                
                batch_size = lr_np.shape[0]
                batch_psnr = 0.0
                
                # 2. Iterate through the batch one image at a time
                for i in range(batch_size):
                    # Transpose from (C, H, W) to (H, W, C) and scale to [0, 255]
                    lr_single = (np.transpose(lr_np[i], (1, 2, 0)) * 255.0).clip(0, 255)
                    hr_single = (np.transpose(hr_np[i], (1, 2, 0)) * 255.0).clip(0, 255)
                    
                    # Baseline is just the PSNR between the degraded image and ground truth
                    batch_psnr += calculate_psnr(lr_single, hr_single)
                
                # Accumulate the average PSNR of this batch
                pca_psnr += (batch_psnr / batch_size)
            
            avg_pca_psnr = pca_psnr / len(loader)
            baseline_results[pca] = avg_pca_psnr
            
            baseline_psnr_sum += pca_psnr
            baseline_total_batches += len(loader)

    baseline_overall_avg = baseline_psnr_sum / baseline_total_batches if baseline_total_batches > 0 else 0.0
    
    baseline_row = f"{'Baseline (No Model)':<25} | "
    baseline_row += " | ".join([f"{baseline_results[p]:.2f} dB" if p in baseline_results else "N/A" for p in ALL_PCA_LEVELS])
    baseline_row += f" | {baseline_overall_avg:.2f} dB\n"
    
    print(baseline_row, end="")
    with open(output_filepath, "a") as f:
        f.write(baseline_row)

    # 4. Evaluation Loop
    for cfg in configs:
        model = ModularSRNet(ch=128, num_blocks=16, downsample_steps=cfg['down_steps']).to(device)
        
        if not load_checkpoint(model, cfg['checkpoint_path'], device):
            continue

        results = {}
        overall_psnr_sum = 0.0
        total_batches = 0

        with torch.no_grad():
            for pca, loader in test_loaders.items():
                pca_psnr = 0.0
                for lr_imgs, hr_imgs in loader:
                    lr_imgs = lr_imgs.to(device, non_blocking=True)
                    hr_imgs = hr_imgs.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        preds = model(lr_imgs)

                    # 1. Detach from GPU and convert to NumPy
                    preds_np = preds.detach().cpu().numpy()
                    hr_np = hr_imgs.detach().cpu().numpy()
                    
                    batch_size = preds_np.shape[0]
                    batch_psnr = 0.0
                    
                    # 2. Iterate through the batch one image at a time
                    for i in range(batch_size):
                        # Transpose from (C, H, W) to (H, W, C) and scale to [0, 255]
                        pred_single = (np.transpose(preds_np[i], (1, 2, 0)) * 255.0).clip(0, 255)
                        hr_single = (np.transpose(hr_np[i], (1, 2, 0)) * 255.0).clip(0, 255)
                        
                        # Calculate PSNR for the network prediction
                        batch_psnr += calculate_psnr(pred_single, hr_single)
                    
                    # Accumulate the average PSNR of this batch
                    pca_psnr += (batch_psnr / batch_size)
                
                avg_pca_psnr = pca_psnr / len(loader)
                results[pca] = avg_pca_psnr
                
                overall_psnr_sum += pca_psnr
                total_batches += len(loader)

        overall_avg = overall_psnr_sum / total_batches if total_batches > 0 else 0.0
        
        # Format and save row
        row = f"{cfg['name']:<25} | "
        row += " | ".join([f"{results[p]:.2f} dB" if p in results else "N/A" for p in ALL_PCA_LEVELS])
        row += f" | {overall_avg:.2f} dB\n"
        
        print(row, end="")
        with open(output_filepath, "a") as f:
            f.write(row)
            
        # Free VRAM
        del model
        torch.cuda.empty_cache()

    footer = "="*80 + "\n"
    print(footer)
    with open(output_filepath, "a") as f:
        f.write(footer)
    print(f"Test set results saved to: {output_filepath}")

if __name__ == "__main__":
    evaluate_test_set()