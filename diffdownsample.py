import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==========================================
# 1. IMPORTS FROM MAIN SCRIPT
# ==========================================
from bettermodel import (
    PCASRDataset, 
    ModularSRNet,
    PerceptualLoss, 
    FFTLoss, 
    calc_psnr
)

# ==========================================
# 2. RUNNER
# ==========================================
def train_downsample_config(config_name, ch, num_blocks, downsample_steps, train_loader, val_loaders, device, output_dir):
    print(f"\n[{config_name}] Initialization: Channels={ch}, Blocks={num_blocks}, Downsamples={downsample_steps}")
    
    # Create the specific subfolder for this run
    run_dir = os.path.join(output_dir, f"downsample_{downsample_steps}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training_log.txt")
    
    model = ModularSRNet(ch=ch, num_blocks=num_blocks, downsample_steps=downsample_steps).to(device)
    
    criterion_l1 = nn.L1Loss().to(device)
    criterion_vgg = PerceptualLoss().to(device)
    criterion_fft = FFTLoss().to(device) 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, fused=True)
    # Using 100 epochs
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    best_psnr = 0.0
    start_time = time.time()

    with open(log_file, "w") as f:
        f.write(f"Starting {config_name}\n")
        f.write(f"Channels: {ch}, Blocks: {num_blocks}, Downsamples: {downsample_steps}\n")
        f.write("-" * 40 + "\n")

    for epoch in range(epochs):
        model.train()
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device, non_blocking=True), hr_imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                preds = model(lr_imgs)
                total_loss = (2.0 * criterion_l1(preds, hr_imgs)) + (0.3 * criterion_vgg(preds, hr_imgs)) + (0.01 * criterion_fft(preds, hr_imgs))
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        val_psnr_dict = {}
        overall_psnr_sum = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for pca, v_loader in val_loaders.items():
                pca_psnr = 0.0
                for lr_imgs, hr_imgs in v_loader:
                    with torch.amp.autocast('cuda'):
                        preds = model(lr_imgs.to(device, non_blocking=True))
                    pca_psnr += calc_psnr(preds, hr_imgs.to(device, non_blocking=True))
                
                # Calculate average PSNR for this specific PCA level
                avg_pca_psnr = pca_psnr / len(v_loader) if len(v_loader) > 0 else 0
                val_psnr_dict[pca] = avg_pca_psnr
                
                # Accumulate for the overall average
                overall_psnr_sum += pca_psnr
                total_batches += len(v_loader)
                
        # Calculate the true overall average across all batches
        avg_val_psnr = overall_psnr_sum / total_batches if total_batches > 0 else 0
        
        # Save best model based on overall PSNR
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))
            
        scheduler.step()
        
        log_str = f"Epoch {epoch+1:03d}/{epochs} | Val PSNR: {avg_val_psnr:.2f} dB"
        print(f"  {log_str}")
        
        # Append epoch result to log
        with open(log_file, "a") as f:
            f.write(log_str + "\n")
            
        # Every 5 epochs, print and log the detailed PCA breakdown
        if (epoch + 1) % 5 == 0:
            breakdown_str = "    --> PCA Breakdown: " + " | ".join([f"PCA {p}: {val_psnr_dict[p]:.2f} dB" for p in val_psnr_dict])
            print(breakdown_str)
            with open(log_file, "a") as f:
                f.write(breakdown_str + "\n")

    total_time = (time.time() - start_time) / 60.0
    final_summary = f"\n[{config_name}] Complete. Best PSNR: {best_psnr:.2f} dB | Total Time: {total_time:.1f} min"
    print(final_summary)
    
    with open(log_file, "a") as f:
        f.write("-" * 40 + "\n")
        f.write(final_summary + "\n")
        
    return best_psnr, total_time

def main():
    ROOT_DATA_DIR = r"C:\Users\jumpj\Documents\summer2026"
    OUTPUT_DIR = r"C:\Users\jumpj\Documents\summer2026\setupcheck\downsample_tests"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Initializing Datasets...")
    train_dataset = PCASRDataset(ROOT_DATA_DIR, split="train", augment=True)
    
    # Configurations for Downsampling
    configs = [
        {"name": "Flat ResNet", "ch": 128, "blocks": 16, "down": 0, "bs": 16},
        {"name": "Downsample x1", "ch": 128, "blocks": 16, "down": 1, "bs": 16},
        #{"name": "Downsample x2", "ch": 128, "blocks": 16, "down": 2, "bs": 16},
        #{"name": "Downsample x3", "ch": 128, "blocks": 16, "down": 3, "bs": 16}
    ]

    for cfg in configs:
        # Dynamic dataloader just in case
        train_loader = DataLoader(train_dataset, batch_size=cfg["bs"], shuffle=True, num_workers=0, pin_memory=True)
        
        # Create a dictionary of validation loaders separated by PCA
        val_loaders = {}
        for pca in [15, 20, 30, 40, 50]:
            v_ds = PCASRDataset(ROOT_DATA_DIR, split="valid", pca_levels=[pca], augment=False)
            val_loaders[pca] = DataLoader(v_ds, batch_size=cfg["bs"], shuffle=False, num_workers=0, pin_memory=True)
        
        train_downsample_config(
            cfg["name"], 
            cfg["ch"], 
            cfg["blocks"], 
            cfg["down"], 
            train_loader, 
            val_loaders, 
            device, 
            OUTPUT_DIR
        )

if __name__ == "__main__":
    main()