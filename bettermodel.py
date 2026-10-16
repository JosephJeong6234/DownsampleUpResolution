import os
import random
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.transforms as T

seed = 42 #consistency
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
random.seed(seed)
np.random.seed(seed)

#model parts
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        nn.init.normal_(self.conv2.weight, std=0.01)
        nn.init.zeros_(self.conv2.bias)
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class ModularSRNet(nn.Module):
    def __init__(self, ch=128, num_blocks=32, downsample_steps=1): #defult parameters outdated, now should be ch=128, numblock=16, downsample=0
        super().__init__()
        encoder_layers = []
        in_c = 3
        for _ in range(downsample_steps): #can technically remove this
            encoder_layers.extend([
                nn.Conv2d(in_c, ch, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ])
            in_c = ch
        self.encoder = nn.Sequential(*encoder_layers)

        first_ch = in_c if downsample_steps > 0 else 3
        self.feat_norm = ConvBlock(first_ch, ch)

        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(num_blocks)])
        self.body_proj = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        
        decoder_layers = []
        for _ in range(downsample_steps):
            decoder_layers.extend([
                nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2, bias=True),
                nn.ReLU(inplace=True)
            ])
        self.decoder = nn.Sequential(*decoder_layers)
        self.tail = nn.Conv2d(ch, 3, kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.tail.weight, std=0.001)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        e = self.encoder(x)
        f = self.feat_norm(e)
        b = self.body_proj(self.body(f)) + f
        d = self.decoder(b)
        return torch.clamp(x + self.tail(d), 0.0, 1.0)

class PCASRDataset(Dataset): #pair lr to hr while using pca15(or techincally 20) hr for all as all same ground truth
    def __init__(self, root_dir, split="train", pca_levels=[15, 20, 30, 40, 50], 
                 lr_folder="LR2", hr_reference_pca="PCA15", hr_folder="HR2", augment=False):
        
        self.lr_paths, self.hr_paths = [], []
        self.augment = augment
        self.size = 256
        
        #ram -> go faster 
        self.use_cache = True  
        self.lr_cache = {}
        self.hr_cache = {}

        if isinstance(pca_levels, (int, str)): pca_levels = [pca_levels]
        hr_dir = os.path.join(root_dir, hr_reference_pca, split, hr_folder)

        for pca in pca_levels:
            pca_name = f"PCA{pca}" if not str(pca).startswith("PCA") else str(pca)
            base_path = os.path.join(root_dir, pca_name, split)
            lr_dir = os.path.join(base_path, lr_folder)
            
            if not os.path.exists(lr_dir) or not os.path.exists(hr_dir): continue
                
            lr_files = [f for f in os.listdir(lr_dir) if not f.startswith('.')]
            for lr_fname in sorted(lr_files):
                lr_path = os.path.join(lr_dir, lr_fname)
                base_name = os.path.splitext(lr_fname)[0]
                
                hr_path_jpg = os.path.join(hr_dir, f"{base_name}.jpg")
                hr_path_jpeg = os.path.join(hr_dir, f"{base_name}.jpeg")
                hr_path_png = os.path.join(hr_dir, f"{base_name}.png")
                
                if os.path.exists(hr_path_jpg): hr_path = hr_path_jpg
                elif os.path.exists(hr_path_jpeg): hr_path = hr_path_jpeg
                elif os.path.exists(hr_path_png): hr_path = hr_path_png
                else: continue 
                
                # if pca in [15, 20]: #old oversampling code when doing encoder decoder
                #     self.lr_paths.extend([lr_path] * 3)
                #     self.hr_paths.extend([hr_path] * 3)
                # else:
                #     self.lr_paths.append(lr_path)
                #     self.hr_paths.append(hr_path)
                self.lr_paths.append(lr_path)
                self.hr_paths.append(hr_path)
                    
    def __len__(self):
        return len(self.lr_paths)

    def _load_and_resize(self, path):
        img = Image.open(path).convert('RGB')
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        try:
            if self.use_cache and idx in self.lr_cache:
                lr_t = self.lr_cache[idx]
                hr_t = self.hr_cache[idx]
            else:
                #only do else once as to actually put everything into the cache/ram 
                lr = self._load_and_resize(self.lr_paths[idx])
                hr = self._load_and_resize(self.hr_paths[idx])
                
                #convert to raw byte tensors [C, H, W] to save memory
                lr_t = torch.from_numpy(np.array(lr, dtype=np.uint8)).permute(2, 0, 1)
                hr_t = torch.from_numpy(np.array(hr, dtype=np.uint8)).permute(2, 0, 1)
                
                if self.use_cache:
                    self.lr_cache[idx] = lr_t
                    self.hr_cache[idx] = hr_t
        except Exception as e:
            new_idx = random.randint(0, len(self.lr_paths) - 1)
            return self.__getitem__(new_idx)

        if self.augment:
            if random.random() > 0.5: lr_t = TF.hflip(lr_t); hr_t = TF.hflip(hr_t)
            if random.random() > 0.5: lr_t = TF.vflip(lr_t); hr_t = TF.vflip(hr_t)
            if random.random() > 0.3:
                angle = random.choice([90, 180, 270])
                lr_t = TF.rotate(lr_t, angle); hr_t = TF.rotate(hr_t, angle)

        #faser if in float
        lr_tensor = lr_t.float() / 255.0
        hr_tensor = hr_t.float() / 255.0
        
        return lr_tensor, hr_tensor

#losses
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #load pretrained VGG19 and grab features
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_submodel = nn.Sequential(*list(vgg.children())[:36]).eval() #not all tho for space
        
        #freeze weights as expesive
        for param in self.vgg_submodel.parameters():
            param.requires_grad = False
            
        #ImageNet requires normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        #for features -> l1 loss
        x_vgg = self.vgg_submodel(self.normalize(x))
        y_vgg = self.vgg_submodel(self.normalize(y))
        return nn.functional.l1_loss(x_vgg, y_vgg)

class EMA: #old description from student
    """Tracks a smooth, Exponential Moving Average of network weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Swaps the EMA weights into the model for validation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restores the actual training weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def calc_psnr(pred, target):
    with torch.no_grad():
        pred, target = pred.clamp(0, 1), target.clamp(0, 1)
        mse = torch.mean((pred - target) ** 2)
        if mse == 0: return 100.0
        return 10.0 * math.log10(1.0 / mse.item())

class FFTLoss(nn.Module):
    """Calculates L1 loss in the Frequency Domain to restore sharp textures and kill grid artifacts."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        return nn.functional.l1_loss(pred_amp, target_amp)

def train_model():
    ROOT_DATA_DIR = "/Users/jumpj/Documents/summer2026"
    CHECKPOINT_DIR = "/Users/jumpj/Documents/summer2026/checkpoints_redo"
    EPOCHS = 100
    BATCH_SIZE = 16 
    NUM_WORKERS = 0 #as on windows now now linux
    LR = 0.0001
    ALL_PCA_LEVELS = [15, 20, 30, 40, 50]
    
    L1_WEIGHT = 2.0 #from design of experiment grid search
    VGG_WEIGHT = 0.3
    FFT_WEIGHT = 0.01

    patience = 100
    epochs_no_improve = 0
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    #dataloader
    train_dataset = PCASRDataset(ROOT_DATA_DIR, split="train", augment=True)
    valid_dataset = PCASRDataset(ROOT_DATA_DIR, split="valid", augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

    pca_loaders = {}
    for pca in ALL_PCA_LEVELS:
        ds = PCASRDataset(root_dir=ROOT_DATA_DIR, split="valid", pca_levels=[pca], augment=False)
        pca_loaders[pca] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    torch.backends.cudnn.benchmark = True

    #loss + model
    model = ModularSRNet(ch=128, num_blocks=16, downsample_steps=0).to(device)
    #model = torch.compile(model) #won't work on windows
    ema = EMA(model, decay=0.999)
    
    criterion_l1 = nn.L1Loss().to(device)
    criterion_vgg = PerceptualLoss().to(device)
    criterion_fft = FFTLoss().to(device) 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    best_psnr = 0.0

    #baseline calc 
    with open(os.path.join(CHECKPOINT_DIR, "final_training_log.txt"), "w") as log_file:
        log_file.write(f"Weights -> L1: {L1_WEIGHT}, VGG: {VGG_WEIGHT}, FFT: {FFT_WEIGHT}\n\n")

        print("Calculating Initial Baseline...")
        base_val_psnr = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                base_val_psnr += calc_psnr(lr_imgs.to(device), hr_imgs.to(device))
        
        avg_base_psnr = base_val_psnr / len(valid_loader)
        log_file.write(f"--- BASELINE METRICS ---\n")
        log_file.write(f"Global Degraded Baseline PSNR: {avg_base_psnr:.4f} dB\n\n")
        log_file.write(f"--- EPOCH LOGS ---\n")
        print(f"Baseline Calculated: {avg_base_psnr:.2f} dB. Starting Training...\n")

        #the actual training loop 
        for epoch in range(EPOCHS):
            model.train()
            train_l1_loss, train_vgg_loss, train_fft_loss = 0.0, 0.0, 0.0
            #tqdm is for progress bar (sanity check)
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for lr_imgs, hr_imgs in loop:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    preds = model(lr_imgs)
                    loss_l1 = criterion_l1(preds, hr_imgs)
                    loss_vgg = criterion_vgg(preds, hr_imgs)
                    loss_fft = criterion_fft(preds, hr_imgs)
                    
                    total_loss = (L1_WEIGHT * loss_l1) + (VGG_WEIGHT * loss_vgg) + (FFT_WEIGHT * loss_fft)
                    
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                ema.update()
                
                train_l1_loss += loss_l1.item()
                train_vgg_loss += loss_vgg.item()
                train_fft_loss += loss_fft.item()
                loop.set_postfix(L1=f"{loss_l1.item():.4f}", VGG=f"{loss_vgg.item():.4f}", FFT=f"{loss_fft.item():.4f}")

            avg_l1 = train_l1_loss / len(train_loader)
            avg_vgg = train_vgg_loss / len(train_loader)
            avg_fft = train_fft_loss / len(train_loader)

            #vall loop -> ema weights 
            ema.apply_shadow()
            model.eval()
            val_psnr = 0.0

            with torch.no_grad():
                for lr_imgs, hr_imgs in tqdm(valid_loader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    with torch.amp.autocast('cuda'):
                        preds = model(lr_imgs)
                    val_psnr += calc_psnr(preds, hr_imgs)
                    
            avg_val_psnr = val_psnr / len(valid_loader)
            ema.restore() 

            #pca check every 5 as too expensive for every epoch
            if epoch % 5 == 0 or epoch == EPOCHS: 
                log_file.write(f"\n--- Epoch {epoch} PCA Breakdown ---\n")
                
                with torch.no_grad():
                    for pca, loader in pca_loaders.items():
                        pca_psnr = 0.0
                        for lr_imgs, hr_imgs in loader:
                            lr_imgs, hr_imgs = lr_imgs.to(device, non_blocking=True), hr_imgs.to(device, non_blocking=True)
                            with torch.amp.autocast('cuda'):
                                preds = model(lr_imgs)
                            pca_psnr += calc_psnr(preds, hr_imgs)
                        
                        avg_pca_psnr = pca_psnr / len(loader) if len(loader) > 0 else 0
                        print(f"     PCA {pca}: {avg_pca_psnr:.2f} dB")
                        
                        log_file.write(f"PCA {pca}: {avg_pca_psnr:.4f} dB | ")
                        
                log_file.write("\n\n")

            scheduler.step()
            
            epoch_log = f"Epoch {epoch+1:03d}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} | L1: {avg_l1:.4f} | VGG: {avg_vgg:.4f} | FFT: {avg_fft:.4f} | Val PSNR: {avg_val_psnr:.2f} dB"
            print(epoch_log)
            log_file.write(epoch_log + "\n")

            is_best = avg_val_psnr > best_psnr
            if is_best: 
                best_psnr = avg_val_psnr
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_psnr': best_psnr
                }
                torch.save(state, os.path.join(CHECKPOINT_DIR, 'model_best.pth'))
                epochs_no_improve = 0  # Reset the clock!
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Best PSNR was {best_psnr:.2f}")
                break
            
            # is_best = avg_val_psnr > best_psnr #this before patience
            # state = {
            #     'epoch': epoch + 1,
            #     'state_dict': ema.shadow, 
            #     'best_psnr': best_psnr,
            # }
            # torch.save(state, os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pth'))
            # if is_best:
            #     torch.save(state, os.path.join(CHECKPOINT_DIR, 'model_best.pth'))

        #final breakdown by pca level
        print("\nLoading Best EMA Model for Final PCA Breakdown...")
        log_file.write("\n--- FINAL BREAKDOWN BY PCA LEVEL ---\n")
        
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'model_best.pth'), map_location=device)
        model.load_state_dict(checkpoint['state_dict']) # Load the EMA weights specifically
        
        total_breakdown_psnr = 0.0
        model.eval()
        with torch.no_grad():
            for pca, loader in pca_loaders.items():
                pca_val_psnr = 0.0
                for lr_imgs, hr_imgs in loader:
                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    with torch.amp.autocast('cuda'):
                        preds = model(lr_imgs)
                    pca_val_psnr += calc_psnr(preds, hr_imgs)
                
                avg_pca_psnr = pca_val_psnr / len(loader) if len(loader) > 0 else 0
                total_breakdown_psnr += avg_pca_psnr
                
                print(f"PCA {pca} PSNR: {avg_pca_psnr:.2f} dB")
                log_file.write(f"PCA {pca}: {avg_pca_psnr:.4f} dB\n")

        log_file.write(f"\nFinal Global Average: {total_breakdown_psnr / len(pca_loaders):.4f} dB\n")
        print(f"\nTraining Complete! Logs saved to {CHECKPOINT_DIR}/final_training_log.txt")

if __name__ == "__main__":
    train_model()