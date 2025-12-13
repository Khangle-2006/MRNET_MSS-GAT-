import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn as nn
import time
import os
import torch.nn.functional as F

# --- 1. CẤU HÌNH ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from src_2.dataset import MRNetDataset 
from src_2.model import MultiViewSRRNet_V9

ROOT_DIR = './MRNet-v1.0' 
CHECKPOINT_PATH = './best_model_v9_multi_scale.pth' # Tên mới cho chiến thuật mới
BACKBONE_WEIGHTS = './resnet18_modan_mulsupcon_1ch.pth' 

BATCH_SIZE = 14            
ACCUMULATION_STEPS = 1    
EPOCHS = 400              
MIXUP_ALPHA = 0.4         

# [CẤU HÌNH ĐA TỶ LỆ]
ZOOM_MID = 0.7   # Cứu Meniscus (vừa đủ rộng)
ZOOM_CLOSE = 0.5 # Soi ACL (cận cảnh)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 2. GPU PROCESSOR (MULTI-SCALE LOGIC) ---
class GPUProcessor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # Augmentation vẫn giữ nhẹ nhàng
        self.geo_transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
            )
        )
        self.eraser = transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0)

    def create_multi_scale_view(self, tensor_batch):
        """
        Tạo 3 kênh RGB từ 3 mức zoom khác nhau.
        Input: (B, 1, 32, 256, 256)
        Output: (B, 3, 32, 224, 224)
        """
        B, C, D, H, W = tensor_batch.shape
        
        # --- Channel R: GLOBAL VIEW (Toàn cảnh) ---
        global_view = F.interpolate(tensor_batch, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # --- Channel G: MID VIEW (Zoom 0.75) -> MENISCUS ---
        crop_h_mid, crop_w_mid = int(H * ZOOM_MID), int(W * ZOOM_MID)
        sh_mid, sw_mid = (H - crop_h_mid)//2, (W - crop_w_mid)//2
        mid_crop = tensor_batch[:, :, :, sh_mid:sh_mid+crop_h_mid, sw_mid:sw_mid+crop_w_mid]
        mid_view = F.interpolate(mid_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # --- Channel B: CLOSE VIEW (Zoom 0.55) -> ACL ---
        crop_h_close, crop_w_close = int(H * ZOOM_CLOSE), int(W * ZOOM_CLOSE)
        sh_close, sw_close = (H - crop_h_close)//2, (W - crop_w_close)//2
        close_crop = tensor_batch[:, :, :, sh_close:sh_close+crop_h_close, sw_close:sw_close+crop_w_close]
        close_view = F.interpolate(close_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # Stack lại thành 3 kênh (R, G, B)
        multi_scale_tensor = torch.cat([global_view, mid_view, close_view], dim=1)
        
        return multi_scale_tensor

    def process_batch_list(self, batch_list, is_training=True):
        views_list = [item[0] for item in batch_list]
        labels_list = [item[1] for item in batch_list]
        labels_tensor = torch.stack(labels_list).to(self.device, non_blocking=True)
        
        final_views = {}
        for view_name in ['sagittal', 'coronal', 'axial']:
            batch_tensors = []
            for i in range(len(views_list)):
                raw_t = views_list[i][view_name].to(self.device, non_blocking=True)
                if raw_t.ndim == 3: raw_5d = raw_t.unsqueeze(0).unsqueeze(0)
                elif raw_t.ndim == 4: raw_5d = raw_t.unsqueeze(0)
                else: raw_5d = raw_t

                resized = F.interpolate(raw_5d, size=(32, 256, 256), mode='trilinear', align_corners=False)
                batch_tensors.append(resized.squeeze(0))
            
            batch_tensor_gpu = torch.stack(batch_tensors)
            
            # [GỌI HÀM MULTI-SCALE]
            ms_tensor = self.create_multi_scale_view(batch_tensor_gpu)
            
            B, C, D, H, W = ms_tensor.shape
            flat = ms_tensor.view(B*D, C, H, W) 
            flat = self.normalize(flat)
            
            if is_training:
                flat = self.geo_transforms(flat)
                flat = self.eraser(flat)
            
            final_views[view_name] = flat.reshape(B, C, D, H, W)
            
        return final_views, labels_tensor

# --- 3. WEIGHT LOADING (Giữ nguyên) ---
def load_and_inflate_weights(model, weight_path, device):
    print(f"--> [Smart Weight Loading] {weight_path}")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
        
        key_mapping = {
            'conv1.': 'features_2d.0.', 'bn1.': 'features_2d.1.',
            'layer1.': 'features_2d.4.', 'layer2.': 'features_2d.5.',
            'layer3.': 'features_2d.6.', 'layer4.': 'features_2d.7.'
        }
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for old, new in key_mapping.items():
                if k.startswith(old):
                    new_key = k.replace(old, new)
                    break
            if 'features_2d.0.weight' in new_key:
                curr_shape = model.backbone_sag.features_2d[0].weight.shape
                if v.shape[2] != curr_shape[2]: continue 
                if v.shape[1] == 1 and curr_shape[1] == 3:
                    v = torch.cat([v, v, v], dim=1) / 3.0
            new_state_dict[new_key] = v

        model.backbone_sag.load_state_dict(new_state_dict, strict=False)
        model.backbone_cor.load_state_dict(new_state_dict, strict=False)
        model.backbone_axi.load_state_dict(new_state_dict, strict=False)
        print("--> Loaded successfully!")
    except Exception as e:
        print(f"!!! Error: {e}")
    return model

# --- 4. LOSS FUNCTIONS ---
class SmoothBCEwLogits(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    def forward(self, logits, target):
        loss = self.bce(logits, target * (1 - self.smoothing) + 0.5 * self.smoothing)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# --- 5. SETUP & MAIN ---
def custom_collate(batch): return batch

device = torch.device('cuda')
gpu_processor = GPUProcessor(device).to(device)

print("--> Loading Dataset...")
train_dataset = MRNetDataset(ROOT_DIR, 'train', transform=None, cache_to_ram=True) 
valid_dataset = MRNetDataset(ROOT_DIR, 'valid', transform=None, cache_to_ram=True) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, 
                          pin_memory=True, collate_fn=custom_collate, persistent_workers=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, 
                          pin_memory=True, collate_fn=custom_collate, persistent_workers=True)

model = MultiViewSRRNet_V9(pretrained_path=None, n_clusters=256, n_neighbors=8, 
                           node_features=128, gnn_out_features=128, num_heads=8, dropout=0.5).to(device)

if os.path.exists(BACKBONE_WEIGHTS):
    model = load_and_inflate_weights(model, BACKBONE_WEIGHTS, device)

optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 5e-5, 'weight_decay': 0.01}
])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Focal Loss
criterion = FocalLoss(alpha=0.75, gamma=2.0)
scaler = torch.amp.GradScaler('cuda')

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- 6. TRAINING LOOP ---
best_meniscus_auc = 0.0 
best_target_auc = 0.0
best_epoch = 0

print(f"--> START MULTI-SCALE TRAINING (Global, Mid=0.75, Close=0.55)...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()
    
    for batch_idx, batch_list in enumerate(train_loader):
        views, labels = gpu_processor.process_batch_list(batch_list, is_training=True)

        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        index = torch.randperm(views['sagittal'].size(0)).to(device)
        
        sag_mix = lam * views['sagittal'] + (1 - lam) * views['sagittal'][index]
        cor_mix = lam * views['coronal'] + (1 - lam) * views['coronal'][index]
        axi_mix = lam * views['axial']    + (1 - lam) * views['axial'][index]
        labels_a, labels_b = labels, labels[index]

        with torch.amp.autocast('cuda'):
            logits, aux_logits, view_embs = model(sag_mix, cor_mix, axi_mix)
            loss_main = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss_aux = mixup_criterion(criterion, aux_logits, labels_a, labels_b, lam)
            h_sag, h_cor, h_axi = view_embs
            loss_cons = (F.mse_loss(h_sag, h_cor) + F.mse_loss(h_cor, h_axi) + F.mse_loss(h_sag, h_axi)) / 3.0
            
            loss = loss_main + 0.4 * loss_aux + 0.1 * loss_cons
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()

    scheduler.step()
    current_lr = optimizer.param_groups[1]['lr']
    epoch_duration = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader)
    
    # --- VALIDATION ---
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch_list in enumerate(valid_loader):
            views, labels = gpu_processor.process_batch_list(batch_list, is_training=False)
            with torch.amp.autocast('cuda'):
                logits, _, _ = model(views['sagittal'], views['coronal'], views['axial'])
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    try:
        auc_abn = roc_auc_score(all_labels[:, 0], all_preds[:, 0])
        auc_acl = roc_auc_score(all_labels[:, 1], all_preds[:, 1])
        auc_men = roc_auc_score(all_labels[:, 2], all_preds[:, 2])
        
        target_metric = (0.4 * auc_acl) + (0.6 * auc_men)
        
        print(f"Epoch {epoch+1:03d} | Time: {epoch_duration:.0f}s | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        print(f"   >>> Val AUC:  Abn={auc_abn:.4f} | ACL={auc_acl:.4f} | Men={auc_men:.4f}")
        print(f"   >>> Avg AUC:  {target_metric:.4f}")
        
        if auc_men > best_meniscus_auc:
            best_meniscus_auc = auc_men
            torch.save({'model_state_dict': model.state_dict(), 'best_meniscus': auc_men}, './best_model_v9_multi_scale_meniscus.pth')
            print(f"   [SAVE] NEW BEST MENISCUS! ({auc_men:.4f})")
            
        if target_metric > best_target_auc:
            best_target_auc = target_metric
            best_epoch = epoch + 1
            torch.save({'model_state_dict': model.state_dict(), 'best_target_auc': best_target_auc}, CHECKPOINT_PATH)
            print(f"   [SAVE] NEW BALANCED BEST! (Avg: {target_metric:.4f})")
        else:
            print(f"   (Best so far: {best_target_auc:.4f} at Epoch {best_epoch})")
            
    except Exception as e:
        print(f"Error computing AUC: {e}")
