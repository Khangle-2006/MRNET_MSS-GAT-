import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import os
import torch.nn.functional as F
import torch.nn as nn

# Import source
from src.dataset import MRNetDataset
from src.model import MultiViewSRRNet_V9

# --- CẤU HÌNH ---
ROOT_DIR = './MRNet-v1.0'
# [QUAN TRỌNG] Load đúng checkpoint High-Performance
CHECKPOINT_PATH = './best_model_v9_high_perf.pth' 

N_CLUSTERS = 256
N_NEIGHBORS = 8
GNN_OUT_FEATURES = 128

# [ĐÃ CẬP NHẬT THEO FILE TRAIN]
ZOOM_MID = 0.65   # Kênh G: Focus Meniscus
ZOOM_CLOSE = 0.55 # Kênh B: Focus ACL

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 1. GPU PROCESSOR (SYNC WITH TRAIN.PY) ---
class GPUProcessor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def uniform_sample(self, tensor_batch, target_depth=32):
        """
        [NEW] Uniform Sampling: Giữ nguyên logic như lúc train để ảnh không bị mờ
        """
        B, C, D, H, W = tensor_batch.shape
        if D == target_depth: return tensor_batch
        
        # Lấy index đều nhau
        indices = torch.linspace(0, D - 1, target_depth).long().to(self.device)
        return tensor_batch[:, :, indices, :, :]

    def create_multi_scale_view(self, tensor_batch):
        """
        [NEW] Tạo 3 kênh RGB từ 3 mức zoom (Logic Inference - Không Random Shift)
        """
        B, C, D, H, W = tensor_batch.shape
        
        # 1. CHANNEL R: GLOBAL VIEW
        global_view = F.interpolate(tensor_batch, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # 2. CHANNEL G: MID VIEW (Zoom 0.65)
        crop_h_mid, crop_w_mid = int(H * ZOOM_MID), int(W * ZOOM_MID)
        sh_mid, sw_mid = (H - crop_h_mid)//2, (W - crop_w_mid)//2
        # Inference: Luôn lấy chính giữa (Center Crop)
        mid_crop = tensor_batch[:, :, :, sh_mid:sh_mid+crop_h_mid, sw_mid:sw_mid+crop_w_mid]
        mid_view = F.interpolate(mid_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # 3. CHANNEL B: CLOSE VIEW (Zoom 0.55)
        crop_h_close, crop_w_close = int(H * ZOOM_CLOSE), int(W * ZOOM_CLOSE)
        sh_close, sw_close = (H - crop_h_close)//2, (W - crop_w_close)//2
        # Inference: Luôn lấy chính giữa
        close_crop = tensor_batch[:, :, :, sh_close:sh_close+crop_h_close, sw_close:sw_close+crop_w_close]
        close_view = F.interpolate(close_crop, size=(32, 224, 224), mode='trilinear', align_corners=False)
        
        # Stack (R, G, B)
        return torch.cat([global_view, mid_view, close_view], dim=1)

    def process_batch_list(self, batch_list):
        views_list = [item[0] for item in batch_list]
        labels_list = [item[1] for item in batch_list]
        
        labels_tensor = torch.stack(labels_list).to(self.device)
        
        final_views = {}
        for view_name in ['sagittal', 'coronal', 'axial']:
            batch_tensors = []
            
            for i in range(len(views_list)):
                raw_t = views_list[i][view_name].to(self.device)
                
                if raw_t.ndim == 3: raw_5d = raw_t.unsqueeze(0).unsqueeze(0)
                elif raw_t.ndim == 4: raw_5d = raw_t.unsqueeze(0)
                else: raw_5d = raw_t

                # [QUAN TRỌNG] Thay Interpolate bằng Uniform Sample
                sampled = self.uniform_sample(raw_5d, target_depth=32)
                batch_tensors.append(sampled.squeeze(0)) 
            
            batch_tensor_gpu = torch.stack(batch_tensors) # (B, 1, 32, 256, 256)
            
            # [GỌI HÀM MULTI-SCALE]
            ms_tensor = self.create_multi_scale_view(batch_tensor_gpu)
            
            B, C, D, H, W = ms_tensor.shape
            flat = ms_tensor.view(B*D, C, H, W)
            norm = self.normalize(flat)
            final_views[view_name] = norm.reshape(B, C, D, H, W)
            
        return final_views, labels_tensor

def custom_collate(batch): return batch

# --- 2. METRICS & EVAL ---
def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    try: auc = roc_auc_score(y_true, y_probs)
    except: auc = 0.0 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "AUC": auc, "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def evaluate(model, dataloader, processor):
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0.0
    
    print(f"--> Evaluating on {len(dataloader.dataset)} cases...")
    
    with torch.no_grad():
        for batch_list in dataloader:
            start = time.time()
            views, labels = processor.process_batch_list(batch_list)
            
            # Có thể thêm amp.autocast ở đây nếu muốn giống hệt train, nhưng inference fp32 cũng tốt
            with torch.amp.autocast('cuda'): 
                logits, _, _ = model(views['sagittal'], views['coronal'], views['axial'])
                preds = torch.sigmoid(logits)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            total_time += (time.time() - start)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    fps = len(all_labels) / total_time
    print("-" * 65)
    print(f"--> Total Time: {total_time:.4f}s | Speed: {fps:.2f} FPS")
    print("-" * 65)

    tasks = ['Abnormal', 'ACL', 'Meniscus']
    print(f"{'TASK':<10} | {'AUC':<7} | {'ACC':<7} | {'SENS':<7} | {'SPEC':<7} | {'F1':<7}")
    print("-" * 65)
    
    avgs = {"AUC": 0, "Accuracy": 0, "Sensitivity": 0, "Specificity": 0, "F1": 0}
    
    for i, task in enumerate(tasks):
        m = compute_metrics(all_labels[:, i], all_preds[:, i])
        print(f"{task:<10} | {m['AUC']:.4f}  | {m['Accuracy']:.4f}  | {m['Sensitivity']:.4f}  | {m['Specificity']:.4f}  | {m['F1']:.4f}")
        for k in avgs: avgs[k] += m[k]

    print("-" * 65)
    for k in avgs: avgs[k] /= 3.0
    print(f"{'AVERAGE':<10} | {avgs['AUC']:.4f}  | {avgs['Accuracy']:.4f}  | {avgs['Sensitivity']:.4f}  | {avgs['Specificity']:.4f}  | {avgs['F1']:.4f}")
    print("=" * 65)

# --- 4. MAIN ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Inference Device: {device}")

    # Dataset (Raw)
    val_dataset = MRNetDataset(ROOT_DIR, 'valid', transform=None, cache_to_ram=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=custom_collate)

    processor = GPUProcessor(device).to(device)

    print("--> Initializing Model V9 (Multi-Scale RGB)...")
    # Đảm bảo src.model đã được cập nhật (có CBAM nếu dùng)
    model = MultiViewSRRNet_V9(
        pretrained_path=None, 
        n_clusters=N_CLUSTERS, n_neighbors=N_NEIGHBORS,
        node_features=128, gnn_out_features=GNN_OUT_FEATURES,
        num_heads=8, dropout=0.0 
    ).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"--> Loading Checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("--> Weights Loaded Successfully!")
        except Exception as e:
            print(f"!!! Error loading weights: {e}")
            exit()
    else:
        print(f"!!! Warning: Checkpoint {CHECKPOINT_PATH} not found.")

    evaluate(model, val_loader, processor)
