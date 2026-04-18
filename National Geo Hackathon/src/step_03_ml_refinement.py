"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_03_ml_refinement.py

Pipeline:
  step_01  →  Data Loading & CRS Validation
  step_02  →  CSF Pseudo-Labeling
  step_03  →  Residual MLP Ground Refinement
  step_04  →  DTM Generation (0.5 m UTM, IDW gap-fill)
  step_05_06 → Hydrological Analysis & Drainage Network Design
  step_07  →  Visualisations (waterlogging, drainage overlay, gallery)
  step_08  →  Metrics Table & Final Summary Report
  step_09  →  Export / Download Outputs
  deploy   →  Batch processing across all villages

Run in order: step_00_install.py first (once per environment), then 01 → 09.
"""


# ==============================================================================

# Step 3 — AI Ground Refinement (Residual MLP)
# A lightweight point-wise Residual MLP is trained on CSF pseudo-labels (no manual annotation needed).
# Input: XYZ ± RGB per point → Output: Ground vs Non-ground classification.
# **Why point-wise?** Learns local elevation/texture patterns that CSF misses in dense abadi areas.
# **Training:** 30 epochs, AMP mixed precision, early stopping. Batch size tuned for A100; reduce for smaller GPUs (see comments).

# ==============================================================================


"""
TerrainFlow - Step 3: Lightweight AI-Based Ground Refinement (Point-wise)

Purpose:
- Train a lightweight point-wise refinement network on CSF pseudo-labels.
- This is NOT RandLA-Net. It does not use neighborhood aggregation or hierarchical encoding.
- The role of this step is optional: it can smooth some CSF errors and improve
  ground continuity for DTM generation.

Outputs:
- A trained refinement model checkpoint
- Refined labels for the full point cloud
- Refined ground points (.npz) for downstream DTM generation
- Training curves (PNG)

Notes:
- All reported "agreement" metrics are agreement with CSF pseudo-labels (diagnostic only).
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# 3.1 Configuration (A100 OPTIMIZED)
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PSEUDO_LABEL_FILE = Path("/content/terrainflow_outputs/RF_209183Pure_csf_pseudolabels.npz")
OUTPUT_DIR = Path("/content/terrainflow_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VILLAGE_NAME = PSEUDO_LABEL_FILE.stem.replace("_csf_pseudolabels", "")

TRAIN_CONFIG = {
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 3,            # 0 unused; 1 non-ground; 2 ground
    "sample_ratio": 0.30,        # train on a subset for speed
    "print_every": 5,
    "early_stop_patience": 8,
    "early_stop_delta": 1e-3,
    "seed": 42,

    # --- A100 TURBO SETTINGS ---
    # 80GB VRAM allows massive throughput.
    "inference_chunk": 10_000_000,   # Process 10M points at once (Safe on 80GB)
    "train_batch_size": 524_288,     # Massive batch size stabilizes gradients & speeds up epoch
    "num_workers": 8,                # Feed data faster to keep GPU busy
    "use_amp": True,                 # Mixed precision for A100 acceleration
}

print(f"[Step 3] Device: {device}")
if torch.cuda.is_available():
    print(f"[Step 3] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Step 3] A100 Optimization Active: Batch={TRAIN_CONFIG['train_batch_size']:,}, Chunk={TRAIN_CONFIG['inference_chunk']:,}")

print(f"[Step 3] Loading pseudo-label data: {PSEUDO_LABEL_FILE.name}")

# -----------------------------------------------------------------------------
# 3.2 Load pseudo-labeled data
# -----------------------------------------------------------------------------
data = np.load(PSEUDO_LABEL_FILE, allow_pickle=True)

xyz = data["xyz"].astype(np.float32)
pseudo_labels = data["pseudo_labels"].astype(np.int32)

# Features handling:
if "features" in data and data["features"] is not None:
    features = data["features"].astype(np.float32)
    feature_source = "features"
elif "rgb" in data and data["rgb"] is not None:
    features = data["rgb"].astype(np.float32)
    feature_source = "rgb"
else:
    features = xyz
    feature_source = "xyz"

n_points = xyz.shape[0]
feature_dim = features.shape[1]

if features.shape[0] != n_points:
    raise ValueError("Features and xyz length mismatch.")

unique, counts = np.unique(pseudo_labels, return_counts=True)
dist = {int(u): int(c) for u, c in zip(unique, counts)}

print(f"[Step 3] Points: {n_points:,} | Feature dim: {feature_dim} (source: {feature_source})")
print(f"[Step 3] Pseudo-label distribution: {dist}")

# -----------------------------------------------------------------------------
# 3.3 Subsample for training (reproducible)
# -----------------------------------------------------------------------------
rng = np.random.default_rng(TRAIN_CONFIG["seed"])
sample_size = int(n_points * TRAIN_CONFIG["sample_ratio"])
sample_idx = rng.choice(n_points, size=sample_size, replace=False)

train_features = features[sample_idx]
train_labels = pseudo_labels[sample_idx]

train_unique, train_counts = np.unique(train_labels, return_counts=True)
train_dist = {int(u): int(c) for u, c in zip(train_unique, train_counts)}
print(f"[Step 3] Training subset: {sample_size:,} points | distribution: {train_dist}")

# -----------------------------------------------------------------------------
# Feature normalization (important for stable MLP training)
# -----------------------------------------------------------------------------
feat_mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
feat_std = train_features.std(axis=0, keepdims=True).astype(np.float32)
feat_std = np.maximum(feat_std, 1e-6)

train_features = (train_features - feat_mean) / feat_std
features = (features - feat_mean) / feat_std

print("[Step 3] Applied feature standardization (mean/std from training subset).")

# -----------------------------------------------------------------------------
# 3.4 Dataset / Loader (A100 settings)
# -----------------------------------------------------------------------------
class PointwiseDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        self.feats = torch.from_numpy(feats).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

train_loader = DataLoader(
    PointwiseDataset(train_features, train_labels),
    batch_size=TRAIN_CONFIG["train_batch_size"],
    shuffle=True,
    num_workers=TRAIN_CONFIG["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

# -----------------------------------------------------------------------------
# 3.5 Model: Dense point-wise refinement network (Residual MLP)
# -----------------------------------------------------------------------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.15):
        super().__init__()
        hidden = dim * hidden_mult
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h

class DensePointwiseRefinementNet(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, width: int = 512, depth: int = 8, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )

        blocks = []
        for _ in range(depth):
            blocks.append(ResidualMLPBlock(width, hidden_mult=2, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)

# Instantiating a larger model for A100
model = DensePointwiseRefinementNet(
    in_dim=feature_dim,
    num_classes=TRAIN_CONFIG["num_classes"],
    width=512,       # Increased for A100
    depth=8,         # Increased for A100
    dropout=0.15
).to(device)

# --- A100 SPEEDUP: COMPILE MODEL ---
try:
    print("[Step 3] Compiling model for A100 (Torch 2.0+)...")
    model = torch.compile(model)
except Exception as e:
    print(f"[Step 3] Warning: Could not compile model (continuing without): {e}")

total_params = sum(p.numel() for p in model.parameters())
print(f"[Step 3] Model params: {total_params:,}")

# -----------------------------------------------------------------------------
# 3.6 Loss & Optimizer
# -----------------------------------------------------------------------------
class_counts = np.bincount(train_labels, minlength=TRAIN_CONFIG["num_classes"]).astype(np.float32)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * TRAIN_CONFIG["num_classes"]
class_weights_t = torch.from_numpy(class_weights).float().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])

# AMP setup
use_amp = bool(TRAIN_CONFIG["use_amp"] and device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -----------------------------------------------------------------------------
# 3.7 Training loop
# -----------------------------------------------------------------------------
train_losses = []
train_agreements = []

best_loss = float("inf")
patience = 0
best_state = None

print(f"[Step 3] Training for up to {TRAIN_CONFIG['num_epochs']} epochs...")

t0 = time.time()
for epoch in range(1, TRAIN_CONFIG["num_epochs"] + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t_epoch = time.time()

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * yb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))

    avg_loss = running_loss / max(total, 1)
    agreement = 100.0 * correct / max(total, 1)

    train_losses.append(avg_loss)
    train_agreements.append(agreement)

    improved = avg_loss < (best_loss - TRAIN_CONFIG["early_stop_delta"])
    if improved:
        best_loss = avg_loss
        patience = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience += 1

    if epoch == 1 or epoch % TRAIN_CONFIG["print_every"] == 0:
        print(
            f"[Step 3] Epoch {epoch:02d} | loss={avg_loss:.4f} | pseudo-agreement={agreement:.2f}% | "
            f"patience={patience} | time={time.time() - t_epoch:.1f}s"
        )

    if patience >= TRAIN_CONFIG["early_stop_patience"]:
        print(f"[Step 3] Early stop at epoch {epoch} (best_loss={best_loss:.4f})")
        break

total_train_time = time.time() - t0

if best_state is not None:
    model.load_state_dict(best_state)

print(f"[Step 3] Training done in {total_train_time/60:.2f} minutes | best_loss={best_loss:.4f}")

# -----------------------------------------------------------------------------
# 3.8 Save model checkpoint
# -----------------------------------------------------------------------------
model_file = OUTPUT_DIR / f"{VILLAGE_NAME}_dense_pointwise_refinement_model.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": TRAIN_CONFIG,
        "feature_source": feature_source,
        "feature_dim": feature_dim,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "train_losses": train_losses,
        "train_pseudo_agreement": train_agreements,
    },
    model_file,
)
print(f"[Step 3] Saved model: {model_file.name}")

# -----------------------------------------------------------------------------
# 3.9 Inference on full point cloud (Massive Chunk for A100)
# -----------------------------------------------------------------------------
print(f"[Step 3] Running inference on {n_points:,} points (chunk={TRAIN_CONFIG['inference_chunk']:,})")

model.eval()
refined_labels = np.empty(n_points, dtype=np.int32)

chunk = int(TRAIN_CONFIG["inference_chunk"])
with torch.no_grad():
    for start in range(0, n_points, chunk):
        end = min(start + chunk, n_points)
        xb = torch.from_numpy(features[start:end]).float().to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
        refined_labels[start:end] = logits.argmax(dim=1).cpu().numpy().astype(np.int32)

agreement_full = 100.0 * float((refined_labels == pseudo_labels).mean())
print(f"[Step 3] Full-cloud pseudo-agreement with CSF labels: {agreement_full:.2f}%")

# -----------------------------------------------------------------------------
# 3.10 Save refined predictions and refined ground points
# -----------------------------------------------------------------------------
refined_file = OUTPUT_DIR / f"{VILLAGE_NAME}_refined_predictions_dense_mlp.npz"
np.savez_compressed(
    refined_file,
    xyz=xyz,
    csf_labels=pseudo_labels,
    refined_labels=refined_labels,
    feature_source=feature_source,
)

refined_ground_xyz = xyz[refined_labels == 2]
refined_ground_file = OUTPUT_DIR / f"{VILLAGE_NAME}_refined_ground_xyz_dense_mlp.npz"
np.savez_compressed(refined_ground_file, xyz=refined_ground_xyz.astype(np.float32))

print(f"[Step 3] Saved predictions: {refined_file.name}")
print(f"[Step 3] Saved refined ground points: {refined_ground_file.name} | n={refined_ground_xyz.shape[0]:,}")

# -----------------------------------------------------------------------------
# 3.11 Training curves
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(train_losses, linewidth=2)
ax[0].set_title("Training loss (pseudo-supervision)")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].grid(True, alpha=0.3)

ax[1].plot(train_agreements, linewidth=2)
ax[1].set_title("Agreement with pseudo-labels")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Agreement (%)")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
curve_file = OUTPUT_DIR / f"{VILLAGE_NAME}_step3_training_curves_dense_mlp.png"
plt.savefig(curve_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"[Step 3] Saved training curves: {curve_file.name}")

STEP3_SUMMARY = {
    "village": VILLAGE_NAME,
    "n_points": int(n_points),
    "feature_source": feature_source,
    "feature_dim": int(feature_dim),
    "training_points": int(sample_size),
    "best_loss": float(best_loss),
    "train_time_seconds": float(total_train_time),
    "full_pseudo_agreement_percent": float(agreement_full),
    "model_file": str(model_file),
    "refined_predictions_file": str(refined_file),
    "refined_ground_file": str(refined_ground_file),
    "train_batch_size": int(TRAIN_CONFIG["train_batch_size"]),
    "use_amp": bool(use_amp),
}
