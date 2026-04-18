"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_02_csf_pseudolabels.py

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

# Step 2 — CSF Pseudo-Labeling
# Cloth Simulation Filtering (CSF) generates weak ground / non-ground labels.
# These are **pseudo-labels** — a geometric prior — not final ground truth.
# CSF is known to over-smooth dense abadi terrain; the next step corrects this via ML refinement.
# | Parameter | Value | Rationale |
# |-----------|-------|-----------|
# | `cloth_resolution` | 0.5 m | Captures narrow lanes & courtyards |
# | `rigidness` | 3 | Soft cloth to follow embankments |
# | `class_threshold` | 0.5 m | Standard threshold |
# | `iterations` | 500 | Sufficient convergence |

# ==============================================================================


"""
TerrainFlow - Step 2: CSF-Based Pseudo-Labeling

Purpose:
- Use Cloth Simulation Filtering (CSF) to generate weak ground / non-ground labels
- These labels act as a geometric prior for downstream learning or direct DTM generation

Important:
- CSF is known to underperform in dense abadi regions; labels are treated as *pseudo-labels*
- This step prioritizes robustness and speed over perfect classification
"""

from pathlib import Path
import numpy as np
import laspy
import CSF
import time
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 2.1 Configuration
# -----------------------------------------------------------------------------
PROTOTYPE_FILE = Path(
    "/content/drive/MyDrive/NAT_GEO_HACKATHON/PureGP_Ortho_Point_data/RF_209183Pure.laz"
)
VILLAGE_NAME = PROTOTYPE_FILE.stem

OUTPUT_DIR = Path("/content/terrainflow_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CSF parameters tuned for rural village terrain
# - Moderate rigidness to avoid over-flattening
# - Sub-meter cloth resolution to capture courtyards and lanes
CSF_PARAMS = {
    "bSloopSmooth": False,
    "cloth_resolution": 0.5,
    "rigidness": 3,
    "class_threshold": 0.5,
    "iterations": 500,
}

# -----------------------------------------------------------------------------
# 2.2 Load point cloud (prototype-scale only)
# -----------------------------------------------------------------------------
# NOTE:
# This loads the full file into memory and is intended only for the smallest village.
# For full-scale deployment, tiling or PDAL-based streaming is required.
start = time.time()
las = laspy.read(PROTOTYPE_FILE)
xyz = np.vstack((las.x, las.y, las.z)).T
load_time = time.time() - start

n_points = xyz.shape[0]

# -----------------------------------------------------------------------------
# 2.3 Run Cloth Simulation Filtering (CSF)
# -----------------------------------------------------------------------------
csf = CSF.CSF()
csf.params.bSloopSmooth = CSF_PARAMS["bSloopSmooth"]
csf.params.cloth_resolution = CSF_PARAMS["cloth_resolution"]
csf.params.rigidness = CSF_PARAMS["rigidness"]
csf.params.class_threshold = CSF_PARAMS["class_threshold"]
csf.params.interations = CSF_PARAMS["iterations"]  # CSF library typo

csf.setPointCloud(xyz)

ground_idx = CSF.VecInt()
non_ground_idx = CSF.VecInt()

start = time.time()
csf.do_filtering(ground_idx, non_ground_idx)
csf_time = time.time() - start

# -----------------------------------------------------------------------------
# 2.4 Convert CSF output to pseudo-labels
# -----------------------------------------------------------------------------
# LAS convention:
#   2 = Ground
#   1 = Non-ground / unassigned
pseudo_labels = np.empty(n_points, dtype=np.int32)
pseudo_labels.fill(1)  # default: non-ground

for i in range(ground_idx.size()):
    pseudo_labels[ground_idx[i]] = 2

ground_mask = pseudo_labels == 2
ground_count = int(ground_mask.sum())
non_ground_count = n_points - ground_count

# -----------------------------------------------------------------------------
# 2.5 Extract RGB features (if available)
# -----------------------------------------------------------------------------
# RGB is optional and used only if present
if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
    rgb = np.stack(
        [
            las.red.astype(np.float32),
            las.green.astype(np.float32),
            las.blue.astype(np.float32),
        ],
        axis=1,
    )
    # Normalize assuming 16-bit color depth
    rgb /= 65535.0
else:
    rgb = None

# IMPORTANT:
# Surface normals are NOT assumed to exist.
# If needed later, they must be explicitly computed from XYZ.

# -----------------------------------------------------------------------------
# 2.6 Save pseudo-labeled dataset
# -----------------------------------------------------------------------------
pseudo_file = OUTPUT_DIR / f"{VILLAGE_NAME}_csf_pseudolabels.npz"
np.savez_compressed(
    pseudo_file,
    xyz=xyz.astype(np.float32),
    pseudo_labels=pseudo_labels,
    rgb=rgb.astype(np.float32) if rgb is not None else None,
)

ground_xyz = xyz[ground_mask]
ground_file = OUTPUT_DIR / f"{VILLAGE_NAME}_csf_ground_xyz.npz"
np.savez_compressed(ground_file, xyz=ground_xyz.astype(np.float32))

# -----------------------------------------------------------------------------
# 2.7 Lightweight visualization (diagnostic only)
# -----------------------------------------------------------------------------
# Sample for plotting
rng = np.random.default_rng(seed=42)
sample_size = min(100_000, n_points)
idx = rng.choice(n_points, size=sample_size, replace=False)

plt.figure(figsize=(10, 8))
plt.scatter(
    xyz[idx, 0],
    xyz[idx, 1],
    c=pseudo_labels[idx],
    s=0.2,
    cmap="coolwarm",
    alpha=0.6,
)
plt.title(f"{VILLAGE_NAME} - CSF Ground (2) vs Non-Ground (1)")
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"{VILLAGE_NAME}_csf_overview.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 2.8 Structured summary (no print spam)
# -----------------------------------------------------------------------------
CSF_SUMMARY = {
    "village": VILLAGE_NAME,
    "total_points": n_points,
    "ground_points": ground_count,
    "non_ground_points": non_ground_count,
    "ground_fraction_percent": 100.0 * ground_count / n_points,
    "csf_runtime_seconds": csf_time,
    "load_time_seconds": load_time,
    "pseudo_label_file": str(pseudo_file),
    "ground_xyz_file": str(ground_file),
}

