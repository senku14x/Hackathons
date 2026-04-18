"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_01_data_loading.py

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

# Step 1 — Data Loading & CRS Inspection
# Mounts Google Drive, discovers LAS/LAZ files, selects the smallest village as a prototype, and validates coordinate reference system. Coordinate integrity is critical: hydrology is physically invalid in geographic degrees — the pipeline enforces UTM reprojection downstream.

# ==============================================================================


"""
TerrainFlow - Step 1: Data Loading & Inspection
Team ID: Nati-250330 | National Geo-AI Hackathon

Purpose:
- Mount Google Drive (if running in Colab)
- Discover LAS/LAZ and orthophoto files
- Select a small prototype LAZ for rapid iteration
- Inspect available point attributes (classification/RGB/returns/etc.)
- Produce quick diagnostic plots for sanity checks

Important:
- Many SVAMITVA point clouds may be unclassified (all class=0). This is expected.
- If coordinates are in EPSG:4326 (degrees), do NOT run gridding/hydrology in degrees.
  Reproject to a projected CRS (e.g., UTM) before terrain and hydrology stages.
"""

from pathlib import Path
import numpy as np
import laspy
import matplotlib.pyplot as plt
import torch

# -----------------------------------------------------------------------------
# 1.1 Mount Google Drive (Colab only) and verify GPU availability
# -----------------------------------------------------------------------------
def try_mount_drive(mount_path: str = "/content/drive") -> None:
    """
    Attempts to mount Google Drive in Colab.
    Safe to call outside Colab (will no-op).
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_path)
    except Exception:
        # Not running in Colab or Drive mount unavailable
        pass


def get_gpu_info() -> dict:
    """
    Returns basic GPU info if CUDA is available.
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_mem_gb": None,
    }
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


try_mount_drive()
GPU_INFO = get_gpu_info()

# -----------------------------------------------------------------------------
# 1.2 Define base path and discover data files
# -----------------------------------------------------------------------------
BASE_PATH = Path("/content/drive/MyDrive/NAT_GEO_HACKATHON")

if not BASE_PATH.exists():
    raise FileNotFoundError(
        f"BASE_PATH does not exist: {BASE_PATH}\n"
        "Verify: Drive is mounted, folder name is correct, and data is present."
    )

# Discover files recursively
laz_files = sorted(list(BASE_PATH.glob("**/*.laz")))
las_files = sorted(list(BASE_PATH.glob("**/*.las")))
tif_files = sorted(list(BASE_PATH.glob("**/*.tif")) + list(BASE_PATH.glob("**/*.tiff")))

pointcloud_files = las_files + laz_files

if not pointcloud_files:
    raise FileNotFoundError(f"No .las/.laz files found under: {BASE_PATH}")

# -----------------------------------------------------------------------------
# 1.3 Select prototype file (smallest point cloud file)
# -----------------------------------------------------------------------------
sizes = [(f, f.stat().st_size) for f in pointcloud_files]
sizes.sort(key=lambda x: x[1])

PROTOTYPE_FILE = sizes[0][0]
VILLAGE_NAME = PROTOTYPE_FILE.stem

# -----------------------------------------------------------------------------
# 1.4 Load prototype point cloud (full read only for the selected prototype)
# -----------------------------------------------------------------------------
# NOTE: This can still be large; since this is the smallest file, it should be safe.
las = laspy.read(PROTOTYPE_FILE)

n_points = len(las.points)
point_format = str(las.point_format)
las_version = str(las.header.version)

# -----------------------------------------------------------------------------
# 1.5 Inspect available attributes
# -----------------------------------------------------------------------------
has_intensity = hasattr(las, "intensity")
has_return_number = hasattr(las, "return_number")
has_num_returns = hasattr(las, "number_of_returns")
has_classification = hasattr(las, "classification")
has_rgb = hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")

extra_dims = list(las.point_format.extra_dimension_names)

# Classification distribution (if present)
unique_classes = np.unique(las.classification) if has_classification else np.array([])

# -----------------------------------------------------------------------------
# 1.6 Spatial extent sanity check
# -----------------------------------------------------------------------------
x_min, x_max = float(las.x.min()), float(las.x.max())
y_min, y_max = float(las.y.min()), float(las.y.max())
z_min, z_max = float(las.z.min()), float(las.z.max())

x_span = x_max - x_min
y_span = y_max - y_min
z_span = z_max - z_min

# Heuristic check: if x/y spans are "tiny" in degrees and values look like lon/lat,
# treat coordinates as geographic (EPSG:4326-like).
# This is a heuristic; the authoritative CRS should come from dataset metadata if available.
looks_like_degrees = (abs(x_min) <= 180 and abs(x_max) <= 180 and abs(y_min) <= 90 and abs(y_max) <= 90)

# Approximate meters only for quick sanity checks (NOT for processing).
# Proper approach: reproject to UTM before DTM and hydrology.
if looks_like_degrees:
    lat_center = 0.5 * (y_min + y_max)
    lon_to_m = 111_320 * np.cos(np.radians(lat_center))
    lat_to_m = 111_000
    width_m_approx = x_span * lon_to_m
    height_m_approx = y_span * lat_to_m
    area_m2_approx = max(width_m_approx * height_m_approx, 0.0)
else:
    # If already projected meters, treat spans as meters
    width_m_approx = x_span
    height_m_approx = y_span
    area_m2_approx = max(width_m_approx * height_m_approx, 0.0)

point_density_approx = (n_points / area_m2_approx) if area_m2_approx > 0 else 0.0

# -----------------------------------------------------------------------------
# 1.7 Quick plots (sampled for speed)
# -----------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)
sample_size = min(100_000, n_points)
sample_idx = rng.choice(n_points, size=sample_size, replace=False)

x_s = las.x[sample_idx]
y_s = las.y[sample_idx]
z_s = las.z[sample_idx]

# Use correct axis labels: degrees if geographic; meters if projected.
x_label = "Longitude (deg)" if looks_like_degrees else "X (m)"
y_label = "Latitude (deg)" if looks_like_degrees else "Y (m)"

# Top-down scatter colored by elevation
plt.figure(figsize=(10, 8))
sc = plt.scatter(x_s, y_s, c=z_s, s=0.1, cmap="terrain", alpha=0.6)
plt.colorbar(sc, label="Elevation (m)")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f"{VILLAGE_NAME} - Top-down view (colored by elevation)")
plt.axis("equal")
plt.tight_layout()
plt.savefig(f"{VILLAGE_NAME}_topdown_view.png", dpi=300, bbox_inches="tight")
plt.close()

# Elevation histogram (use full z for stable distribution)
plt.figure(figsize=(10, 3))
plt.hist(las.z, bins=100, edgecolor="black", alpha=0.7)
plt.xlabel("Elevation (m)")
plt.ylabel("Count")
plt.title(f"{VILLAGE_NAME} - Elevation distribution")
plt.tight_layout()
plt.savefig(f"{VILLAGE_NAME}_elevation_histogram.png", dpi=300, bbox_inches="tight")
plt.close()

# Optional RGB plot
if has_rgb:
    # LAS RGB is often 16-bit; normalize to [0,1]
    r = las.red[sample_idx].astype(np.float32)
    g = las.green[sample_idx].astype(np.float32)
    b = las.blue[sample_idx].astype(np.float32)
    denom = 65535.0 if (r.max() > 255 or g.max() > 255 or b.max() > 255) else 255.0
    rgb = np.stack([r, g, b], axis=1) / denom

    plt.figure(figsize=(10, 8))
    plt.scatter(x_s, y_s, c=rgb, s=0.1, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{VILLAGE_NAME} - Top-down RGB view")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{VILLAGE_NAME}_rgb_view.png", dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------------------------
# 1.8 Summary dictionary (useful for later steps without noisy printing)
# -----------------------------------------------------------------------------
INSPECTION_SUMMARY = {
    "base_path": str(BASE_PATH),
    "prototype_file": str(PROTOTYPE_FILE),
    "village_name": VILLAGE_NAME,
    "n_points": int(n_points),
    "point_format": point_format,
    "las_version": las_version,
    "has_intensity": bool(has_intensity),
    "has_return_number": bool(has_return_number),
    "has_number_of_returns": bool(has_num_returns),
    "has_classification": bool(has_classification),
    "has_rgb": bool(has_rgb),
    "extra_dimensions": extra_dims,
    "unique_classes": unique_classes.tolist() if has_classification else [],
    "looks_like_degrees": bool(looks_like_degrees),
    "x_range": [x_min, x_max],
    "y_range": [y_min, y_max],
    "z_range": [z_min, z_max],
    "x_span": float(x_span),
    "y_span": float(y_span),
    "z_span": float(z_span),
    "approx_width_m": float(width_m_approx),
    "approx_height_m": float(height_m_approx),
    "approx_area_m2": float(area_m2_approx),
    "approx_point_density_per_m2": float(point_density_approx),
    "gpu_info": GPU_INFO,
}

