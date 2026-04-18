"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_04_dtm_generation.py

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

# Step 4 — DTM Generation (UTM Raster Binning)
# Converts refined ground points to a 0.5 m resolution GeoTIFF in UTM Zone 44N (EPSG:32644).
# Generates multiple DTMs with different per-cell statistics (median, p05) for robustness analysis.
# Gap filling uses IDW (k=8, power=2). Output is hydrology-ready.
# **Outputs:**
# - `<VILLAGE>_DTM_refined_UTM_median.tif` — main terrain model
# - `<VILLAGE>_DTM_refined_UTM_p05.tif` — conservative bare-earth estimate
# - `<VILLAGE>_DTM_csf_UTM_median.tif` — CSF baseline for comparison

# ==============================================================================


"""
TerrainFlow - Step 4: DTM Generation (UTM, fast raster binning) + Inspection

Adds:
- Strong inspection diagnostics (coverage, point-per-cell stats, fill ratio, roughness)
- Robust "bare-earth" rasters using low-percentile per-cell stats (p05/p10/min)
- Saves multiple GeoTIFFs + comparisons

Outputs (examples):
- <VILLAGE>_DTM_refined_UTM_median.tif
- <VILLAGE>_DTM_refined_UTM_p05.tif
- <VILLAGE>_DTM_csf_UTM_median.tif
- <VILLAGE>_DTM_csf_UTM_p05.tif
- PNG comparisons
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

import rasterio
from rasterio.transform import from_origin

from pyproj import CRS, Transformer
from matplotlib.colors import LightSource


# -----------------------------------------------------------------------------
# 4.1 Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"

CSF_GROUND_FILE = OUTPUT_DIR / f"{VILLAGE_NAME}_csf_ground_xyz.npz"
REFINED_GROUND_FILE = OUTPUT_DIR / f"{VILLAGE_NAME}_refined_ground_xyz_dense_mlp.npz"

print("[Step 4] Using:")
print("  CSF_GROUND_FILE     =", CSF_GROUND_FILE)
print("  REFINED_GROUND_FILE =", REFINED_GROUND_FILE)

if not CSF_GROUND_FILE.exists():
    raise FileNotFoundError(f"Missing CSF ground file: {CSF_GROUND_FILE}")
if not REFINED_GROUND_FILE.exists():
    raise FileNotFoundError(f"Missing refined ground file: {REFINED_GROUND_FILE}")

DTM_CONFIG = {
    "resolution_m": 0.5,          # meters
    "fill_method": "idw",         # "nearest" or "idw"
    "idw_k": 8,                   # neighbors for IDW
    "idw_power": 2.0,             # IDW power
    "smoothing_sigma": 1.0,       # gaussian smoothing in pixels (0 disables)
    "nodata_value": -9999.0,

    # We'll generate multiple DTMs with different cell statistics.
    # For hydrology/bare-earth, low-percentile tends to be safer than median.
    "cell_stats_to_make": ["median", "p05"],  # choose from: mean, median, min, p05, p10
}

print("[Step 4] DTM Generation (UTM, fast binning)")
print(f"[Step 4] Village: {VILLAGE_NAME}")
print(f"[Step 4] Config: {DTM_CONFIG}")


# -----------------------------------------------------------------------------
# 4.2 Load ground points
# -----------------------------------------------------------------------------
def load_xyz(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    xyz = d["xyz"].astype(np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Bad xyz shape in {npz_path}: {xyz.shape}")
    return xyz

t0 = time.time()
csf_ground_ll = load_xyz(CSF_GROUND_FILE)
ref_ground_ll = load_xyz(REFINED_GROUND_FILE)
print(f"[Step 4] Loaded CSF points:     {csf_ground_ll.shape[0]:,}")
print(f"[Step 4] Loaded refined points: {ref_ground_ll.shape[0]:,}")
print(f"[Step 4] Load time: {time.time()-t0:.2f}s")

def looks_like_lonlat(xyz: np.ndarray) -> bool:
    x = xyz[:, 0]
    y = xyz[:, 1]
    return (np.all(np.abs(x) <= 180.0) and np.all(np.abs(y) <= 90.0))

print("[Step 4] Lon/Lat heuristic:",
      "CSF=", looks_like_lonlat(csf_ground_ll),
      "| Refined=", looks_like_lonlat(ref_ground_ll))

print("[Step 4] Raw extent (input coords):")
def print_extent(xyz, name):
    print(f"  {name}: X[{xyz[:,0].min():.6f},{xyz[:,0].max():.6f}]  "
          f"Y[{xyz[:,1].min():.6f},{xyz[:,1].max():.6f}]  "
          f"Z[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")
print_extent(csf_ground_ll, "CSF")
print_extent(ref_ground_ll, "Refined")


# -----------------------------------------------------------------------------
# 4.3 Build UTM transformer from center of refined extent
# -----------------------------------------------------------------------------
lon_center = float(0.5 * (ref_ground_ll[:, 0].min() + ref_ground_ll[:, 0].max()))
lat_center = float(0.5 * (ref_ground_ll[:, 1].min() + ref_ground_ll[:, 1].max()))

utm_zone = int(np.floor((lon_center + 180.0) / 6.0) + 1)
is_north = lat_center >= 0
epsg_utm = 32600 + utm_zone if is_north else 32700 + utm_zone

crs_src = CRS.from_epsg(4326)
crs_dst = CRS.from_epsg(epsg_utm)

transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
print(f"[Step 4] UTM: zone={utm_zone} hemisphere={'N' if is_north else 'S'} EPSG:{epsg_utm}")

def reproj_ll_to_utm(xyz_ll: np.ndarray) -> np.ndarray:
    x, y = transformer.transform(xyz_ll[:, 0], xyz_ll[:, 1])
    z = xyz_ll[:, 2]
    return np.stack([x, y, z], axis=1).astype(np.float64)

t0 = time.time()
csf_ground = reproj_ll_to_utm(csf_ground_ll)
ref_ground = reproj_ll_to_utm(ref_ground_ll)
print(f"[Step 4] Reprojection time: {time.time()-t0:.2f}s")

print("[Step 4] Extent (UTM meters):")
print_extent(csf_ground, "CSF_UTM")
print_extent(ref_ground, "REF_UTM")


# -----------------------------------------------------------------------------
# 4.4 Define grid (meters) using refined extent (more complete)
# -----------------------------------------------------------------------------
res = float(DTM_CONFIG["resolution_m"])

x_min = float(ref_ground[:, 0].min())
x_max = float(ref_ground[:, 0].max())
y_min = float(ref_ground[:, 1].min())
y_max = float(ref_ground[:, 1].max())

width_m = x_max - x_min
height_m = y_max - y_min

ncols = int(np.ceil(width_m / res))
nrows = int(np.ceil(height_m / res))

cells = nrows * ncols
approx_mem_mb = cells * 8 / (1024**2)  # float64, rough
print(f"[Step 4] Grid: {nrows} rows x {ncols} cols  (res={res} m, cells={cells:,}, ~{approx_mem_mb:.1f} MB/float64 layer)")

transform = from_origin(x_min, y_max, res, res)  # top-left origin


# -----------------------------------------------------------------------------
# 4.5 Fast gridding by binning points into cells
# -----------------------------------------------------------------------------
def compute_rowcol(points_xyz: np.ndarray):
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    # col: left->right, row: top->bottom (y_max at row 0)
    col = np.floor((x - x_min) / res).astype(np.int64)
    row = np.floor((y_max - y) / res).astype(np.int64)
    valid = (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
    return row[valid], col[valid], points_xyz[:, 2][valid]

def rasterize_points_to_grid(points_xyz: np.ndarray, cell_stat: str = "median") -> np.ndarray:
    """
    cell_stat: mean | median | min | p05 | p10
    returns grid float64 with NaNs where empty
    """
    row, col, z = compute_rowcol(points_xyz)
    lin = row * ncols + col

    order = np.argsort(lin)
    lin_sorted = lin[order]
    z_sorted = z[order]

    dtm = np.full((nrows * ncols,), np.nan, dtype=np.float64)

    if cell_stat == "mean":
        sums = np.bincount(lin_sorted, weights=z_sorted, minlength=nrows*ncols).astype(np.float64)
        cnts = np.bincount(lin_sorted, minlength=nrows*ncols).astype(np.float64)
        m = np.divide(sums, np.maximum(cnts, 1.0))
        m[cnts == 0] = np.nan
        return m.reshape((nrows, ncols))

    # stats that require per-cell slices
    uniq, start_idx = np.unique(lin_sorted, return_index=True)
    ends = np.r_[start_idx[1:], len(lin_sorted)]

    if cell_stat == "min":
        for u, s, e in zip(uniq, start_idx, ends):
            dtm[u] = np.min(z_sorted[s:e])
        return dtm.reshape((nrows, ncols))

    if cell_stat == "median":
        for u, s, e in zip(uniq, start_idx, ends):
            dtm[u] = np.median(z_sorted[s:e])
        return dtm.reshape((nrows, ncols))

    if cell_stat in ("p05", "p10"):
        q = 5 if cell_stat == "p05" else 10
        for u, s, e in zip(uniq, start_idx, ends):
            dtm[u] = np.percentile(z_sorted[s:e], q)
        return dtm.reshape((nrows, ncols))

    raise ValueError("cell_stat must be one of: mean, median, min, p05, p10")

def fill_gaps(dtm: np.ndarray, method: str = "nearest", k: int = 8, power: float = 2.0) -> np.ndarray:
    out = dtm.copy()
    nan_mask = np.isnan(out)
    if nan_mask.sum() == 0:
        return out

    valid_mask = ~nan_mask
    vr, vc = np.where(valid_mask)

    vx = x_min + (vc + 0.5) * res
    vy = y_max - (vr + 0.5) * res
    vz = out[valid_mask].astype(np.float64)

    tree = cKDTree(np.c_[vx, vy])

    mr, mc = np.where(nan_mask)
    mx = x_min + (mc + 0.5) * res
    my = y_max - (mr + 0.5) * res
    query_pts = np.c_[mx, my]

    if method == "nearest":
        _, idx = tree.query(query_pts, k=1)
        out[nan_mask] = vz[idx]
        return out

    if method == "idw":
        kk = min(int(k), len(vz))
        dist, idx = tree.query(query_pts, k=kk)
        if kk == 1:
            out[nan_mask] = vz[idx]
            return out
        w = 1.0 / np.maximum(dist, 1e-6) ** power
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        out[nan_mask] = (vz[idx] * w).sum(axis=1)
        return out

    raise ValueError("fill_method must be 'nearest' or 'idw'")


# -----------------------------------------------------------------------------
# 4.5B Inspection helpers
# -----------------------------------------------------------------------------
def point_per_cell_report(points_xyz: np.ndarray, name: str):
    row, col, _ = compute_rowcol(points_xyz)
    lin = row * ncols + col
    counts = np.bincount(lin, minlength=nrows*ncols)
    nonzero = counts[counts > 0]
    coverage = 100.0 * (nonzero.size / counts.size)
    if nonzero.size == 0:
        print(f"[Inspect] {name}: no points landed in grid (check CRS/resolution).")
        return
    p50 = int(np.percentile(nonzero, 50))
    p90 = int(np.percentile(nonzero, 90))
    p99 = int(np.percentile(nonzero, 99))
    print(f"[Inspect] {name}: cells w/points = {nonzero.size:,}/{counts.size:,} ({coverage:.2f}%)")
    print(f"[Inspect] {name}: points/cell (non-empty) min={nonzero.min()} p50={p50} p90={p90} p99={p99} max={nonzero.max()}")
    # "sparse warning" heuristic
    if p50 <= 1:
        print(f"[Inspect] {name}: WARNING: median points-per-nonempty-cell <= 1 (DTM may be interpolation-dominated at 0.5m).")

def roughness(a: np.ndarray) -> float:
    g = a.copy()
    dx = np.nan_to_num(np.diff(g, axis=1), nan=0.0)  # shape: (nrows, ncols-1)
    dy = np.nan_to_num(np.diff(g, axis=0), nan=0.0)  # shape: (nrows-1, ncols)
    # To combine, trim both to (nrows-1, ncols-1)
    return float(np.sqrt(dx[:-1, :]**2 + dy[:, :-1]**2).mean())

def summarize_grid(a: np.ndarray, name: str):
    valid = np.isfinite(a)
    print(f"[Stats] {name}: valid={valid.sum():,}/{a.size:,} ({100.0*valid.mean():.2f}%) "
          f"min/med/max={np.nanmin(a):.2f}/{np.nanmedian(a):.2f}/{np.nanmax(a):.2f} "
          f"mean/std={np.nanmean(a):.2f}/{np.nanstd(a):.2f} rough={roughness(a):.4f}")


# -----------------------------------------------------------------------------
# 4.6 Generate DTMs (Refined vs CSF) for multiple cell stats
# -----------------------------------------------------------------------------
def make_dtm(points_xyz: np.ndarray, name: str, cell_stat: str) -> np.ndarray:
    t0 = time.time()
    dtm = rasterize_points_to_grid(points_xyz, cell_stat=cell_stat)
    nan0 = int(np.isnan(dtm).sum())
    pct0 = 100.0 * nan0 / dtm.size
    print(f"[Step 4] {name}({cell_stat}): initial NaNs: {nan0:,} ({pct0:.2f}%) | {time.time()-t0:.2f}s")

    t1 = time.time()
    dtm_filled = fill_gaps(
        dtm,
        method=DTM_CONFIG["fill_method"],
        k=int(DTM_CONFIG["idw_k"]),
        power=float(DTM_CONFIG["idw_power"]),
    )
    nan1 = int(np.isnan(dtm_filled).sum())
    pct1 = 100.0 * nan1 / dtm_filled.size
    print(f"[Step 4] {name}({cell_stat}): after fill NaNs: {nan1:,} ({pct1:.2f}%) | {time.time()-t1:.2f}s")

    if float(DTM_CONFIG["smoothing_sigma"]) > 0:
        t2 = time.time()
        dtm_filled = gaussian_filter(dtm_filled, sigma=float(DTM_CONFIG["smoothing_sigma"]))
        print(f"[Step 4] {name}({cell_stat}): smoothing sigma={DTM_CONFIG['smoothing_sigma']} | {time.time()-t2:.2f}s")

    summarize_grid(dtm_filled, f"{name}({cell_stat})")
    return dtm_filled

print("\n" + "="*80)
print("[Step 4] Inspection: points-per-cell diagnostics (at chosen resolution)")
print("="*80)
point_per_cell_report(ref_ground, "Refined points")
point_per_cell_report(csf_ground, "CSF points")

dtms = {}  # (source, stat) -> grid
for stat in DTM_CONFIG["cell_stats_to_make"]:
    dtms[("refined", stat)] = make_dtm(ref_ground, "Refined", stat)
    dtms[("csf", stat)] = make_dtm(csf_ground, "CSF", stat)

# Primary diff for reporting: refined(stat0) - csf(stat0) using first stat in list
primary_stat = DTM_CONFIG["cell_stats_to_make"][0]
dtm_ref = dtms[("refined", primary_stat)]
dtm_csf = dtms[("csf", primary_stat)]
dtm_diff = dtm_ref - dtm_csf
summarize_grid(dtm_diff, f"Diff(refined-csf)({primary_stat})")

# Extra leakage check: compare refined(median) vs refined(p05) if both exist
if ("refined", "median") in dtms and ("refined", "p05") in dtms:
    leak_map = dtms[("refined", "median")] - dtms[("refined", "p05")]
    # If median is much higher than p05 in abadi, that suggests raised-object leakage.
    frac_gt2 = float((leak_map > 2.0).mean()) * 100.0
    print(f"[Inspect] Refined leakage proxy: fraction(median - p05 > 2m) = {frac_gt2:.2f}% (high -> roof/veg risk)")


# -----------------------------------------------------------------------------
# 4.7 Save GeoTIFFs (UTM)
# -----------------------------------------------------------------------------
def save_geotiff(path: Path, grid: np.ndarray):
    nodata = float(DTM_CONFIG["nodata_value"])
    out = grid.copy().astype(np.float32)
    out[np.isnan(out)] = nodata

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs=crs_dst.to_string(),
        transform=transform,
        compress="lzw",
        nodata=nodata,
    ) as dst:
        dst.write(out, 1)

saved_tifs = []
t0 = time.time()
for stat in DTM_CONFIG["cell_stats_to_make"]:
    ref_tif = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_{stat}.tif"
    csf_tif = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_csf_UTM_{stat}.tif"
    save_geotiff(ref_tif, dtms[("refined", stat)])
    save_geotiff(csf_tif, dtms[("csf", stat)])
    saved_tifs.extend([ref_tif, csf_tif])

print(f"[Step 4] Saved GeoTIFFs in {time.time()-t0:.2f}s")
for p in saved_tifs:
    print(f"  - {p.name} ({p.stat().st_size/(1024**2):.1f} MB)")


# -----------------------------------------------------------------------------
# 4.8 Visualizations (for primary_stat only)
# -----------------------------------------------------------------------------
cmp_png = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_comparison_UTM_{primary_stat}.png"
diff_png = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_difference_UTM_{primary_stat}.png"
hs_png   = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_hillshade_UTM_{primary_stat}.png"

# side-by-side
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
im1 = axes[0].imshow(dtm_ref, cmap="terrain")
axes[0].set_title(f"{VILLAGE_NAME} - Refined DTM (UTM) [{primary_stat}]")
plt.colorbar(im1, ax=axes[0], label="Elevation (m)")
im2 = axes[1].imshow(dtm_csf, cmap="terrain")
axes[1].set_title(f"{VILLAGE_NAME} - CSF DTM (UTM) [{primary_stat}]")
plt.colorbar(im2, ax=axes[1], label="Elevation (m)")
for ax in axes:
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
plt.tight_layout()
plt.savefig(cmp_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Step 4] Saved: {cmp_png.name}")

# difference map (robust scaling)
v = np.nanpercentile(np.abs(dtm_diff), 98)
v = float(max(v, 0.5))
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
im = ax.imshow(dtm_diff, cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_title(f"{VILLAGE_NAME} - DTM Difference (Refined - CSF) [m] [{primary_stat}]")
plt.colorbar(im, ax=ax, label="Elevation difference (m)")
plt.tight_layout()
plt.savefig(diff_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Step 4] Saved: {diff_png.name}")

# hillshade comparison
ls = LightSource(azdeg=315, altdeg=45)
hs_ref = ls.hillshade(dtm_ref, vert_exag=2.0, dx=res, dy=res)
hs_csf = ls.hillshade(dtm_csf, vert_exag=2.0, dx=res, dy=res)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].imshow(hs_ref, cmap="gray")
axes[0].set_title(f"{VILLAGE_NAME} - Refined Hillshade [{primary_stat}]")
axes[0].axis("off")
axes[1].imshow(hs_csf, cmap="gray")
axes[1].set_title(f"{VILLAGE_NAME} - CSF Hillshade [{primary_stat}]")
axes[1].axis("off")
plt.tight_layout()
plt.savefig(hs_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"[Step 4] Saved: {hs_png.name}")


# -----------------------------------------------------------------------------
# 4.9 Summary
# -----------------------------------------------------------------------------
print("[Step 4] Summary")
print(f"[Step 4] CRS: {crs_dst.to_string()} | Resolution: {res} m")
print(f"[Step 4] Grid: {nrows} x {ncols} cells")
print("[Step 4] Outputs:")
for p in saved_tifs:
    print("  -", p.name)
print("  -", cmp_png.name)
print("  -", diff_png.name)
print("  -", hs_png.name)

STEP4_SUMMARY = {
    "village": VILLAGE_NAME,
    "crs_utm_epsg": epsg_utm,
    "resolution_m": res,
    "grid_rows": nrows,
    "grid_cols": ncols,
    "primary_stat": primary_stat,
    "saved_tifs": [str(p) for p in saved_tifs],
    "comparison_png": str(cmp_png),
    "difference_png": str(diff_png),
    "hillshade_png": str(hs_png),
}
