"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_07_visualise_waterlogging.py

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

# Visualisation A — Waterlogging Hotspot Analysis (TWI)
# Computes and visualises TWI alongside the orthophoto for spatial context.

# ==============================================================================


"""
TerrainFlow - Waterlogging Hotspot Analysis
Proper TWI calculation and visualization
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import rasterio
from scipy.ndimage import sobel, gaussian_filter

OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"
BASE_PATH = Path("/content/drive/MyDrive/NAT_GEO_HACKATHON")

print("Calculating Waterlogging Hotspots (TWI)...\n")

# ============================================================================
# Load DTM and Orthophoto
# ============================================================================
dtm_file = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_p05.tif"  # Use p05 for conservative ground
if not dtm_file.exists():
    dtm_file = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif"

with rasterio.open(dtm_file) as src:
    dtm = src.read(1).astype(np.float64)
    dtm[dtm < -1000] = np.nan
    transform = src.transform
    crs = src.crs

cell_size = 0.5 * (abs(transform.a) + abs(transform.e))
print(f"DTM loaded: {dtm.shape}, cell size: {cell_size:.2f}m")

# Load orthophoto
ortho_file = BASE_PATH / "PureGP_Ortho_Point_data" / "209183Pure_transparent_mosaic_group1.tif"
if not ortho_file.exists():
    ortho_file = BASE_PATH / "PureGP_Ortho_Point_data" / "RF_209183Pure_ORTHO.tif"

ortho = None
if ortho_file.exists():
    try:
        with rasterio.open(ortho_file) as src:
            if src.count >= 3:
                r = src.read(1).astype(np.float32)
                g = src.read(2).astype(np.float32)
                b = src.read(3).astype(np.float32)
                ortho = np.dstack([r, g, b])
                if ortho.max() > 1:
                    ortho = ortho / 255.0
        print(f"Orthophoto loaded: {ortho.shape}\n")
    except:
        pass

# ============================================================================
# Step 1: Fill Depressions (Simple Priority Flood)
# ============================================================================
print("[Step 1/5] Filling depressions...")

def fill_depressions_simple(dem):
    """Priority flood depression filling"""
    from heapq import heappush, heappop

    filled = dem.copy()
    h, w = dem.shape
    visited = np.zeros((h, w), dtype=bool)
    pq = []

    # Initialize from edges
    for i in range(h):
        for j in range(w):
            if i == 0 or j == 0 or i == h-1 or j == w-1:
                if not np.isnan(dem[i, j]):
                    heappush(pq, (dem[i, j], i, j))
                    visited[i, j] = True

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while pq:
        elev, r, c = heappop(pq)
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if np.isnan(dem[nr, nc]):
                    visited[nr, nc] = True
                    continue
                filled[nr, nc] = max(filled[nr, nc], elev)
                heappush(pq, (filled[nr, nc], nr, nc))
                visited[nr, nc] = True

    return filled

dtm_filled = fill_depressions_simple(dtm)
print(f"  Depressions filled\n")

# ============================================================================
# Step 2: Calculate Slope
# ============================================================================
print("[Step 2/5] Calculating slope...")

dzdx = sobel(dtm_filled, axis=1) / (8 * cell_size)
dzdy = sobel(dtm_filled, axis=0) / (8 * cell_size)
slope = np.sqrt(dzdx**2 + dzdy**2)
slope = np.nan_to_num(slope, nan=0.0)
slope = np.maximum(slope, 0.001)  # Minimum slope to avoid division by zero

print(f"  Mean slope: {np.mean(slope[slope>0]):.4f}\n")

# ============================================================================
# Step 3: Calculate Flow Accumulation (D8)
# ============================================================================
print("[Step 3/5] Calculating flow accumulation...")

def flow_direction_d8(dem):
    """D8 flow direction"""
    h, w = dem.shape
    fd = np.zeros((h, w), dtype=np.int8)
    dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    for i in range(1, h-1):
        for j in range(1, w-1):
            z = dem[i, j]
            if np.isnan(z):
                continue
            best = -np.inf
            best_d = 0
            for d, (dy, dx) in enumerate(dirs):
                nz = dem[i+dy, j+dx]
                if np.isnan(nz):
                    continue
                dist = np.hypot(dy, dx)
                slope_val = (z - nz) / dist
                if slope_val > best:
                    best = slope_val
                    best_d = d
            fd[i, j] = best_d
    return fd

def flow_accumulation_d8(flow_dir, dem):
    """D8 flow accumulation"""
    h, w = dem.shape
    acc = np.ones((h, w), dtype=np.float64)
    dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    # Sort by elevation (high to low)
    order = np.argsort(dem.flatten())[::-1]

    for idx in order:
        i, j = np.unravel_index(idx, dem.shape)
        if np.isnan(dem[i, j]):
            continue
        d = flow_dir[i, j]
        dy, dx = dirs[d]
        ni, nj = i + dy, j + dx
        if 0 <= ni < h and 0 <= nj < w:
            acc[ni, nj] += acc[i, j]

    return acc

flow_dir = flow_direction_d8(dtm_filled)
flow_acc = flow_accumulation_d8(flow_dir, dtm_filled)

print(f"  Flow accumulation calculated\n")

# ============================================================================
# Step 4: Calculate Topographic Wetness Index (TWI)
# ============================================================================
print("[Step 4/5] Calculating TWI (Topographic Wetness Index)...")

# Specific Catchment Area (SCA) = flow_acc * cell_area / cell_width
sca = flow_acc * (cell_size ** 2) / cell_size

# TWI = ln(SCA / tan(slope))
twi = np.log(sca / slope)
twi = np.nan_to_num(twi, nan=0.0, posinf=0.0, neginf=0.0)

# Smooth TWI to reduce noise
twi_smooth = gaussian_filter(twi, sigma=2.0)

print(f"  TWI range: {twi_smooth.min():.2f} to {twi_smooth.max():.2f}")
print(f"  TWI mean: {twi_smooth.mean():.2f}\n")

# ============================================================================
# Step 5: Identify Waterlogging Hotspots
# ============================================================================
print("[Step 5/5] Identifying waterlogging hotspots...")

# Method 1: High TWI values (top 10%)
twi_threshold_90 = np.percentile(twi_smooth[twi_smooth > 0], 90)
twi_threshold_95 = np.percentile(twi_smooth[twi_smooth > 0], 95)

hotspots_moderate = twi_smooth >= twi_threshold_90  # Top 10% (moderate risk)
hotspots_high = twi_smooth >= twi_threshold_95      # Top 5% (high risk)

# Method 2: Combined criteria (high TWI + low slope + high flow accumulation)
flow_threshold = np.percentile(flow_acc[flow_acc > 1], 90)
slope_low = slope < np.percentile(slope[slope > 0], 25)  # Bottom 25% slope

hotspots_combined = (twi_smooth >= twi_threshold_90) & slope_low & (flow_acc >= flow_threshold)

print(f"  Moderate risk areas: {hotspots_moderate.sum():,} cells ({100*hotspots_moderate.mean():.2f}%)")
print(f"  High risk areas: {hotspots_high.sum():,} cells ({100*hotspots_high.mean():.2f}%)")
print(f"  Critical zones (combined): {hotspots_combined.sum():,} cells ({100*hotspots_combined.mean():.2f}%)\n")

# ============================================================================
# Save TWI and Hotspots as GeoTIFF
# ============================================================================
print("Saving outputs...")

def save_geotiff(filename, data):
    """Save array as GeoTIFF"""
    with rasterio.open(
        OUTPUT_DIR / filename, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)

save_geotiff(f"{VILLAGE_NAME}_TWI.tif", twi_smooth.astype(np.float32))
save_geotiff(f"{VILLAGE_NAME}_flow_accumulation.tif", flow_acc.astype(np.float32))
save_geotiff(f"{VILLAGE_NAME}_slope.tif", slope.astype(np.float32))
save_geotiff(f"{VILLAGE_NAME}_waterlogging_hotspots.tif", hotspots_high.astype(np.uint8))

print("  ✓ GeoTIFFs saved\n")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("Creating visualizations...\n")

fig, axes = plt.subplots(3, 3, figsize=(24, 24))

# Row 1: Inputs
# 1.1 DTM
im = axes[0, 0].imshow(dtm, cmap='terrain')
axes[0, 0].set_title('DTM (Refined)', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')
plt.colorbar(im, ax=axes[0, 0], label='Elevation (m)', fraction=0.046)

# 1.2 Slope
im = axes[0, 1].imshow(slope, cmap='YlOrRd', vmin=0, vmax=np.percentile(slope, 95))
axes[0, 1].set_title('Slope', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im, ax=axes[0, 1], label='Slope', fraction=0.046)

# 1.3 Flow Accumulation
im = axes[0, 2].imshow(np.log10(flow_acc + 1), cmap='Blues')
axes[0, 2].set_title('Flow Accumulation (log scale)', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')
plt.colorbar(im, ax=axes[0, 2], label='log10(cells)', fraction=0.046)

# Row 2: TWI and Hotspots
# 2.1 TWI
im = axes[1, 0].imshow(twi_smooth, cmap='RdYlBu_r', vmin=np.percentile(twi_smooth, 5), vmax=np.percentile(twi_smooth, 95))
axes[1, 0].set_title(f'Topographic Wetness Index (TWI)', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], label='TWI', fraction=0.046)

# 2.2 Moderate risk on orthophoto
if ortho is not None:
    axes[1, 1].imshow(ortho, alpha=0.6)
axes[1, 1].imshow(hotspots_moderate, cmap='YlOrRd', alpha=0.7, vmin=0, vmax=1)
axes[1, 1].set_title(f'Moderate Risk Zones (Top 10% TWI)\n{hotspots_moderate.sum():,} cells', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

# 2.3 High risk on orthophoto
if ortho is not None:
    axes[1, 2].imshow(ortho, alpha=0.6)
axes[1, 2].imshow(hotspots_high, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
axes[1, 2].set_title(f'High Risk Zones (Top 5% TWI)\n{hotspots_high.sum():,} cells', fontsize=14, fontweight='bold')
axes[1, 2].axis('off')

# Row 3: Combined analysis
# 3.1 Critical zones on orthophoto
if ortho is not None:
    axes[2, 0].imshow(ortho, alpha=0.6)
ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(dtm, vert_exag=2.0)
axes[2, 0].imshow(hs, cmap='gray', alpha=0.3)
axes[2, 0].imshow(hotspots_combined, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
axes[2, 0].set_title(f'Critical Waterlogging Zones\n(High TWI + Low Slope + High Flow)\n{hotspots_combined.sum():,} cells',
                     fontsize=14, fontweight='bold')
axes[2, 0].axis('off')

# 3.2 All risk levels combined
if ortho is not None:
    axes[2, 1].imshow(ortho, alpha=0.5)

# Create composite risk map
risk_map = np.zeros_like(twi_smooth)
risk_map[hotspots_moderate] = 1
risk_map[hotspots_high] = 2
risk_map[hotspots_combined] = 3

from matplotlib.colors import ListedColormap
colors = ['white', 'yellow', 'orange', 'red']
cmap_risk = ListedColormap(colors)

im = axes[2, 1].imshow(risk_map, cmap=cmap_risk, alpha=0.7, vmin=0, vmax=3)
axes[2, 1].set_title('Composite Risk Map\n(Yellow=Moderate, Orange=High, Red=Critical)', fontsize=14, fontweight='bold')
axes[2, 1].axis('off')

# 3.3 Statistics
stats_text = f"""WATERLOGGING ANALYSIS

TWI Statistics:
  Min:     {twi_smooth.min():8.2f}
  Max:     {twi_smooth.max():8.2f}
  Mean:    {twi_smooth.mean():8.2f}
  Std:     {twi_smooth.std():8.2f}

Risk Thresholds:
  90th %:  {twi_threshold_90:8.2f}
  95th %:  {twi_threshold_95:8.2f}

Risk Areas:
  Moderate: {hotspots_moderate.sum():7,} cells
            {100*hotspots_moderate.mean():6.2f}% coverage

  High:     {hotspots_high.sum():7,} cells
            {100*hotspots_high.mean():6.2f}% coverage

  Critical: {hotspots_combined.sum():7,} cells
            {100*hotspots_combined.mean():6.2f}% coverage

Total Area at Risk:
  {(hotspots_moderate.sum() * cell_size**2 / 10000):.2f} hectares
"""

axes[2, 2].text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=axes[2, 2].transAxes)
axes[2, 2].axis('off')

plt.suptitle(f'{VILLAGE_NAME} - Waterlogging Hotspot Analysis (TWI-Based)',
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'FINAL_waterlogging_hotspots.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Waterlogging analysis complete!")
print(f"\nOutputs saved:")
print(f"  • {VILLAGE_NAME}_TWI.tif")
print(f"  • {VILLAGE_NAME}_waterlogging_hotspots.tif")
print(f"  • {VILLAGE_NAME}_flow_accumulation.tif")
print(f"  • {VILLAGE_NAME}_slope.tif")
print(f"  • FINAL_waterlogging_hotspots.png")


# ==============================================================================

# Visualisation B — Drainage Network Overlay
# Overlays the optimised drainage network on the waterlogging risk map and orthophoto.

# ==============================================================================


"""
TerrainFlow - Drainage Network on Waterlogging Risk Map
Overlay drainage design on risk zones
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
import rasterio
import geopandas as gpd

OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"
BASE_PATH = Path("/content/drive/MyDrive/NAT_GEO_HACKATHON")

print("Creating Drainage Network on Risk Map...\n")

# ============================================================================
# Load all data
# ============================================================================

# DTM
dtm_file = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif"
with rasterio.open(dtm_file) as src:
    dtm = src.read(1)
    dtm[dtm < -1000] = np.nan
    transform = src.transform

# Waterlogging hotspots
hotspots_file = OUTPUT_DIR / f"{VILLAGE_NAME}_waterlogging_hotspots.tif"
hotspots = None
if hotspots_file.exists():
    with rasterio.open(hotspots_file) as src:
        hotspots = src.read(1)
    print("✓ Loaded waterlogging hotspots")

# TWI
twi_file = OUTPUT_DIR / f"{VILLAGE_NAME}_TWI.tif"
twi = None
if twi_file.exists():
    with rasterio.open(twi_file) as src:
        twi = src.read(1)
    print("✓ Loaded TWI")

# Drainage network
drainage_file = OUTPUT_DIR / f"{VILLAGE_NAME}_drainage_network.geojson"
gdf = None
if drainage_file.exists():
    gdf = gpd.read_file(drainage_file)
    print(f"✓ Loaded drainage network ({len(gdf)} paths)")

# Orthophoto (downsampled)
ortho_file = BASE_PATH / "PureGP_Ortho_Point_data" / "209183Pure_transparent_mosaic_group1.tif"
if not ortho_file.exists():
    ortho_file = BASE_PATH / "PureGP_Ortho_Point_data" / "RF_209183Pure_ORTHO.tif"

ortho = None
if ortho_file.exists():
    try:
        with rasterio.open(ortho_file) as src:
            r = src.read(1, out_shape=(src.height // 4, src.width // 4))
            g = src.read(2, out_shape=(src.height // 4, src.width // 4))
            b = src.read(3, out_shape=(src.height // 4, src.width // 4))
            ortho = np.dstack([r, g, b]).astype(np.float32)
            if ortho.max() > 1:
                ortho = ortho / 255.0
        print("✓ Loaded orthophoto\n")
    except:
        pass

# ============================================================================
# Create comprehensive drainage + risk visualization
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(24, 24))

ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(dtm, vert_exag=2.0)

# -----------------------------------------------------------------------------
# Plot 1: Drainage Network on Orthophoto
# -----------------------------------------------------------------------------
if ortho is not None:
    axes[0, 0].imshow(ortho, alpha=0.8)
else:
    axes[0, 0].imshow(hs, cmap='gray', alpha=0.5)

if gdf is not None:
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            xs, ys = geom.xy
            cols = [(x - transform.c) / transform.a for x in xs]
            rows = [(y - transform.f) / transform.e for y in ys]
            axes[0, 0].plot(cols, rows, 'cyan', linewidth=4, alpha=0.9,
                           label='Drainage Channel' if idx==0 else '')

            # Start point
            x_start, y_start = geom.coords[0]
            col_start = (x_start - transform.c) / transform.a
            row_start = (y_start - transform.f) / transform.e
            axes[0, 0].scatter(col_start, row_start, c='red', s=200, marker='o',
                             edgecolors='white', linewidth=3, zorder=10,
                             label='Collection Point' if idx==0 else '')

            # End point
            x_end, y_end = geom.coords[-1]
            col_end = (x_end - transform.c) / transform.a
            row_end = (y_end - transform.f) / transform.e
            axes[0, 0].scatter(col_end, row_end, c='lime', s=400, marker='*',
                             edgecolors='white', linewidth=3, zorder=11,
                             label='Discharge Point' if idx==0 else '')

axes[0, 0].set_title('Drainage Network on Orthophoto\n(Cyan=Channels, Red=Collection, Green=Discharge)',
                     fontsize=16, fontweight='bold')
axes[0, 0].axis('off')
if gdf is not None and len(gdf) > 0:
    axes[0, 0].legend(loc='upper right', fontsize=12, framealpha=0.9)

# -----------------------------------------------------------------------------
# Plot 2: Drainage Network on DTM
# -----------------------------------------------------------------------------
im = axes[0, 1].imshow(dtm, cmap='terrain', alpha=0.8)
axes[0, 1].imshow(hs, cmap='gray', alpha=0.3)

if gdf is not None:
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            xs, ys = geom.xy
            cols = [(x - transform.c) / transform.a for x in xs]
            rows = [(y - transform.f) / transform.e for y in ys]
            axes[0, 1].plot(cols, rows, 'blue', linewidth=4, alpha=0.9)

            # Add flow direction arrows
            if len(cols) > 10:
                for i in range(5, len(cols)-5, max(1, len(cols)//5)):
                    dx = cols[i+1] - cols[i]
                    dy = rows[i+1] - rows[i]
                    axes[0, 1].arrow(cols[i], rows[i], dx*3, dy*3,
                                   head_width=20, head_length=20,
                                   fc='blue', ec='blue', alpha=0.8)

axes[0, 1].set_title('Drainage Network on DTM\n(Blue arrows show flow direction)',
                     fontsize=16, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im, ax=axes[0, 1], label='Elevation (m)', fraction=0.046)

# -----------------------------------------------------------------------------
# Plot 3: Drainage Network on Waterlogging Risk Map
# -----------------------------------------------------------------------------
if twi is not None and hotspots is not None:
    # Create composite risk map
    twi_90 = np.percentile(twi[twi > 0], 90)
    twi_95 = np.percentile(twi[twi > 0], 95)

    moderate = twi >= twi_90
    high = twi >= twi_95

    risk_map = np.zeros_like(twi)
    risk_map[moderate] = 1
    risk_map[high] = 2

    # Custom colormap
    colors = ['white', 'yellow', 'orange']
    cmap_risk = ListedColormap(colors)

    im = axes[1, 0].imshow(risk_map, cmap=cmap_risk, alpha=0.8, vmin=0, vmax=2)
    axes[1, 0].imshow(hs, cmap='gray', alpha=0.2)

    # Overlay drainage network
    if gdf is not None:
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'LineString':
                xs, ys = geom.xy
                cols = [(x - transform.c) / transform.a for x in xs]
                rows = [(y - transform.f) / transform.e for y in ys]
                axes[1, 0].plot(cols, rows, 'blue', linewidth=5, alpha=0.9,
                               label='Proposed Drainage' if idx==0 else '')

                # Mark collection points
                x_start, y_start = geom.coords[0]
                col_start = (x_start - transform.c) / transform.a
                row_start = (y_start - transform.f) / transform.e
                axes[1, 0].scatter(col_start, row_start, c='darkred', s=250,
                                  marker='o', edgecolors='white', linewidth=4,
                                  zorder=10, label='Collection' if idx==0 else '')

    axes[1, 0].set_title('Drainage Network on Waterlogging Risk Map\n(Yellow=Moderate Risk, Orange=High Risk)',
                         fontsize=16, fontweight='bold')
    axes[1, 0].legend(loc='upper right', fontsize=12, framealpha=0.9)
else:
    axes[1, 0].text(0.5, 0.5, 'Waterlogging Risk Map\nNot Available',
                   ha='center', va='center', fontsize=16,
                   transform=axes[1, 0].transAxes)

axes[1, 0].axis('off')

# -----------------------------------------------------------------------------
# Plot 4: Integrated Solution View
# -----------------------------------------------------------------------------
if ortho is not None:
    axes[1, 1].imshow(ortho, alpha=0.5)

# Show risk zones
if twi is not None:
    axes[1, 1].imshow(risk_map, cmap=cmap_risk, alpha=0.5, vmin=0, vmax=2)

# Overlay drainage
if gdf is not None:
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            xs, ys = geom.xy
            cols = [(x - transform.c) / transform.a for x in xs]
            rows = [(y - transform.f) / transform.e for y in ys]

            # Thicker, more visible drainage lines
            axes[1, 1].plot(cols, rows, 'cyan', linewidth=6, alpha=1.0,
                           solid_capstyle='round', label='Drainage Channel' if idx==0 else '')

            # Collection points
            x_start, y_start = geom.coords[0]
            col_start = (x_start - transform.c) / transform.a
            row_start = (y_start - transform.f) / transform.e
            axes[1, 1].scatter(col_start, row_start, c='red', s=300,
                              marker='o', edgecolors='yellow', linewidth=4,
                              zorder=10, label='Water Collection' if idx==0 else '')

            # Discharge point
            x_end, y_end = geom.coords[-1]
            col_end = (x_end - transform.c) / transform.a
            row_end = (y_end - transform.f) / transform.e
            axes[1, 1].scatter(col_end, row_end, c='lime', s=500,
                              marker='*', edgecolors='yellow', linewidth=4,
                              zorder=11, label='Water Discharge' if idx==0 else '')

axes[1, 1].set_title('Integrated Solution: Drainage Design on Orthophoto + Risk Zones\n(Cyan channels drain water from red collection points to green discharge)',
                     fontsize=16, fontweight='bold')
axes[1, 1].axis('off')
if gdf is not None and len(gdf) > 0:
    axes[1, 1].legend(loc='upper right', fontsize=12, framealpha=0.9,
                     edgecolor='black', fancybox=True)

plt.suptitle(f'{VILLAGE_NAME} - Complete Drainage Solution with Risk Assessment',
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'FINAL_drainage_on_orthophoto_and_risk.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved: FINAL_drainage_on_orthophoto_and_risk.png")

# ============================================================================
# Summary Statistics
# ============================================================================
if gdf is not None:
    print(f"\nDrainage Network Summary:")
    print(f"  Total channels: {len(gdf)}")
    print(f"  Total length: {gdf.geometry.length.sum():.1f} m")
    print(f"  Average channel length: {gdf.geometry.length.mean():.1f} m")
    print(f"  Longest channel: {gdf.geometry.length.max():.1f} m")

    if twi is not None and hotspots is not None:
        print(f"\nWaterlogging Risk Coverage:")
        print(f"  Moderate risk cells: {moderate.sum():,}")
        print(f"  High risk cells: {high.sum():,}")
        print(f"  Total area at risk: {(moderate.sum() * 0.25) / 10000:.2f} hectares")


# ==============================================================================

# Visualisation C — Complete Pipeline Gallery
# Fast multi-panel summary: orthophoto, DTM, hillshade, waterlogging hotspots, TWI, drainage network.

# ==============================================================================


"""
TerrainFlow - Fast Final Visualization
Optimized for speed - loads existing outputs only
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
import rasterio
import geopandas as gpd

OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"
BASE_PATH = Path("/content/drive/MyDrive/NAT_GEO_HACKATHON")

print(f"TerrainFlow Final Visualization - {VILLAGE_NAME}\n")

# ============================================================================
# Load all data ONCE
# ============================================================================
print("Loading data...")

# DTMs
dtm_refined = None
dtm_csf = None
transform = None
ortho = None

dtm_refined_file = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif"
if dtm_refined_file.exists():
    with rasterio.open(dtm_refined_file) as src:
        dtm_refined = src.read(1)
        dtm_refined[dtm_refined < -1000] = np.nan
        transform = src.transform
    print("  ✓ Refined DTM")

dtm_csf_file = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_csf_UTM_median.tif"
if dtm_csf_file.exists():
    with rasterio.open(dtm_csf_file) as src:
        dtm_csf = src.read(1)
        dtm_csf[dtm_csf < -1000] = np.nan
    print("  ✓ CSF DTM")

# Waterlogging hotspots (if exists)
hotspots_file = OUTPUT_DIR / f"{VILLAGE_NAME}_waterlogging_hotspots.tif"
hotspots = None
if hotspots_file.exists():
    with rasterio.open(hotspots_file) as src:
        hotspots = src.read(1)
    print("  ✓ Waterlogging hotspots")

# TWI (if exists)
twi_file = OUTPUT_DIR / f"{VILLAGE_NAME}_TWI.tif"
twi = None
if twi_file.exists():
    with rasterio.open(twi_file) as src:
        twi = src.read(1)
    print("  ✓ TWI")

# Drainage network
drainage_file = OUTPUT_DIR / f"{VILLAGE_NAME}_drainage_network.geojson"
gdf = None
if drainage_file.exists():
    gdf = gpd.read_file(drainage_file)
    print(f"  ✓ Drainage network ({len(gdf)} paths)")

# Orthophoto (downsample for speed)
ortho_file = BASE_PATH / "PureGP_Ortho_Point_data" / "209183Pure_transparent_mosaic_group1.tif"
if not ortho_file.exists():
    ortho_file = BASE_PATH / "PureGP_Ortho_Point_data" / "RF_209183Pure_ORTHO.tif"

if ortho_file.exists():
    try:
        with rasterio.open(ortho_file) as src:
            # Downsample by factor of 4 for speed
            r = src.read(1, out_shape=(src.height // 4, src.width // 4))
            g = src.read(2, out_shape=(src.height // 4, src.width // 4))
            b = src.read(3, out_shape=(src.height // 4, src.width // 4))
            ortho = np.dstack([r, g, b]).astype(np.float32)
            if ortho.max() > 1:
                ortho = ortho / 255.0
        print("  ✓ Orthophoto (downsampled)")
    except:
        pass

print()

# ============================================================================
# FIGURE 1: Complete Pipeline Overview (Single Figure)
# ============================================================================
print("[1/2] Creating pipeline overview...")

fig, axes = plt.subplots(2, 3, figsize=(24, 16))

ls = LightSource(azdeg=315, altdeg=45)

# 1. Orthophoto
if ortho is not None:
    axes[0, 0].imshow(ortho)
    axes[0, 0].set_title('Orthophoto', fontsize=14, fontweight='bold')
else:
    axes[0, 0].text(0.5, 0.5, 'Orthophoto\nNot Available',
                   ha='center', va='center', fontsize=14, transform=axes[0, 0].transAxes)
axes[0, 0].axis('off')

# 2. DTM
if dtm_refined is not None:
    if ortho is not None:
        axes[0, 1].imshow(ortho, alpha=0.5)
    im = axes[0, 1].imshow(dtm_refined, cmap='terrain', alpha=0.7)
    axes[0, 1].set_title(f'Refined DTM\nRange: {np.nanmin(dtm_refined):.1f}-{np.nanmax(dtm_refined):.1f}m',
                        fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
axes[0, 1].axis('off')

# 3. Hillshade
if dtm_refined is not None:
    hs = ls.hillshade(dtm_refined, vert_exag=2.0)
    if ortho is not None:
        axes[0, 2].imshow(ortho, alpha=0.4)
    axes[0, 2].imshow(hs, cmap='gray', alpha=0.8)
    axes[0, 2].set_title('Terrain Hillshade', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

# 4. Waterlogging hotspots
if hotspots is not None and dtm_refined is not None:
    if ortho is not None:
        axes[1, 0].imshow(ortho, alpha=0.6)
    axes[1, 0].imshow(hotspots, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[1, 0].set_title(f'Waterlogging Hotspots\n{hotspots.sum():,} cells at risk',
                        fontsize=14, fontweight='bold')
elif dtm_refined is not None:
    # Fallback: show low areas
    low_threshold = np.nanpercentile(dtm_refined, 10)
    low_areas = dtm_refined <= low_threshold
    if ortho is not None:
        axes[1, 0].imshow(ortho, alpha=0.6)
    axes[1, 0].imshow(low_areas, cmap='Blues', alpha=0.7)
    axes[1, 0].set_title('Low-Lying Areas (Bottom 10%)', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

# 5. TWI or Slope
if twi is not None:
    im = axes[1, 1].imshow(twi, cmap='RdYlBu_r', vmin=np.percentile(twi, 5), vmax=np.percentile(twi, 95))
    axes[1, 1].set_title('Topographic Wetness Index', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
elif dtm_refined is not None:
    from scipy.ndimage import sobel
    dzdx = sobel(dtm_refined, axis=1) / 4.0
    dzdy = sobel(dtm_refined, axis=0) / 4.0
    slope = np.sqrt(dzdx**2 + dzdy**2)
    im = axes[1, 1].imshow(slope, cmap='YlOrRd', vmin=0, vmax=np.percentile(slope[slope>0], 95))
    axes[1, 1].set_title('Slope', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
axes[1, 1].axis('off')

# 6. Drainage network
if gdf is not None and dtm_refined is not None:
    if ortho is not None:
        axes[1, 2].imshow(ortho, alpha=0.6)
    axes[1, 2].imshow(hs, cmap='gray', alpha=0.4)

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            xs, ys = geom.xy
            cols = [(x - transform.c) / transform.a for x in xs]
            rows = [(y - transform.f) / transform.e for y in ys]
            axes[1, 2].plot(cols, rows, 'cyan', linewidth=3, alpha=0.9)

            if idx == 0:
                x_start, y_start = geom.coords[0]
                col_start = (x_start - transform.c) / transform.a
                row_start = (y_start - transform.f) / transform.e
                axes[1, 2].scatter(col_start, row_start, c='red', s=150, marker='o',
                                 edgecolors='white', linewidth=3, zorder=10)

    axes[1, 2].set_title(f'Drainage Network\n{len(gdf)} channels, {gdf.geometry.length.sum():.1f}m',
                        fontsize=14, fontweight='bold')
axes[1, 2].axis('off')

plt.suptitle(f'TerrainFlow Complete Pipeline - {VILLAGE_NAME}', fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'FINAL_complete_pipeline.png', dpi=300, bbox_inches='tight')
plt.show()
print("  ✓ Saved FINAL_complete_pipeline.png\n")

# ============================================================================
# FIGURE 2: Drainage Network Detail
# ============================================================================
print("[2/2] Creating drainage network detail...")

if gdf is not None and dtm_refined is not None:
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))

    # Background
    if ortho is not None:
        ax.imshow(ortho, alpha=0.7)
    ax.imshow(hs, cmap='gray', alpha=0.3)

    # Drainage paths
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            xs, ys = geom.xy
            cols = [(x - transform.c) / transform.a for x in xs]
            rows = [(y - transform.f) / transform.e for y in ys]
            ax.plot(cols, rows, 'cyan', linewidth=4, alpha=0.9, label='Drainage Channel' if idx==0 else '')

            # Start point
            x_start, y_start = geom.coords[0]
            col_start = (x_start - transform.c) / transform.a
            row_start = (y_start - transform.f) / transform.e
            ax.scatter(col_start, row_start, c='red', s=200, marker='o',
                      edgecolors='white', linewidth=3, zorder=10, label='Collection Point' if idx==0 else '')

            # End point
            x_end, y_end = geom.coords[-1]
            col_end = (x_end - transform.c) / transform.a
            row_end = (y_end - transform.f) / transform.e
            ax.scatter(col_end, row_end, c='lime', s=400, marker='*',
                      edgecolors='white', linewidth=3, zorder=11, label='Discharge Point' if idx==0 else '')

    ax.set_title(f'{VILLAGE_NAME} - Optimized Drainage Network Design\n{len(gdf)} Channels | Total Length: {gdf.geometry.length.sum():.1f}m',
                fontsize=16, fontweight='bold')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'FINAL_drainage_network_detail.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved FINAL_drainage_network_detail.png\n")

print("Complete! Generated:")
print("  • FINAL_complete_pipeline.png")
print("  • FINAL_drainage_network_detail.png")
