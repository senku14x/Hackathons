"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_05_06_hydrology_drainage.py

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

# Steps 5 & 6 — Hydrological Analysis & Drainage Network Design
# Hydrology (Step 5)
# 1. Depression filling (priority-flood algorithm)
# 2. D8 flow direction
# 3. Flow accumulation
# 4. Slope (Sobel operator)
# 5. Topographic Wetness Index (TWI = ln(SCA / tan β))
# 6. Waterlogging hotspot delineation (TWI threshold)
# Drainage Design (Step 6)
# Least-cost path routing on a cost surface derived from normalised elevation + slope.
# Routes channels from each low-lying collection node to a single lowest feasible discharge outlet.
# Output exported as GeoJSON (GIS-ready).

# ==============================================================================


"""
TerrainFlow - Steps 5 & 6: Hydrological Analysis & Drainage Network
FIXED VERSION (UTM-safe, nodata-safe)
"""

import subprocess
import sys
import os
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import rasterio
from scipy import ndimage
from scipy.ndimage import gaussian_filter, generic_filter
import geopandas as gpd
from shapely.geometry import LineString
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# ============================================================================
# Optional RichDEM
# ============================================================================
RICHDEM_AVAILABLE = False
try:
    import richdem as rd
    RICHDEM_AVAILABLE = True
except ImportError:
    pass

print(f"[Info] Using hydrology backend: {'RichDEM (D∞)' if RICHDEM_AVAILABLE else 'Custom D8'}")

# ============================================================================
# Custom Hydrology (fallback)
# ============================================================================

class HydrologicalAnalysis:

    @staticmethod
    def fill_depressions_simple(dem):
        """
        Priority-flood style depression filling
        FIX: Proper NaN handling
        """
        dem = dem.copy().astype(np.float64)
        dem[dem < -1000] = np.nan   # FIX: sanitize nodata

        filled = dem.copy()
        h, w = dem.shape
        visited = np.zeros((h, w), dtype=bool)

        from heapq import heappush, heappop
        pq = []

        # initialize edges
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

    @staticmethod
    def flow_direction_d8(dem):
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
                    slope = (z - nz) / dist
                    if slope > best:
                        best = slope
                        best_d = d
                fd[i, j] = best_d
        return fd

    @staticmethod
    def flow_accumulation_d8(flow_dir, dem):
        h, w = dem.shape
        acc = np.ones((h, w), dtype=np.float64)
        dirs = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

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

    @staticmethod
    def calculate_slope(dem, cell_size):
        dzdx = ndimage.sobel(dem, axis=1) / (8 * cell_size)
        dzdy = ndimage.sobel(dem, axis=0) / (8 * cell_size)
        return np.sqrt(dzdx**2 + dzdy**2)

    @staticmethod
    def calculate_twi(flow_accum, slope, cell_size):
        sca = flow_accum * cell_size**2
        slope_safe = np.maximum(slope, 0.001)
        twi = np.log(sca / slope_safe)
        return np.nan_to_num(twi, nan=0.0)

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"
DTM_FILE = OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif"

# ============================================================================
# Load DTM
# ============================================================================
with rasterio.open(DTM_FILE) as src:
    dtm = src.read(1).astype(np.float64)
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs

print(f"[DTM] Loaded {dtm.shape}, CRS={crs}")

# FIX: UTM cell size is already meters
cell_size = 0.5 * (abs(transform.a) + abs(transform.e))
print(f"[DTM] Cell size = {cell_size:.2f} m (UTM-correct)")

# ============================================================================
# Depression Filling
# ============================================================================
hydro = HydrologicalAnalysis()

if RICHDEM_AVAILABLE:
    rd_dem = rd.rdarray(dtm, no_data=-9999)
    rd_dem.geotransform = [
        transform.c, transform.a, 0,
        transform.f, 0, transform.e
    ]
    dtm_filled = np.array(rd.FillDepressions(rd_dem, epsilon=True))
else:
    dtm_filled = hydro.fill_depressions_simple(dtm)

print("[Hydrology] Depressions filled")

# ============================================================================
# Flow Direction & Accumulation
# ============================================================================
if RICHDEM_AVAILABLE:
    flow_dir = rd.FlowDirTarboton(rd_dem)
    flow_acc = np.array(rd.FlowAccumFromProps(rd_dem, flow_dir))
else:
    flow_dir = hydro.flow_direction_d8(dtm_filled)
    flow_acc = hydro.flow_accumulation_d8(flow_dir, dtm_filled)

print("[Hydrology] Flow accumulation computed")

# ============================================================================
# Slope & TWI
# ============================================================================
slope = hydro.calculate_slope(dtm_filled, cell_size)
twi = hydro.calculate_twi(flow_acc, slope, cell_size)

print(f"[Slope] mean={np.nanmean(slope):.4f}")
print(f"[TWI] mean={np.nanmean(twi):.2f}")

# ============================================================================
# Drainage Design
# ============================================================================
elev_norm = (dtm - dtm.min()) / (dtm.max() - dtm.min())
slope_norm = (slope - slope.min()) / (slope.max() - slope.min())

cost = 0.4 * (1 - elev_norm) + 0.6 * (1 - slope_norm)

# Drain points
acc_flat = flow_acc.flatten()
idxs = np.argsort(acc_flat)[::-1]

drain_pts = []
for idx in idxs:
    r, c = np.unravel_index(idx, dtm.shape)
    if all(np.hypot(r-pr, c-pc) > 50 for pr, pc in drain_pts):
        drain_pts.append((r, c))
    if len(drain_pts) >= 10:
        break

# Discharge (lowest edge)
edges = np.r_[dtm[0,:], dtm[-1,:], dtm[:,0], dtm[:,-1]]
dis_idx = np.nanargmin(edges)
h, w = dtm.shape
if dis_idx < w:
    dr, dc = 0, dis_idx
elif dis_idx < 2*w:
    dr, dc = h-1, dis_idx-w
elif dis_idx < 2*w+h:
    dr, dc = dis_idx-2*w, 0
else:
    dr, dc = dis_idx-(2*w+h), w-1

# Paths
paths = []
lines = []
for r, c in drain_pts:
    inds, _ = route_through_array(cost, (r, c), (dr, dc),
                                  fully_connected=True, geometric=True)
    paths.append(inds)
    coords = [(transform.c + cc*transform.a,
               transform.f + rr*transform.e) for rr, cc in inds]
    lines.append(LineString(coords))

gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
gdf.to_file(OUTPUT_DIR / f"{VILLAGE_NAME}_drainage_network.geojson", driver="GeoJSON")

print("[Done] Drainage network exported")

