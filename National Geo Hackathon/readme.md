# TerrainFlow

**AI-Driven Hydrological Modelling & Drainage Design from Drone Point Clouds**

> Submitted to the **National Geo-AI Hackathon** — Techfest IIT Bombay 2025-26  
> In collaboration with the Ministry of Panchayati Raj (SVAMITVA Scheme)  
> **Team:** The Error Guy | **Team ID:** Nati-250330 | **Theme 2:** DTM & Drainage Network Design

---

## The Problem

Under the SVAMITVA Scheme, millions of rural *abadi* (inhabited) areas across India are being mapped using high-resolution drone surveys. These maps are excellent for property demarcation — but they lack **engineering-grade terrain intelligence**. There is no reliable basis for computing slope, flow direction, or drainage planning from raw drone outputs.

The deeper technical failure: standard ground-filtering algorithms (e.g., Cloth Simulation Filtering / CSF) **collapse** in dense village environments. They over-smooth the terrain, flattening embankments, lanes, and depressions into a near-flat surface. The result is a hydrologically useless model where meaningful water-flow cannot be computed.

**CSF baseline on RF_209183Pure:** Elevation range = 0.98 m (essentially flat).  
**TerrainFlow output:** Elevation range = 29.68 m — true topography recovered.

---

## Pipeline Overview

```
LAS/LAZ Point Cloud (drone survey)
    │
    ▼  Step 1 ── Data Loading, CRS Validation & Attribute Inspection
    ▼  Step 2 ── CSF Cloth-Simulation Pseudo-Labeling (geometric prior)
    ▼  Step 3 ── Residual MLP Ground Refinement (weak supervision, no manual labels)
    ▼  Step 4 ── DTM Generation (0.5 m UTM raster, IDW gap-fill, multi-stat)
    ▼  Step 5 ── Hydrological Analysis (D8 flow direction, flow accumulation, TWI)
    ▼  Step 6 ── Drainage Network Design (least-cost path routing → GeoJSON)
    │
    └─ Outputs: GeoTIFF + GeoJSON (QGIS / ArcGIS / government GIS stacks ready)
```

---

## Methodology

### Step 1 — Data Loading & CRS Validation

The pipeline opens LAS/LAZ files with `laspy`, inspects available attributes (classification, intensity, return number, RGB, extra dimensions), and validates the coordinate reference system via a spatial-extent heuristic.

**Why CRS matters first:** Slope, flow direction, and flow accumulation are physically invalid in geographic degrees. Any village whose coordinates are in EPSG:4326 is reprojected to **UTM Zone 44N (EPSG:32644)** before any terrain math is performed. This is enforced as a hard constraint, not an optional step.

---

### Step 2 — CSF Pseudo-Labeling

Cloth Simulation Filtering (CSF) drapes a simulated cloth over the inverted point cloud to separate ground from non-ground points.

| Parameter | Value | Design Rationale |
|-----------|-------|-----------------|
| `cloth_resolution` | 0.5 m | Resolves narrow lanes and courtyards |
| `rigidness` | 3 | Soft cloth follows embankments; avoids over-flattening |
| `class_threshold` | 0.5 m | Standard separation threshold |
| `iterations` | 500 | Sufficient for convergence |

CSF outputs are deliberately treated as **weak pseudo-labels**, not final ground truth. The next step uses them only as a noisy geometric prior for ML supervision.

---

### Step 3 — AI Ground Refinement (Residual MLP)

A lightweight **Dense Pointwise Refinement Net** — a point-wise Residual MLP — is trained on CSF pseudo-labels using weak supervision (no manual annotation of millions of points required).

**Architecture:** Point-wise MLP with residual connections. Input is XYZ coordinates (optionally with RGB or intensity). Output is a per-point binary classification: ground (class 2) vs non-ground (class 1).

**Why point-wise?** The network learns local elevation and texture patterns that CSF misses in densely built-up areas — correcting the classical algorithm's bias without requiring an explicit neighbourhood aggregation graph (unlike RandLA-Net or PointNet++).

**Training configuration:**
- 30 epochs, Adam optimiser (lr=1e-3, weight decay=1e-4)
- AMP mixed-precision on CUDA (A100 optimised; batch size configurable for smaller GPUs)
- Early stopping (patience=8)
- 30% subsample for training; full-cloud inference in 10M-point chunks

**Result on prototype village:** CSF identified 780,362 ground points (3.32% grid coverage). The refined model recovered **4,957,151 ground points (24.68% coverage)** — a 6.4× improvement.

---

### Step 4 — DTM Generation

Refined ground points are gridded into a **0.5 m resolution GeoTIFF** in UTM space using fast array binning.

- Per-cell statistics: `median` (robust against outliers) and `p05` (conservative bare-earth)
- Gap filling: Inverse Distance Weighting (IDW, k=8 neighbours, power=2)
- Optional Gaussian smoothing (σ=1 pixel) to reduce micro-roughness for hydraulics
- NoData value: −9999 (standard for GIS tools)

Multiple rasters are generated per village to allow downstream selection of the most appropriate surface for hydrology vs visualisation.

---

### Steps 5 & 6 — Hydrological Analysis & Drainage Design

#### Hydrology (Step 5)

1. **Depression filling** — priority-flood algorithm initialised from raster edges; ensures continuous flow paths with no spurious sinks
2. **D8 flow direction** — each cell drains to its steepest downslope neighbour among 8 directions
3. **Flow accumulation** — upstream contributing area per cell (sorted by elevation, high→low)
4. **Slope** — computed via Sobel operator (rise/run in metric UTM space)
5. **Topographic Wetness Index (TWI):** `TWI = ln(SCA / tan β)` where SCA = specific catchment area, β = local slope angle
6. **Waterlogging hotspot delineation** — cells exceeding a TWI threshold flagged as persistent water-accumulation zones

**Prototype result:** 2.53 hectares flagged as high waterlogging risk.

RichDEM (D∞ routing) is used when available for higher accuracy; a custom D8 implementation serves as the fallback.

#### Drainage Network Design (Step 6)

An optimised drainage network is routed from identified low-lying collection nodes to a single lowest feasible discharge outlet using **least-cost path routing** on a cost surface derived from normalised elevation + slope. This enforces both constructibility and gravity compliance.

**Prototype result:** 10 channels, 5,008 m total length, 1 discharge outlet. Exported as GeoJSON.

---

## Results

| Metric | CSF Baseline | TerrainFlow | Improvement |
|--------|-------------|-------------|-------------|
| Ground points classified | 780,362 | 4,957,151 | **6.4×** |
| Grid coverage | 3.32% | 24.68% | **7.4×** |
| Elevation range | 0.98 m | 29.68 m | **30×** |
| Elevation std dev | 0.19 m | 2.41 m | **12.7×** |
| Waterlogging detection | Not viable | 2.53 ha identified | Enabled |
| Drainage design | Not viable | 10 channels, 5,008 m | Enabled |

---

## Outputs

All outputs are saved to `terrainflow_outputs/` with a `<VILLAGE_NAME>` prefix:

| File | Description |
|------|-------------|
| `*_DTM_refined_UTM_median.tif` | Primary hydrology-ready terrain model (GeoTIFF) |
| `*_DTM_refined_UTM_p05.tif` | Conservative bare-earth estimate |
| `*_DTM_csf_UTM_median.tif` | CSF baseline for comparison |
| `*_TWI.tif` | Topographic Wetness Index raster |
| `*_waterlogging_hotspots.tif` | Binary waterlogging risk raster |
| `*_flow_accumulation.tif` | Upstream contributing area raster |
| `*_slope.tif` | Slope raster |
| `*_drainage_network.geojson` | Optimised drainage channels (vector) |
| `*_COMPLETE_SUMMARY.json` | Full metrics and processing log |
| `*_FINAL_REPORT.txt` | Human-readable summary report |

All rasters use EPSG:32644 (UTM Zone 44N) and are directly loadable in QGIS, ArcGIS, or any government GIS stack.

---

## Installation

```bash
pip install laspy[lazrs] numpy pandas matplotlib seaborn scipy scikit-learn
pip install cloth-simulation-filter richdem rasterio pyproj geopandas scikit-image
pip install torch  # GPU runtime recommended (A100 / T4 on Colab)
```

---

## Repository Structure

```
terrainflow/
    step_00_install.py               # Install all dependencies (run once)
    step_01_data_loading.py          # Drive mount, LAS/LAZ discovery, CRS validation
    step_02_csf_pseudolabels.py      # CSF cloth-simulation pseudo-labeling
    step_03_ml_refinement.py         # Residual MLP ground refinement
    step_04_dtm_generation.py        # 0.5 m UTM raster generation, IDW gap-fill
    step_05_06_hydrology_drainage.py # D8 flow, TWI, waterlogging, drainage routing
    step_07_visualise_waterlogging.py# Waterlogging maps, drainage overlay, pipeline gallery
    step_08_metrics_and_report.py    # Comparison table and JSON/text summary report
    step_09_export.py                # Output verification and ZIP download
    deploy_batch.py                  # Batch loop to scale across all villages
    README.md
```

## Usage

Designed to run in Google Colab with a GPU runtime (A100 recommended; T4 also supported).

1. Upload your LAS/LAZ files to Google Drive under `NAT_GEO_HACKATHON/`
2. Run `step_00_install.py` once to install all dependencies
3. Execute steps 01 through 09 in order
4. Run `step_09_export.py` to download a ZIP of all outputs

### Scaling to multiple villages

Change only two lines at the top of steps 02 through 06:

```python
PROTOTYPE_FILE = Path("/content/drive/MyDrive/NAT_GEO_HACKATHON/<FOLDER>/<FILENAME>.laz")
VILLAGE_NAME   = "<FILENAME_WITHOUT_EXTENSION>"
```

For automated batch processing across all villages, use `deploy_batch.py` directly.

Expected processing time on A100: ~4-6 h for a 200 MB village, ~15-20 h for 1.5 GB.  
For T4 (16 GB), reduce `train_batch_size` from 524,288 to 131,072 in `step_03_ml_refinement.py`.

---

## Tech Stack

- **Point cloud I/O:** `laspy`
- **Ground filtering:** `cloth-simulation-filter` (CSF)
- **ML refinement:** PyTorch (Residual MLP, AMP mixed precision)
- **Terrain rasterisation:** NumPy array binning + `rasterio`
- **CRS handling:** `pyproj`
- **Hydrology:** `richdem` (D∞) with custom D8 fallback; `scipy.ndimage`
- **Drainage routing:** `scikit-image` (least-cost path)
- **Vector export:** `geopandas` + GeoJSON
- **Visualisation:** `matplotlib` with `LightSource` hillshading

---

## Limitations & Future Work

- The Residual MLP is a point-wise model; incorporating local neighbourhood context (e.g., KNN or graph-based layers) would improve classification in highly cluttered areas
- The D8 flow model can produce unrealistic parallel channels on flat terrain; D∞ (available via RichDEM) is preferred when installed
- Drainage channel widths and depths are not yet estimated — a hydraulic design step using Manning's equation is planned
- Batch processing across all 5 villages is sequential; parallelisation across cloud instances would reduce total wall time significantly

---

## Acknowledgements

Data provided under the SVAMITVA Scheme by the Ministry of Panchayati Raj, Government of India, via the National Geo-AI Hackathon organised by Techfest IIT Bombay 2025-26 in partnership with the National Informatics Centre (NIC).
