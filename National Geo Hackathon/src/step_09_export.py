"""
TerrainFlow — AI-Driven Hydrological Modelling & Drainage Design
Team ID: Nati-250330 | National Geo-AI Hackathon, Techfest IIT Bombay 2025-26
Theme 2: DTM & Drainage Network Design (SVAMITVA Scheme)

File: step_09_export.py

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

# Export Outputs
# Download the complete output folder as a ZIP, or selected files individually.

# ==============================================================================


# ── Verification ──────────────────────────────────────────────────────────
# Quick verification
print("\n" + "="*80)
print("PRE-SHUTDOWN CHECKLIST")
print("="*80)

required_files = [
    f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif",
    f"{VILLAGE_NAME}_DTM_refined_UTM_p05.tif",
    f"{VILLAGE_NAME}_TWI.tif",
    f"{VILLAGE_NAME}_waterlogging_hotspots.tif",
    f"{VILLAGE_NAME}_drainage_network.geojson",
    f"{VILLAGE_NAME}_COMPLETE_SUMMARY.json",
    f"{VILLAGE_NAME}_FINAL_REPORT.txt",
    "FINAL_complete_pipeline.png",
    "FINAL_waterlogging_hotspots.png",
    "FINAL_drainage_on_orthophoto_and_risk.png",
    "FINAL_comparison_table.png"
]

print("\nVerifying key deliverables:")
for f in required_files:
    exists = (OUTPUT_DIR / f).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")

print("\n" + "="*80)
print("DOWNLOAD INSTRUCTIONS")
print("="*80)
print(f"\nDownload entire folder: {OUTPUT_DIR}")
print("OR download individual files listed above")
print("\n✓ Ready for prototype submission!")

# ── Download ──────────────────────────────────────────────────────────────
"""
TerrainFlow - Download All Outputs to Local Machine
Manually download all generated files
"""

from pathlib import Path
import os
from google.colab import files

OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"

print("Preparing to download all TerrainFlow outputs...\n")

# ============================================================================
# Option 1: Download Everything as ZIP (RECOMMENDED)
# ============================================================================
print("="*80)
print("OPTION 1: Download Everything as ZIP (Recommended)")
print("="*80)

import shutil

zip_name = f"TerrainFlow_{VILLAGE_NAME}_Complete"
zip_path = f"/content/{zip_name}"

print(f"\nCreating ZIP archive: {zip_name}.zip")
shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)

print(f"Archive created: {zip_path}.zip")
print(f"Size: {os.path.getsize(zip_path + '.zip') / (1024**2):.1f} MB\n")

print("Downloading ZIP file...")
files.download(zip_path + '.zip')
print("✓ ZIP download started!\n")

# ============================================================================
# Option 2: Download Individual Files (if ZIP fails)
# ============================================================================
print("="*80)
print("OPTION 2: Download Individual Key Files")
print("="*80)
print("(Run this if ZIP download fails)\n")

# List of critical files to download
critical_files = [
    # Summary reports
    f"{VILLAGE_NAME}_COMPLETE_SUMMARY.json",
    f"{VILLAGE_NAME}_FINAL_REPORT.txt",

    # DTM outputs
    f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif",
    f"{VILLAGE_NAME}_DTM_refined_UTM_p05.tif",
    f"{VILLAGE_NAME}_DTM_csf_UTM_median.tif",

    # Hydrology outputs
    f"{VILLAGE_NAME}_TWI.tif",
    f"{VILLAGE_NAME}_waterlogging_hotspots.tif",
    f"{VILLAGE_NAME}_flow_accumulation.tif",
    f"{VILLAGE_NAME}_slope.tif",

    # Drainage network
    f"{VILLAGE_NAME}_drainage_network.geojson",

    # Visualizations
    "FINAL_complete_pipeline.png",
    "FINAL_waterlogging_hotspots.png",
    "FINAL_drainage_on_orthophoto_and_risk.png",
    "FINAL_drainage_network_detail.png",
    "FINAL_comparison_table.png",
]

print("Critical files to download individually:")
for i, filename in enumerate(critical_files, 1):
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024**2)
        print(f"  {i}. {filename} ({size_mb:.2f} MB)")
    else:
        print(f"  {i}. {filename} (NOT FOUND)")

print("\nTo download individual files, uncomment and run the code below:\n")

# ============================================================================
# Option 3: Selective Download (uncomment to use)
# ============================================================================
print("="*80)
print("OPTION 3: Selective Download Code (copy-paste if needed)")
print("="*80)

download_code = '''
# Download specific files individually
from google.colab import files
from pathlib import Path

OUTPUT_DIR = Path("/content/terrainflow_outputs")
VILLAGE_NAME = "RF_209183Pure"

# Download summary reports
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_COMPLETE_SUMMARY.json")
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_FINAL_REPORT.txt")

# Download DTMs
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_median.tif")
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_DTM_refined_UTM_p05.tif")

# Download waterlogging analysis
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_TWI.tif")
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_waterlogging_hotspots.tif")

# Download drainage network
files.download(OUTPUT_DIR / f"{VILLAGE_NAME}_drainage_network.geojson")

# Download visualizations
files.download(OUTPUT_DIR / "FINAL_complete_pipeline.png")
files.download(OUTPUT_DIR / "FINAL_waterlogging_hotspots.png")
files.download(OUTPUT_DIR / "FINAL_drainage_on_orthophoto_and_risk.png")
files.download(OUTPUT_DIR / "FINAL_drainage_network_detail.png")

print("✓ All critical files downloaded!")
'''

print(download_code)

# ============================================================================
# File Inventory
# ============================================================================
print("\n" + "="*80)
print("COMPLETE FILE INVENTORY")
print("="*80)

all_files = sorted(OUTPUT_DIR.glob("*"))
total_size = 0

print(f"\nTotal files in {OUTPUT_DIR}: {len(all_files)}\n")

# Group by type
geotiffs = []
geojsons = []
npz_files = []
models = []
visualizations = []
reports = []
other = []

for f in all_files:
    size_mb = f.stat().st_size / (1024**2)
    total_size += f.stat().st_size

    if f.suffix == '.tif':
        geotiffs.append((f.name, size_mb))
    elif f.suffix == '.geojson':
        geojsons.append((f.name, size_mb))
    elif f.suffix == '.npz':
        npz_files.append((f.name, size_mb))
    elif f.suffix == '.pth':
        models.append((f.name, size_mb))
    elif f.suffix == '.png':
        visualizations.append((f.name, size_mb))
    elif f.suffix in ['.json', '.txt']:
        reports.append((f.name, size_mb))
    else:
        other.append((f.name, size_mb))

print(f"📊 GeoTIFF Rasters ({len(geotiffs)}):")
for name, size in geotiffs:
    print(f"  • {name} ({size:.2f} MB)")

print(f"\n🗺️  GeoJSON Vectors ({len(geojsons)}):")
for name, size in geojsons:
    print(f"  • {name} ({size:.2f} MB)")

print(f"\n📦 NumPy Archives ({len(npz_files)}):")
for name, size in npz_files:
    print(f"  • {name} ({size:.2f} MB)")

print(f"\n🤖 ML Models ({len(models)}):")
for name, size in models:
    print(f"  • {name} ({size:.2f} MB)")

print(f"\n🎨 Visualizations ({len(visualizations)}):")
for name, size in visualizations:
    print(f"  • {name} ({size:.2f} MB)")

print(f"\n📄 Reports ({len(reports)}):")
for name, size in reports:
    print(f"  • {name} ({size:.2f} MB)")

if other:
    print(f"\n📁 Other Files ({len(other)}):")
    for name, size in other:
        print(f"  • {name} ({size:.2f} MB)")

print(f"\n{'='*80}")
print(f"TOTAL SIZE: {total_size / (1024**2):.2f} MB ({total_size / (1024**3):.2f} GB)")
print(f"{'='*80}\n")

# ============================================================================
# Google Drive Backup (Alternative)
# ============================================================================
print("="*80)
print("OPTION 4: Copy to Google Drive (Permanent Backup)")
print("="*80)

drive_backup_code = '''
# Copy everything to Google Drive for permanent storage
import shutil
from pathlib import Path

OUTPUT_DIR = Path("/content/terrainflow_outputs")
DRIVE_BACKUP = Path("/content/drive/MyDrive/TerrainFlow_Backup")

# Create backup directory
DRIVE_BACKUP.mkdir(parents=True, exist_ok=True)

# Copy all files
print("Copying files to Google Drive...")
for file in OUTPUT_DIR.glob("*"):
    if file.is_file():
        shutil.copy2(file, DRIVE_BACKUP / file.name)
        print(f"  ✓ {file.name}")

print(f"\\n✓ All files backed up to: {DRIVE_BACKUP}")
print("Files will persist even after runtime disconnects!")
'''

print("\nTo backup to Google Drive, run:\n")
print(drive_backup_code)

print("\n" + "="*80)
print("DOWNLOAD COMPLETE!")
print("="*80)
print("\n✅ Your ZIP file should be downloading now")
print("✅ All files inventoried and ready")
print("✅ Alternative download methods provided above")
print("\n🎉 TerrainFlow prototype complete - ready for presentation!")
