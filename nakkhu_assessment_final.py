"""
=======================================================================
Nakkhu Basin — ML Building Flood Risk Assessment Pipeline
=======================================================================
Project : Machine Learning Based Building Detection for Flood Loss
          Assessment (Nakkhu River, Sep 27 2024 Flash Flood)
Authors : Anupa Ranabhat, Bimala Sapkota, Manjil Theeng
          Pulchowk Campus, IOE, Tribhuvan University

-----------------------------------------------------------------------
LOGIC / METHODOLOGY (for documentation):
-----------------------------------------------------------------------

STEP 1 — Building Inventory from YOLO Detections
  - Each tile folder contains a *_geo_locations.csv file
  - These CSVs were produced by running YOLOv8-OBB inference on
    640x640 satellite image tiles
  - Each row = one detected building with:
      easting_utm_x  : UTM X coordinate of building centroid (EPSG:32645)
      northing_utm_y : UTM Y coordinate of building centroid (EPSG:32645)
      confidence     : YOLO detection confidence score [0-1]
  - Buildings with confidence < threshold are discarded (noise removal)

STEP 2 — HAND Value Extraction (The "Z-Axis" Check)
  - HAND = Height Above Nearest Drainage
  - HAND raster encodes how many meters each pixel sits ABOVE the
    nearest river/drainage channel
  - HAND = 0   → pixel IS the drainage channel (river itself)
  - HAND = 2   → pixel sits 2m above the nearest channel
  - HAND = 15  → pixel is on a hillside, far from flooding
  - For each detected building, we "poke" the HAND raster at the
    building's UTM coordinate and extract its HAND value
  - This converts a 2D satellite detection into a 3D flood risk score

STEP 3 — Risk Classification
  Based on HAND value thresholds (standard in flood literature):
    HAND = 0m        → On Drainage  (river channel / false positive)
    HAND 0–2m        → High Risk    (inundation depth ~1m+)
    HAND 2–5m        → Moderate Risk(inundation depth 0.3-1m)
    HAND > 5m        → Low Risk     (minimal inundation)
    HAND = nodata    → No Data

STEP 4 — Economic Loss Estimation
  Using Nepal-specific depth-damage model:
    - Average building footprint : 60 m² (peri-urban Lalitpur)
    - Replacement cost           : NPR 3,000/m² (Nepal CBS 2023)
    - Damage fractions by risk:
        High Risk     → 30% structural damage
        Moderate Risk → 12% structural damage
        Low Risk      →  3% structural damage (seepage/surface water)
    - Loss per building = Area × Cost × Damage_Fraction
    - Total loss = sum across all detected buildings
  Reference: PDNA Nepal guidelines, NDRRMA loss assessment framework

STEP 5 — Outputs
  1. Building_Risk_Assessment.csv — per-building risk table
  2. risk_chart.png              — pie chart + economic bar chart
  3. Nakkhu_Risk_Map.html        — interactive Folium map (satellite)

-----------------------------------------------------------------------
INPUTS:
  --geo-dir   : Folder containing 1200 tile subfolders, each with
                a *_geo_locations.csv file
  --hand      : HAND raster GeoTIFF (EPSG:32645, 12.5m resolution)
  --outdir    : Output directory
  --conf      : Minimum YOLO confidence (default 0.35)

USAGE:
  pip install rasterio folium matplotlib pandas pyproj tqdm

  python nakkhu_assessment_final.py \
      --geo-dir "/content/drive/MyDrive/Nakkhu Building Analytics" \
      --hand    "/content/drive/MyDrive/hand_map.tif" \
      --outdir  "/content/results"
=======================================================================
"""

import os
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import MarkerCluster
import rasterio
from pyproj import Transformer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

CONF_THRESHOLD  = 0.35       # discard YOLO detections below this
NODATA_VAL      = 32767.0    # HAND raster nodata sentinel value
BLDG_AREA_M2    = 60         # avg building footprint (m²), Lalitpur peri-urban
COST_NPR_M2     = 3000       # replacement cost NPR/m² (Nepal CBS 2023)
NPR_TO_USD      = 135        # exchange rate
SRC_EPSG        = 32645      # your tiles CRS

# Risk thresholds and damage fractions
# (label, max_hand_value, map_color, damage_fraction)
RISK_LEVELS = [
    ("On Drainage",   0.0,  "#8e44ad", 0.80),   # HAND=0: on river channel
    ("High Risk",     2.0,  "#e74c3c", 0.30),   # 0-2m above drainage
    ("Moderate Risk", 5.0,  "#f39c12", 0.12),   # 2-5m above drainage
    ("Low Risk",      9999, "#27ae60", 0.03),   # >5m above drainage
]
COLOR_MAP = {r[0]: r[2] for r in RISK_LEVELS}
COLOR_MAP["No Data"] = "#95a5a6"
DISPLAY_ORDER = ["On Drainage", "High Risk", "Moderate Risk",
                 "Low Risk", "No Data"]


# ═══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def classify_hand(h: float) -> tuple:
    """
    Classify a HAND value into a risk category.

    Returns (risk_label, hex_color, damage_fraction)

    Logic:
      - NaN or nodata  → No Data
      - h == 0         → On Drainage (river channel cell)
      - h <= 2         → High Risk
      - h <= 5         → Moderate Risk
      - h >  5         → Low Risk
    """
    if h is None or np.isnan(h) or h >= NODATA_VAL:
        return "No Data", COLOR_MAP["No Data"], 0.0
    if h == 0.0:
        return RISK_LEVELS[0][0], RISK_LEVELS[0][2], RISK_LEVELS[0][3]
    for label, max_h, color, frac in RISK_LEVELS[1:]:
        if h <= max_h:
            return label, color, frac
    return "Low Risk", COLOR_MAP["Low Risk"], 0.03


def compute_loss(damage_frac: float) -> tuple:
    """
    Compute economic loss for one building.

    Formula:
      Loss (NPR) = Building_Area(m²) × Cost(NPR/m²) × Damage_Fraction
      Loss (USD) = Loss(NPR) / Exchange_Rate

    Returns (loss_usd, loss_npr)
    """
    loss_npr = BLDG_AREA_M2 * COST_NPR_M2 * damage_frac
    loss_usd = loss_npr / NPR_TO_USD
    return round(loss_usd, 2), round(loss_npr, 0)


def load_all_detections(geo_dir: str, conf_thresh: float) -> pd.DataFrame:
    """
    Walk all tile subfolders and load every *_geo_locations.csv.
    Apply confidence filter. Return merged DataFrame.
    """
    files = glob.glob(
        os.path.join(geo_dir, "**", "*geo_location*.csv"),
        recursive=True
    )
    if not files:
        raise FileNotFoundError(
            f"No *_geo_locations.csv files found under:\n  {geo_dir}\n"
            f"Check folder structure: geo_dir/tile_name/tile_name_geo_locations.csv"
        )

    frames = []
    for f in sorted(files):
        df = pd.read_csv(f)
        df["source_tile"] = Path(f).parts[-2]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]

    # Flexible column detection
    x_col = next(c for c in df.columns if "easting"  in c)
    y_col = next(c for c in df.columns if "northing" in c)

    # Confidence filter
    raw = len(df)
    if "confidence" in df.columns:
        df = df[df["confidence"] >= conf_thresh].reset_index(drop=True)

    return df, x_col, y_col, raw


def sample_hand_raster(hand_src, hand_data: np.ndarray,
                       utm_x: float, utm_y: float) -> float:
    """
    Sample HAND raster at a single UTM coordinate.

    Steps:
      1. Check point is within raster bounds
      2. Convert UTM → raster row/col via affine transform
      3. Read pixel value
      4. Return NaN if nodata or out of bounds
    """
    try:
        b = hand_src.bounds
        if not (b.left <= utm_x <= b.right and b.bottom <= utm_y <= b.top):
            return float("nan")
        row, col = rasterio.transform.rowcol(hand_src.transform, utm_x, utm_y)
        if 0 <= row < hand_src.height and 0 <= col < hand_src.width:
            val = float(hand_data[row, col])
            if hand_src.nodata and val == float(hand_src.nodata):
                return float("nan")
            return val
    except Exception:
        pass
    return float("nan")


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def save_chart(out_df: pd.DataFrame, outdir: Path):
    total     = len(out_df)
    total_usd = out_df["Loss_USD"].sum()
    counts    = out_df["Risk_Level"].value_counts()
    lbls      = [r for r in DISPLAY_ORDER if r in counts]
    sizes     = [counts[r] for r in lbls]
    clrs      = [COLOR_MAP[r] for r in lbls]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Nakkhu Basin — ML Building Flood Risk Assessment\n"
        "Sep 27, 2024 Flash Flood | Lalitpur, Nepal",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Pie chart
    wedges, _, autotexts = axes[0].pie(
        sizes, colors=clrs, startangle=140, pctdistance=0.72,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100))})",
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold")
    axes[0].set_title("Risk Distribution", fontsize=12, pad=12)
    axes[0].legend(
        handles=[mpatches.Patch(color=COLOR_MAP[r],
                 label=f"{r}  ({counts.get(r,0):,} bldgs)")
                 for r in lbls],
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        ncol=2, fontsize=9
    )

    # Bar chart — economic loss
    loss_by = out_df.groupby("Risk_Level")["Loss_USD"].sum()\
                    .reindex(lbls, fill_value=0)
    axes[1].bar(lbls, loss_by.values / 1e3, color=clrs,
                edgecolor="white", linewidth=1.5, width=0.5)
    axes[1].set_title("Economic Loss by Category", fontsize=12, pad=12)
    axes[1].set_ylabel("Estimated Loss (USD Thousand)", fontsize=10)
    for i, v in enumerate(loss_by.values):
        axes[1].text(
            i, v/1e3 + 0.3,
            f"${v/1e3:.1f}K\nNPR {v*NPR_TO_USD/1e3:.0f}K",
            ha="center", fontsize=8, fontweight="bold"
        )
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].set_facecolor("#f8f9fa")

    plt.tight_layout()
    path = outdir / "risk_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_map(out_df: pd.DataFrame, outdir: Path):
    dm = out_df.dropna(subset=["Latitude", "Longitude"])

    # Satellite as default layer
    m = folium.Map(
        location=[dm["Latitude"].mean(), dm["Longitude"].mean()],
        zoom_start=15,
        tiles=None
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
        overlay=False, control=True, show=True
    ).add_to(m)
    folium.TileLayer(
        "CartoDB positron", name="Street Map",
        overlay=False, control=True, show=False
    ).add_to(m)

    clusters = {
        r: MarkerCluster(name=f"● {r}").add_to(m)
        for r in DISPLAY_ORDER
    }

    for _, row in dm.iterrows():
        r   = row["Risk_Level"]
        col = COLOR_MAP.get(r, "#95a5a6")
        hv  = f"{row['HAND_Value_m']:.2f}m" \
              if pd.notna(row["HAND_Value_m"]) else "N/A"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6, color="white", weight=1,
            fill=True, fill_color=col, fill_opacity=0.9,
            popup=folium.Popup(
                f"<div style='font-family:Arial;font-size:12px'>"
                f"<b>Building ID:</b> {row['Building_ID']}<br>"
                f"<b>Risk Level:</b> "
                f"<span style='color:{col}'><b>{r}</b></span><br>"
                f"<b>HAND Value:</b> {hv}<br>"
                f"<b>Confidence:</b> {row['Confidence']:.1%}<br>"
                f"<b>Loss:</b> ${row['Loss_USD']:,.0f} "
                f"/ NPR {row['Loss_NPR']:,.0f}<br>"
                f"<b>Tile:</b> {row['Tile']}"
                f"</div>",
                max_width=230
            ),
            tooltip=f"{r} | HAND:{hv}"
        ).add_to(clusters.get(r, clusters["No Data"]))

    # Legend
    m.get_root().html.add_child(folium.Element(f"""
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:14px;border-radius:8px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.3);
                font-family:Arial;font-size:12px;line-height:1.8'>
      <b>HAND Flood Risk — Nakkhu Basin</b><br>
      <span style='color:#8e44ad;font-size:16px'>●</span>
        On Drainage (HAND=0m) — river channel<br>
      <span style='color:#e74c3c;font-size:16px'>●</span>
        High Risk (0–2m) — 30% structural damage<br>
      <span style='color:#f39c12;font-size:16px'>●</span>
        Moderate Risk (2–5m) — 12% structural damage<br>
      <span style='color:#27ae60;font-size:16px'>●</span>
        Low Risk (>5m) — 3% structural damage<br>
      <span style='color:#95a5a6;font-size:16px'>●</span>
        No Data<br>
      <hr style='margin:6px 0'>
      <small>Model: NPR {COST_NPR_M2}/m² × {BLDG_AREA_M2}m²<br>
      HAND Source: ALOS 12.5m DEM | EPSG:{SRC_EPSG}</small>
    </div>"""))

    folium.LayerControl(collapsed=False).add_to(m)
    path = outdir / "Nakkhu_Risk_Map.html"
    m.save(str(path))
    return path


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run(geo_dir: str, hand_path: str, outdir: str, conf_thresh: float):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  Nakkhu Basin — ML Flood Risk Assessment Pipeline")
    print("="*65)

    # ── STEP 1: Load detections ───────────────────────────────────────
    print(f"\n[STEP 1] Loading YOLO detections from:\n  {geo_dir}")
    df, x_col, y_col, raw_count = load_all_detections(geo_dir, conf_thresh)
    print(f"  Raw detections         : {raw_count:,}")
    print(f"  After conf ≥ {conf_thresh}     : {len(df):,} buildings retained")

    # ── STEP 2: Sample HAND raster ────────────────────────────────────
    print(f"\n[STEP 2] Sampling HAND raster at building centroids...")
    print(f"  HAND file: {hand_path}")
    hand_src  = rasterio.open(hand_path)
    hand_data = hand_src.read(1)
    hand_epsg = hand_src.crs.to_epsg()
    print(f"  HAND CRS  : EPSG:{hand_epsg}")
    print(f"  HAND res  : {hand_src.res[0]}m")
    print(f"  HAND shape: {hand_data.shape}")

    # UTM → WGS84 converter for Folium map
    to_wgs84 = Transformer.from_crs(
        f"EPSG:{SRC_EPSG}", "EPSG:4326", always_xy=True
    )

    hand_vals, lats, lons = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Sampling"):
        ux = float(row[x_col])
        uy = float(row[y_col])
        hand_vals.append(sample_hand_raster(hand_src, hand_data, ux, uy))
        lo, la = to_wgs84.transform(ux, uy)
        lons.append(round(lo, 7))
        lats.append(round(la, 7))

    hand_src.close()
    df["hand_value_m"] = hand_vals
    df["latitude"]     = lats
    df["longitude"]    = lons

    # ── STEP 3 & 4: Classify risk + compute loss ──────────────────────
    print(f"\n[STEP 3+4] Classifying risk levels and computing losses...")
    risks, losses_usd, losses_npr = [], [], []
    for h in df["hand_value_m"]:
        label, _, frac = classify_hand(h)
        l_usd, l_npr   = compute_loss(frac)
        risks.append(label)
        losses_usd.append(l_usd)
        losses_npr.append(l_npr)

    df["risk_level"] = risks
    df["loss_usd"]   = losses_usd
    df["loss_npr"]   = losses_npr

    # ── STEP 5: Build output DataFrame ───────────────────────────────
    out_df = pd.DataFrame({
        "Building_ID"  : range(1, len(df) + 1),
        "Tile"         : df["source_tile"],
        "Latitude"     : df["latitude"],
        "Longitude"    : df["longitude"],
        "Easting_UTM"  : df[x_col].round(2),
        "Northing_UTM" : df[y_col].round(2),
        "Confidence"   : df["confidence"].round(4)
                         if "confidence" in df.columns else 1.0,
        "HAND_Value_m" : pd.Series(hand_vals).round(3),
        "Risk_Level"   : risks,
        "Loss_USD"     : losses_usd,
        "Loss_NPR"     : losses_npr,
    })

    # Save CSV
    csv_path = outdir / "Building_Risk_Assessment.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV    : {csv_path}")

    # Save chart
    chart_path = save_chart(out_df, outdir)
    print(f"  ✓ Chart  : {chart_path}")

    # Save map
    map_path = save_map(out_df, outdir)
    print(f"  ✓ Map    : {map_path}")

    # ── Final summary ─────────────────────────────────────────────────
    total      = len(out_df)
    total_usd  = out_df["Loss_USD"].sum()
    total_npr  = out_df["Loss_NPR"].sum()
    counts     = out_df["Risk_Level"].value_counts()
    impacted   = sum(counts.get(r, 0)
                     for r in ["On Drainage","High Risk","Moderate Risk"])

    summary = f"""
{'='*65}
  NAKKHU BASIN — FINAL ASSESSMENT RESULTS
  Sep 27, 2024 Flash Flood | Lalitpur, Nepal
{'='*65}
  METHOD
    Building inventory  : YOLOv8-OBB detection ({total:,} buildings)
    Confidence filter   : ≥ {conf_thresh} ({raw_count - total:,} discarded)
    Flood risk source   : HAND raster (ALOS 12.5m DEM)
    Damage model        : NPR {COST_NPR_M2}/m² × {BLDG_AREA_M2}m²
                          (depth-damage fractions, NDRRMA framework)
{'─'*65}
  RESULTS
    Total buildings detected    : {total:,}
    Buildings at risk (≤5m)     : {impacted:,} ({impacted/total*100:.1f}%)

    On Drainage  (HAND=0m)      : {counts.get('On Drainage',0):>6,}
    High Risk    (HAND 0–2m)    : {counts.get('High Risk',0):>6,}
    Moderate Risk(HAND 2–5m)    : {counts.get('Moderate Risk',0):>6,}
    Low Risk     (HAND >5m)     : {counts.get('Low Risk',0):>6,}
    No Data                     : {counts.get('No Data',0):>6,}
{'─'*65}
  ECONOMIC LOSS ESTIMATE
    Total loss (USD)            : ${total_usd:>12,.0f}
    Total loss (NPR)            : NPR {total_npr:>10,.0f}
    Average per building        : ${total_usd/max(total,1):>12,.0f}
{'='*65}
  OUTPUT FILES
    {csv_path}
    {chart_path}
    {map_path}
{'='*65}"""

    print(summary)

    # Save summary
    (outdir / "summary_report.txt").write_text(summary)
    print(f"  ✓ Report : {outdir / 'summary_report.txt'}\n")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Nakkhu Basin ML Flood Risk Assessment"
    )
    p.add_argument("--geo-dir", required=True,
                   help="Parent folder containing tile subfolders with "
                        "*_geo_locations.csv files")
    p.add_argument("--hand",    required=True,
                   help="HAND raster GeoTIFF (hand_map.tif)")
    p.add_argument("--outdir",  default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Min YOLO confidence threshold (default: 0.35)")
    args = p.parse_args()

    run(args.geo_dir, args.hand, args.outdir, args.conf)
