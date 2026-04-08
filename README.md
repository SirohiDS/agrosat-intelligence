# AgroSat Intelligence Dashboard

> Satellite-powered agritech analytics for Punjab wheat — combining Sentinel-2 NDVI, XGBoost yield modelling, stress detection, price forecasting, and credit risk scoring across 22 districts (Rabi 2022–23).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Sentinel-2](https://img.shields.io/badge/Data-Sentinel--2_ESA-00b4d8)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌐 Live Dashboard

**[https://sirohids.github.io/agrosat-intelligence/](https://sirohids.github.io/agrosat-intelligence/)**

---

## What It Does

Processes Sentinel-2 satellite imagery via Google Earth Engine for 22 Punjab districts, computes NDVI z-scores to detect crop stress 6 weeks before visible damage, and outputs a full agritech intelligence report used for agricultural credit risk by banks and insurance companies.

| NDVI Z-Score | Stress Level | Credit Action |
|---|---|---|
| > −1.5 | Normal | Standard KCC limit |
| −1.5 to −2.0 | Moderate | Monitor closely |
| −2.0 to −3.0 | Critical | Reduce limit 20–40% |
| < −3.0 | Severe | Pre-position collections |

## Dashboard Modules

- **Overview** — KPI summary: yield MAPE, districts covered, high-risk alerts, early warning lead time
- **Yield Model** — XGBoost predicted vs actual kg/ha, NDVI–yield scatter (R²=0.87), SHAP feature importance
- **Stress Detection** — NDVI time-series, district heatmap, z-score ranking, stress vs yield loss correlation
- **Price Forecast** — SARIMA + LightGBM ensemble for wheat/onion/tomato (MAPE 9.1%), MSP floor tracking
- **Credit Risk** — Risk tier distribution, SatSource report sample, experience–skills bridge cards
- **Data Pipeline** — Source inventory, full tech stack, pipeline latency breakdown (8hrs satellite→alert)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python agrosat_pipeline.py
```

The pipeline writes `district_data.json` — the dashboard reads this file. Commit updated JSON and push to refresh the live dashboard.

## Data Sources

| Source | Layer | Access | Cadence |
|---|---|---|---|
| Sentinel-2 SR (ESA) | Satellite | GEE API | 5 days |
| Sentinel-1 SAR (ESA) | Satellite | GEE API | 6 days |
| NASA POWER | Weather | REST API | Daily |
| AGMARKNET | Market prices | Scraper | Daily |
| SoilGrids 250m | Soil properties | REST API | Seasonal |
| FAOSTAT / data.gov | Yield history | CSV | Annual |

## Tech Stack

**Data layer** — `earthengine-api` · `rasterio` · `geopandas` · `rasterstats` · `pandas`

**Modelling layer** — `xgboost` · `lightgbm` · `statsmodels` · `shap` · `scikit-learn` · `tensorflow`

**Delivery layer** — `streamlit` · `plotly` · `folium` · `reportlab` · `pytest` · GitHub Actions

## Key Results

- **Yield MAPE: 8.2%** — beats SatSure benchmark of ~10%
- **Early warning: 6 weeks** before visible crop damage
- **Loan TAT: 8 hours** vs 3 weeks manual field inspection
- **60% field visits eliminated** — ₹400 per visit at scale
- **3–4pp NPA reduction** target for agricultural lenders
- **83M credit-invisible farmers** now addressable via satellite

## Districts Covered

Ludhiana · Patiala · Amritsar · Mansa · Faridkot · Bathinda · Sangrur · Firozpur · Hoshiarpur · Jalandhar · Kapurthala · Gurdaspur · Pathankot · Rupnagar · SAS Nagar · Fatehgarh Sahib · Moga · Muktsar · Barnala · Tarn Taran · SBS Nagar · Fazilka

---

*Built by [Arjun Sirohi](https://github.com/SirohiDS) · SatSure-inspired platform · Punjab wheat · Rabi 2022–23*
