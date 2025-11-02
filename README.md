# Causal-GeoSim
Causal-GeoSim: Evaluating Directional Robustness and Spatial Risk Awareness of Large Language Models in Climate–Agriculture Systems
Causal-GeoSim
Directional robustness + spatial risk auditing for LLMs in climate–agriculture systems.
Causal-GeoSim is a benchmark and reproducible evaluation pipeline for testing whether large language models (LLMs) actually understand cause → effect in real corn production settings — and whether they stay consistent across space.
We focus on U.S. Corn Belt counties and ask, for each county-year:

“Did heat + drought drive yield loss?” (causal)
vs
“Did low yield cause the heat + drought?” (anti-causal)

Models must answer in a forced-choice Answer: A/B format.
We then score:


CAI (Causal Advantage Index):
How often does the model get the direction right?
(P(correct causal) – P(correct anti-causal))


Geo-CAI:
Spatial consistency. We weight CAI by Moran’s I to detect weird spatial pockets of failure.


GRS (Geo-Risk Score):
How bad are those failures in places that actually matter? We penalize directional confusion more in yield-critical (stress-vulnerable) counties.



tl;dr: It’s not just “can the model explain drought.” It’s “will this model hallucinate backwards causality in a high-risk county.”


Table of Contents


Highlights


Pipeline Overview


Repository Structure


Quick Start


Inputs & Required Keys


What the Pipeline Produces


Metrics


Multi-Model Benchmarking


Notes on Reproducibility


Citing Causal-GeoSim


License


Contact



Highlights


Real data, not toy prompts.
We fuse:


USDA NASS county-level corn yield (bushels/acre)


ERA5-Land climate reanalysis (temp & precip)


US Census county geometries




Automatic prompt builder.
For each county-year we generate:


A causal question: “Did climate stress cause low yield?”


A matched anti-causal question: “Did low yield cause climate stress?”


Both demand:
Answer with A or B:
A = Yes ...
B = No ...


Spatial auditability.
We don’t just check if the model is locally “smart.”
We map where it breaks.


One-cell runner.
The notebook is structured as a single executable pipeline:


Download / load data


Build prompts


Query models (OpenAI or OpenRouter, with offline mock fallback)


Score CAI, Geo-CAI, GRS


Export paper-ready figures and tables




Paper-ready exports.
The pipeline writes:


prompts.csv


results.csv


summary_table.csv


summary_table.json


table1.tex (LaTeX-ready row for your results table)


CAI_county_map.png (choropleth for the paper)


a .zip with all figure assets for submission





Pipeline Overview
The notebook walks through these numbered stages (mirrors the comments # 1., # 2., … in the code):


USDA Yield Fetch
Uses USDA NASS QuickStats API to fetch county-level corn yield
(unit_desc = "BU / ACRE", statisticcat_desc = "YIELD", agg_level_desc = "COUNTY").
We keep year, state, county, FIPS, and yield.


Climate Fetch / Load
Loads ERA5-Land reanalysis (surface temperature, total precip) for a defined window (e.g. June 2020).
The code includes an automatic CDSAPI-based download attempt (cdsapi + xarray) and a fallback: you can manually provide a file like era5_midwest_2020_06.nc.


Spatial Join


Downloads U.S. Census county shapefiles (cb_2020_us_county_500k.zip) on first run.


Computes county centroids and samples ERA5 values at the nearest grid cell.


Produces per-county climate features like mean 2m temperature (°C) and cumulative precip (mm).




Merge Climate + Yield
Joins yield and climate by county FIPS for a target year (e.g., 2020).


Prompt Generation
For each sampled county:


Builds a causal prompt (“climate stress → yield drop?”)


Builds a paired anti-causal prompt (“yield drop → climate stress?”)


Forces the model to answer A or B and briefly justify.


Assigns a per-county risk_w weight (e.g. higher weight if yield is already below median), which later feeds into GRS.


These go into prompts.csv.


LLM Inference
Supports two modes out of the box:


OpenAI client (openai / gpt-4o, etc.)


OpenRouter client
(openai/gpt-4o-mini, anthropic/claude-3.5-sonnet,
meta-llama/llama-3.1-70b-instruct, mistralai/mistral-large-latest, etc.)


If no API key is provided, the pipeline falls back to a deterministic mock model that answers realistically and still emits Answer: A / Answer: B.
This means you can run end-to-end without any paid API access.
Output from this stage is results.csv / results_single_model.csv.


Basic Metrics (CAI)
We parse each model’s text and extract the letter after Answer:.
We then compute:


causal accuracy  = fraction of causal prompts answered with A


anti accuracy    = fraction of anti-causal prompts answered with B


CAI          = causal_acc − anti_acc




Geo-Risk Score (GRS)
For each county:


CAI_county = mean(causal_correct) − mean(anti_correct)


risk_term = risk_w * (1 - CAI_county)


GRS = mean(risk_term) over all sampled counties


Intuition: if the model is confused and that county is “risk-weighted,” GRS goes up.


Geo-CAI + Moran’s I


We merge county scores back into the county shapefile (GeoPandas).


We build a Queen contiguity graph using libpysal.Queen.


We compute Moran’s I for spatial autocorrelation in CAI_county.


Geo-CAI = mean(CAI_county) × Moran’s I


This tells you if the model is spatially stable or if it behaves erratically across neighboring counties.


Visualization
We generate a choropleth:
CAI_county_map.png
Counties are colored by CAI_county, borders drawn in black, and the figure is saved at 300 dpi for publication.


Exports
We dump:


CSV / JSON summaries


LaTeX tables (table1.tex, table1_multimodel.tex)


Figures


Optional .zip bundle of figures for submission





Repository Structure
A suggested layout for your repo:
causal-geosim/
├─ README.md               <- this file
├─ causal_geosim.ipynb     <- "Causal-GeoSim One-Cell Pipeline" notebook (OpenRouter edition)
├─ data/
│   ├─ era5_midwest_2020_06.nc        <- ERA5-Land climate slice (you provide or auto-download)
│   └─ counties_shp/                  <- US Census county shapefile (auto-downloaded)
├─ outputs/
│   ├─ prompts.csv
│   ├─ results_single_model.csv
│   ├─ summary_table.csv
│   ├─ summary_table.json
│   ├─ table1.tex
│   ├─ CAI_county_map.png
│   └─ figures_bundle.zip             <- optional zip export for paper
└─ scripts/
    └─ utils.py (optional future refactor of helper functions)

Right now, most logic lives in the notebook. You can later refactor to scripts/ if you want a pure-Python CLI.

Quick Start
1. Clone & enter
git clone https://github.com/YOUR_USERNAME/causal-geosim.git
cd causal-geosim

2. Create environment (recommended)
You can use conda, venv, or just run the built-in ensure([...]) installer block in the notebook.
Core Python deps used in the pipeline:
python>=3.9
pandas
numpy
requests
xarray
netCDF4
cdsapi
geopandas
shapely
pyproj
rtree
libpysal
esda
matplotlib
openai              # if calling OpenAI models directly

For OpenRouter-based inference:
requests           # already listed above

For Colab convenience (downloads, zips):
google.colab       # auto-imported if running in Colab

3. Open the notebook
Open causal_geosim.ipynb in:


Jupyter Lab / VSCode


or Google Colab (recommended for first run)


4. Run all cells top-to-bottom
The notebook is designed as a “one-cell runner” style pipeline:


it will attempt to install any missing packages,


download shapefiles,


load or request ERA5 climate data,


build prompts,


query models,


compute metrics,


and export paper figures/tables.



Inputs & Required Keys
The notebook will politely prompt you for secrets using hidden input (so they don’t end up in your bash history).
You’ll see helper functions like get_env_secret(...) doing this.
You may be asked for:


USDA NASS QuickStats API key
Used to pull county-level corn yield.
(Free registration at USDA QuickStats.)


ERA5-Land data
The script tries to download a climate slice via cdsapi.
If that fails (e.g., you don’t have Copernicus credentials in your environment), it will ask you to manually provide a file such as
era5_midwest_2020_06.nc.
This NetCDF file is expected to include daily/hourly t2m (2m air temp) and tp (total precipitation).


OpenAI API key OR OpenRouter API key (optional)


If provided, we will really call models (GPT-4o, Claude-3.5-Sonnet, Llama-3.1-70B, Mistral-Large, etc.).


If not provided, the notebook falls back to a deterministic mock model that simulates A/B answers in a realistic pattern.


Either way, the rest of the pipeline (metrics/maps/tables) still runs.




No key? You still get:


prompts


mock results


CAI, Geo-CAI, GRS


spatial plots


LaTeX tables


So you can demo the full research story offline.

What the Pipeline Produces
After a successful run, you should see (and these are automatically saved to disk):


prompts.csv
All causal & anti-causal question pairs by county.


results_single_model.csv (or results.csv)
Raw model answers:


model name actually used


county FIPS / name / state / year


causal_ans, anti_ans (A or B)


causal_correct, anti_correct (booleans)




summary_table.csv and summary_table.json
Aggregated metrics for that model:


causal accuracy


anti accuracy


CAI


Geo-CAI


Moran’s I


GRS




table1.tex
A one-row LaTeX table fragment for dropping into the paper (“Table 1: Model comparison”).


CAI_county_map.png
Choropleth of CAI_county at the county level.


(Optional) table1_multimodel.csv / table1_multimodel.tex
If you run the multi-model loop.


(Optional) figures_bundle.zip
A zipped bundle of all publication figures, ready to attach to submissions.



Metrics
We compute several interpretable, publication-facing metrics right in the notebook:
CAI (Causal Advantage Index)
CAI = P(model answers A to causal_prompt)
    - P(model answers B to anti_causal_prompt)

Higher CAI → better directional robustness.
CAI_county
Per-county version of CAI.
GRS (Geo-Risk Score)
We weight directional confusion more in “bad” counties.
Sketch:
CAI_county = mean(causal_correct) - mean(anti_correct)

risk_term = risk_w * (1 - CAI_county)

GRS = average over counties of risk_term

Interpretation:
If the model is confused and the county is flagged as high-risk (e.g. below-median yield), it hurts more.
Moran’s I
Spatial autocorrelation of CAI_county computed with libpysal.Queen.
If Moran’s I is strongly negative, that means county-level causal behavior is spatially noisy or even adversarially inconsistent.
Geo-CAI
Geo-CAI = mean(CAI_county) × Moran’s I

This couples how right you are with how spatially coherent you are.

Multi-Model Benchmarking
The notebook includes an (initially commented) block that loops over multiple model names, e.g.:
model_list = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-large-latest"
]

For each:


Run inference on the same county prompt set.


Compute CAI, Geo-CAI, Moran’s I, and GRS.


Append a row to table1_multimodel.csv (and table1_multimodel.tex for LaTeX).


This is what feeds the paper’s claim like

“GPT-4 and Llama-3.1 achieve the most consistent causal reasoning (Geo-CAI ≈ 0, GRS ≈ 1.48), while Mistral-Large exhibits higher geo-risk (GRS ≈ 1.62) and spatial inconsistency (Moran’s I ≈ –0.10).”

Those numbers come directly from these exported summary tables.

Notes on Reproducibility


Random seed
The notebook fixes RNG_SEED = 42 (also seeds numpy and random).
This controls county subsampling for prompt generation.


County sampling
By default we subsample a small number of counties (e.g. 10) to keep evaluation cheap.
You can scale n_samples up in build_prompts(...) for more statistical confidence.


Environment dump
The code is already structured to be “camera-ready”:


Installs missing pip dependencies at runtime (ensure([...]))


Saves all exported assets with stable filenames


Optionally zips figures for submission




Offline mode
If no API keys are provided, we still produce a valid results_* dataframe using a mock inference function that:


Forces the Answer: X format (so parsing is stable)


Gets direction correct ~80% of the time
(this is useful for debugging the metric math and plots even with zero credits / zero internet)





Citing Causal-GeoSim
If you use this repository, notebooks, metrics, or result tables in academic work, please cite:

Causal-GeoSim: Evaluating Directional Robustness and Spatial Risk Awareness of Large Language Models in Climate–Agriculture Systems
Youla Yang
Luddy School of Informatics, Computing, and Engineering
Indiana University Bloomington
yangyoul@iu.edu
2025

(Full BibTeX coming soon.)

License
TBD.
Recommended: choose a permissive license (MIT, Apache-2.0) if you want others to reuse the benchmark definition, or a research-only license if you want to restrict commercial decision-support systems.
Add LICENSE to the repo root and mention it here.

Contact


Maintainer: Youla Yang


Affiliation: Luddy School of Informatics, Computing, and Engineering, Indiana University Bloomington


Email: yangyoul@iu.edu


If you extend this benchmark (new crops, new stressors, new regions, or multi-modal inputs like NDVI), feel free to open an Issue or PR.
