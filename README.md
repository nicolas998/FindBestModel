# FindBestModel

Find which of the 46 [MARRMoT](https://github.com/wknoben/MARRMoT) conceptual
rainfall-runoff model structures best reproduces the daily streamflow of a USGS
watershed.

Give it a USGS gauge number and a period. It downloads the observed flow and the
forcing data, runs a Latin-hypercube random search over the parameters of every
model in parallel, and saves the performance of every realization. A Jupyter
notebook then ranks the models, plots the best ones against the observations and
exports the selected parameter set as a starting point for a full calibration.

## How it works

```
USGS gauge ──► find_best_model.py ──► performance/<gauge>/*.parquet ──► compare_models.ipynb
                 │                                                          │
                 ├─ streamflow       (USGS NWIS)                             ├─ rank by any metric
                 ├─ precipitation    (NOAA AORC basin average, or NASA POWER)├─ constraints (e.g. |PBIAS| < 15)
                 ├─ ET + temperature (NASA POWER)                            ├─ hydrographs, FDC, Q-Q plots
                 └─ N random realizations × 46 models                        └─ best_model_<metric>.json
```

Every realization is scored on NSE, KGE (and its r, alpha, beta parts), PBIAS,
correlation, mean annual peak-flow difference, log-NSE and several
`pdi_*` indexes. The first year(s) are a warm-up period and are not scored.

## Installation

Python 3.10+ (tested with 3.12).

```bash
git clone https://github.com/nicolas998/FindBestModel.git
cd FindBestModel
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

All data sources are public, so you don't need any accounts or API keys.

## Example: USGS gauge 02249500

### 1. Run the model search

```bash
python find_best_model.py --gauge 02249500 --n 200 --start-year 2015 --end-year 2025 --n-procs 8 --skip-snow
```

This:

1. downloads daily streamflow for 2014–2025 (2014 is the warm-up year) to
   `data/usgs_02249500_daily.parquet`;
2. downloads the basin boundary, computes basin-averaged AORC rainfall, adds
   NASA POWER ET, and saves them to `data/usgs_02249500_forcing.parquet`.
   Products already in that file are reused on later runs;
3. runs 200 parameter sets for each of the 34 models that don't need
   temperature, on 8 processes;
4. writes the results to `performance/02249500/`:

```
performance/02249500/
├── model_01_Collie_River_Basin_1.parquet   # one row per realization: parameters + metrics
├── model_02_Wetland_model.parquet
├── ...
├── summary.parquet                         # best scores of every model
└── run_info.json                           # settings used, models removed and why
```

If you run it with no arguments (`python find_best_model.py`), it asks for
each setting.

### 2. Take a quick look at the summary

```python
import pandas as pd

summary = pd.read_parquet("performance/02249500/summary.parquet")
print(summary.sort_values("best_KGE", ascending=False).head())
```

### 3. Compare the models in the notebook

Open `compare_models.ipynb`, edit the **Settings** cell and run all cells:

```python
GAUGE = "02249500"
METRIC = "KGE"                                  # NSE, KGE, log_NSE, pdi_ave, pdi_low, PBIAS, ...
TOP_N = 10                                      # models shown in the ranking
CONSTRAINTS = {"PBIAS": (-15, 15), "NSE": (0, None)}  # (min, max); None = no bound
SELECTED_MODEL = None                           # None = the top-ranked model
```

The notebook shows:

- a ranking of the models by `METRIC`, keeping only realizations that meet `CONSTRAINTS`;
- the spread of the metric within each model (box plots), to tell robust models
  from lucky ones;
- a table of each model's best value on each key metric;
- the best realizations re-run and plotted against the observations
  (hydrograph, flow-duration curve, Q-Q plot);
- side-by-side comparison of up to 5 models, and an interactive (plotly) view of
  the best model for each metric.

The selected model, its parameters and its metrics are saved to
`performance/02249500/best_model_KGE.json`.

## Designing a composite metric: `cpm_app.py`

A Streamlit app for building a composite performance metric (CPM) that
scores both the timing of a simulation and its flow distribution:

```bash
streamlit run cpm_app.py
```

In the sidebar, choose a gauge, a period and a model (CRB3, model 11, by
default), and optionally narrow the parameter ranges. **Download data and
run** fetches the data (same sources and `data/` cache as
`find_best_model.py`) and runs N random parameter sets (100 by default),
keeping every simulated series.

Like KGE, the CPM is one minus a Euclidean distance from the ideal point:

```
CPM = 1 - sqrt( sum_i (w_i / max(w) * (x_i - 1)) ** 2 )
```

Its components x_i are NSE (capped to 0–1), the mean and standard-deviation
ratios, the sim/obs ratios of the 5, 10, 25, 75, 90 and 95th percentiles, and
the Pearson and Spearman correlations. Each weight w_i (0–10, step 0.5) has its
own slider. The weights are relative, so equal weights on Pearson r, mean ratio
and deviation ratio alone give exactly KGE.

You can edit up to 5 configurations side by side. Each one has an identifier
computed from its weights (`CPM-XXXXXX`: the same weights always give the
same ID) and its own color. Each configuration picks its best realization,
which the app shows over the observed hydrograph and the top-ranked members
of the ensemble. Below the hydrograph are a table of every component of that
best run, its relative error through time, its flow PDF minus the observed
PDF, and its flow duration curve and Q-Q plot against the observations
(each with a log/linear switch). **Save to library** writes the configuration to
`cpm_configs/<ID>.json`, with the run settings and the best run's components
and parameters. Saved files can be loaded again from the library or uploaded.

## Multi-objective calibration: `nsga_app.py`

A Streamlit app that calibrates one model with NSGA-II against one to three
CPMs:

```bash
streamlit run nsga_app.py
```

In the sidebar, choose a gauge, the scored years, a spin-up in days (the
simulation starts that many days before January 1st of the first year), the
precipitation source, the potential-ET method and a model, and optionally
narrow its parameter search ranges. Then pick one to three CPMs: the presets,
any configuration saved with `cpm_app.py` in `cpm_configs/`, or JSON files
uploaded in the app (the files `cpm_app.py` saves or downloads). The NSGA-II
settings are the calibrations per generation (population size), the number
of generations, the crossover probability and distribution index (SBX), the
mutation rate (per parameter; 1 / number of parameters by default) and
distribution index (polynomial mutation), the seed and the number of
processes.

NSGA-II searches in the unit cube of the model's free parameters, so
thresholds defined as fractions (e.g. field capacity of maximum storage)
always stay valid.

With **Dynamic search ranges** (on by default), every few generations a
search range widens on any side where the population's Pareto front crowds
against it: by default, when at least 25% of the front lies within 5% of a
bound, that bound moves out by 50% of the range. It never moves past the
parameter's **min/max allowed** in the parameter table. By default maximum
storages may grow to 4000 mm (`STORE_LIMIT` in `model_parameter_space.py`),
and every other parameter only up to its default range, because rates,
fractions, exponents, temperatures and days of the year have physical or
numerical limits. Raise a limit in the table to let a parameter go further.

The results show:

- the objective space: every simulation and the Pareto front. With three
  CPMs, one panel per pair of CPMs, each with the front of that pair in its
  own color, plus the front over all three (also in a 3-D view);
- the hydrograph, flow duration curve and Q-Q plot of the members of the
  fronts you select, over the observations, with one member highlighted on
  demand;
- the distribution of every parameter on each selected front, over its
  search range (front members piled against a bound suggest widening it),
  and the search range changes of a dynamic run;
- the best CPM per generation, and the table of front members with their
  parameters.

Every run is saved in `nsga_calib/<date>_usgs<gauge>_m<model>/`:

| File | Contents |
|---|---|
| `settings.json` | gauge, period, spin-up, forcing, model, objectives and their weights, NSGA-II settings, the initial and final parameter ranges, their allowed limits and every range change |
| `evaluations.csv` | every simulation: generation, parameters, CPM of each objective, every CPM component, KGE, NSE, error, and one `front_*` flag per front |
| `history.csv` | best CPM so far per objective after each generation |
| `pareto_fronts.csv` | the simulations on any front |
| `front_series.parquet` | observed flow and the simulated flow of every front member |

`evaluations.csv` is written after every generation, so a run that is
stopped (changing a control while it runs stops it) keeps what it finished;
loading it from **Saved calibrations** computes its fronts. A run can also be
opened directly with `?run=<folder>` in the app's URL.

To move runs between machines (e.g. to and from the app on a web server),
the results of a run have three buttons:

- **Download run (.zip)**: every file of the run. Uploading that zip under
  **Saved calibrations** unpacks it into `nsga_calib/` and opens it. Only the
  five run files are read from the zip, under their own names, and the
  folder is named from the run's settings; the same run already in
  `nsga_calib/` is reused instead of copied, and a zip that would unpack to
  more than 500 MB is refused.
- **Download settings (.json)**: the run's `settings.json`. Uploading it
  puts that setup in the sidebar.
- **Use these settings in the sidebar**: the same, from the open run.

The setup is the gauge, period, spin-up, forcing, model, the initial search
ranges and their limits, the CPMs (with their weights, so a CPM missing from
`cpm_configs/` comes back too) and every NSGA-II setting except the number of
processes, which belongs to the machine. Change anything, then press
**Calibrate** to run it again.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--gauge` | *asked* | USGS site number, e.g. `02317620` |
| `--n` | *asked* | random realizations per model |
| `--start-year`, `--end-year` | *asked* | scored period |
| `--n-procs` | *asked* | number of parallel processes |
| `--warmup-years` | `1` | years simulated before `start-year` but not scored |
| `--precip-source` | `aorc` | `aorc` (basin-averaged NOAA AORC, hourly summed to local days) or `nasa` (NASA POWER) |
| `--nasa-sampling` | `point` | NASA POWER grid cell at the gauge (`point`) or area-weighted over the basin (`basin`) |
| `--et-method` | `Turc` | `Penman`, `Hargreaves-Samani`, `Blaney-Criddle`, `Priestley-Taylor` or `Turc` |
| `--models` | all 46 | subset, e.g. `"7,8,10-12"` |
| `--skip-snow` | off | leave out the 12 snow models that need temperature |
| `--seed` | `1` | random seed |

For example, to try only GR4J (7) and a few others with NASA POWER rain averaged
over the basin:

```bash
python find_best_model.py --gauge 02317620 --n 500 --start-year 2021 --end-year 2025 \
    --n-procs 8 --models "7,11,13,28,34" --precip-source nasa --nasa-sampling basin
```

## Files

| File | Purpose |
|---|---|
| `find_best_model.py` | main script: data download, parallel random search, result tables |
| `compare_models.ipynb` | ranking, plots and model selection |
| `cpm_app.py` | Streamlit app to design and compare composite performance metrics |
| `composite_metric.py` | CPM components and score, configuration JSON files, ensemble runner that keeps the series |
| `nsga_app.py` | Streamlit app to calibrate a model with NSGA-II against up to three CPMs |
| `nsga_calibration.py` | NSGA-II, Pareto fronts, spin-up in days, and the `nsga_calib/` run files |
| `marrmot_models.py` | Python port of the 46 MARRMoT model structures (`MODEL_REGISTRY`) |
| `model_parameter_space.py` | parameter sampling ranges for every model; edit these to change the prior |
| `performance_metrics.py` | NSE, KGE, PBIAS, pdi and other goodness-of-fit metrics |
| `get_usgs_gauge_data.py` | USGS streamflow + NASA POWER forcing and potential ET methods |
| `get_aorc_basin_rainfall.py` | basin-averaged hourly AORC rainfall read from the AWS Zarr store; also runs on its own |
| `get_watershed_boundary.py` | basin polygon from the USGS NLDI service |

`data/`, `performance/` and `nsga_calib/` are created when you run the
scripts and are not tracked by git.

## Notes

- This is a random search, not a calibration. Use it to pick a model
  structure, then calibrate it, starting from the parameters in
  `best_model_<metric>.json`.
- A model is dropped from the results (and listed in `run_info.json` under
  `models_removed`) when all of its realizations fail.
- The parameter ranges in `model_parameter_space.py` fill gaps in the MARRMoT
  catalogue. Treat them as a starting point, not a calibrated prior.

## References

Knoben, W. J. M., Freer, J. E., Fowler, K. J. A., Peel, M. C., & Woods, R. A.
(2019). Modular Assessment of Rainfall–Runoff Models Toolbox (MARRMoT) v1.2.
*Geoscientific Model Development*, 12, 2463–2480.
https://doi.org/10.5194/gmd-12-2463-2019
