"""
Composite performance metric (CPM) for simulated vs. observed discharge.

The CPM is built like KGE: one minus the (weighted) Euclidean distance from
the ideal point in the space of its components, every one of which is 1 for a
perfect simulation:

    CPM = 1 - sqrt( sum_i (s_i * (x_i - 1)) ** 2 ),   s_i = w_i / max(w)

with the components x_i

    NSE         Nash-Sutcliffe efficiency, capped to [0, 1]
    mean_ratio  mean(sim) / mean(obs)                      (KGE's beta)
    std_ratio   std(sim) / std(obs)                        (KGE's alpha)
    P5 ... P95  sim_p / obs_p for the 5, 10, 25, 75, 90 and 95th percentiles
    pearson     Pearson correlation                        (KGE's r)
    spearman    Spearman rank correlation

The weights w_i (0-10) are relative: they are divided by the largest one, so
giving the same weight to pearson, mean_ratio and std_ratio only (and 0 to
the rest) returns exactly KGE. NSE mixes timing and volume errors, the ratios
and percentiles compare the flow distributions, and the two correlations
measure timing (Spearman is less dominated by the peaks than Pearson).

This module also runs the random parameter ensemble whose series the
Streamlit app (`cpm_app.py`) scores, and reads/writes CPM configurations as
JSON files.
"""

import contextlib
import hashlib
import json
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, rankdata

import marrmot_models as marrmot

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "cpm_configs"

INITIAL_STORAGE = 1.0  # [mm] same as find_best_model.py
Q_FLOOR = 1e-4         # [mm/day] same as find_best_model.py
MAX_CHUNK = 20         # realizations per parallel task

PERCENTILES = (5, 10, 25, 75, 90, 95)

# key -> (label, help text), in the order they are shown and stored
COMPONENTS = {
    "NSE": ("NSE (capped 0–1)", "Nash-Sutcliffe efficiency clipped to [0, 1]: day-by-day squared error"),
    "mean_ratio": ("Mean ratio", "mean(sim) / mean(obs): total volume (KGE's beta)"),
    "std_ratio": ("Deviation ratio", "std(sim) / std(obs): flow variability (KGE's alpha)"),
    **{f"P{p}": (f"P{p} ratio", f"sim / obs flow at the {p}th percentile of each series")
       for p in PERCENTILES},
    "pearson": ("Pearson r", "linear correlation: timing, dominated by the peaks"),
    "spearman": ("Spearman ρ", "rank correlation: timing across all flow levels"),
}
COMPONENT_KEYS = list(COMPONENTS)

DEFAULT_WEIGHTS = dict.fromkeys(COMPONENT_KEYS, 5.0)

PRESETS = {
    "Equal weights": DEFAULT_WEIGHTS,
    "KGE (r, mean, deviation)": {"pearson": 10.0, "mean_ratio": 10.0, "std_ratio": 10.0},
    "Timing only": {"NSE": 10.0, "pearson": 10.0, "spearman": 10.0},
    "Distribution only": {"mean_ratio": 10.0, "std_ratio": 10.0,
                          **{f"P{p}": 10.0 for p in PERCENTILES}},
}


# ---------------------------------------------------------------------------
# Components and score
# ---------------------------------------------------------------------------

def _row_pearson(a, b):
    """Pearson r of every row of `a` (R, T) with the vector `b` (T,)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a @ b) / denom
    # A constant simulation (e.g. a model that never produces flow) has no
    # correlation with anything; count it as 0 instead of undefined.
    return np.where(np.isfinite(a).all(axis=1) & (denom == 0), 0.0, r)


def compute_components(sims, qobs):
    """
    Every CPM component (plus raw NSE and KGE, for reference) for each row of
    `sims`.

    Parameters:
        sims (np.ndarray): (R, T) simulated flows; a failed realization is a
            row of NaN.
        qobs (np.ndarray): (T,) observed flows; NaN days are left out.

    Returns:
        pd.DataFrame: one row per realization, columns COMPONENT_KEYS +
            ['NSE_raw', 'KGE']. A component that is undefined for this gauge
            (an observed percentile of 0) is NaN for every row.
    """
    valid = np.isfinite(qobs)
    obs, sim = qobs[valid], sims[:, valid]
    out = {}

    nse = 1.0 - ((sim - obs) ** 2).sum(axis=1) / ((obs - obs.mean()) ** 2).sum()
    out["NSE"] = np.clip(nse, 0.0, 1.0)
    out["mean_ratio"] = sim.mean(axis=1) / obs.mean()
    out["std_ratio"] = sim.std(axis=1) / obs.std()

    sim_p = np.percentile(sim, PERCENTILES, axis=1)  # (P, R)
    obs_p = np.percentile(obs, PERCENTILES)
    for p, s_p, o_p in zip(PERCENTILES, sim_p, obs_p):
        out[f"P{p}"] = s_p / o_p if o_p > 0 else np.full(len(sim), np.nan)

    out["pearson"] = _row_pearson(sim, obs)
    failed = ~np.isfinite(sim).all(axis=1)
    ranks = rankdata(np.where(failed[:, None], 0.0, sim), axis=1)
    out["spearman"] = np.where(failed, np.nan, _row_pearson(ranks, rankdata(obs)))

    out["NSE_raw"] = nse
    out["KGE"] = 1.0 - np.sqrt((out["pearson"] - 1) ** 2 + (out["std_ratio"] - 1) ** 2
                               + (out["mean_ratio"] - 1) ** 2)
    table = pd.DataFrame(out)
    table.index.name = "realization"
    return table


def undefined_components(components):
    """Components that are NaN for every realization (not computable here)."""
    return [k for k in COMPONENT_KEYS if components[k].isna().all()]


def cpm_score(components, weights):
    """
    CPM of every realization.

    Components that are undefined for every realization are left out of the
    distance (see `undefined_components`); a realization with any other
    missing component (a failed run) scores NaN.

    Parameters:
        components (pd.DataFrame): output of `compute_components`.
        weights (dict): component key -> weight (>= 0); missing keys are 0.

    Returns:
        pd.Series: CPM per realization (1 = perfect).
    """
    w = np.array([float(weights.get(k, 0.0)) for k in COMPONENT_KEYS])
    if w.max() <= 0:
        raise ValueError("At least one CPM weight must be above 0")
    skip = set(undefined_components(components))
    keys = [k for k, wk in zip(COMPONENT_KEYS, w) if wk > 0 and k not in skip]
    if not keys:
        raise ValueError("Every weighted CPM component is undefined for this gauge")
    scale = np.array([weights[k] for k in keys]) / w.max()
    deviation = components[keys].to_numpy() - 1.0
    return pd.Series(1.0 - np.sqrt(((scale * deviation) ** 2).sum(axis=1)),
                     index=components.index, name="CPM")


# ---------------------------------------------------------------------------
# Diagnostics of one simulation
# ---------------------------------------------------------------------------

def relative_error(sim, qobs, window=1):
    """100 * (sim - obs) / obs per day (NaN where obs <= 0), optionally a
    centered rolling mean over `window` days."""
    sim, qobs = pd.Series(sim), pd.Series(qobs)
    error = 100.0 * (sim - qobs) / qobs.where(qobs > 0)
    if window > 1:
        error = error.rolling(window, center=True, min_periods=1).mean()
    return error.to_numpy()


def flow_duration(q):
    """Flow duration curve: (exceedance probability in %, flows sorted from
    high to low), finite values only, Weibull plotting position."""
    flows = np.sort(q[np.isfinite(q)])[::-1]
    return 100.0 * np.arange(1, flows.size + 1) / (flows.size + 1), flows


def qq_quantiles(sim, qobs, percentiles=np.linspace(0, 100, 101)):
    """(observed, simulated) flow quantiles at `percentiles`, using only the
    days with an observed value."""
    valid = np.isfinite(qobs) & np.isfinite(sim)
    return np.percentile(qobs[valid], percentiles), np.percentile(sim[valid], percentiles)


def log_flow_grid(*series, n=256):
    """Evaluation points (log10 of flow) spanning every positive value of
    `series`, with a 5% margin on each side."""
    values = np.concatenate([np.log10(s[np.isfinite(s) & (s > 0)]) for s in series])
    lo, hi = values.min(), values.max()
    pad = 0.05 * (hi - lo or 1.0)
    return np.linspace(lo - pad, hi + pad, n)


def log_flow_pdf(q, grid):
    """Gaussian-KDE density of log10(q) on `grid` (positive, finite q only)."""
    values = np.log10(q[np.isfinite(q) & (q > 0)])
    if values.size < 2 or values.std() == 0:
        return np.full(grid.shape, np.nan)
    return gaussian_kde(values)(grid)


# ---------------------------------------------------------------------------
# Configurations (weights + identifier) as JSON
# ---------------------------------------------------------------------------

def config_id(weights):
    """Identifier of a CPM: the same weights always give the same ID."""
    canonical = {k: round(float(weights.get(k, 0.0)) * 2) / 2 for k in COMPONENT_KEYS}
    digest = hashlib.sha1(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    return f"CPM-{digest[:6].upper()}"


def config_record(name, weights, context=None, notes=""):
    """The JSON-ready description of a configuration."""
    weights = {k: float(weights.get(k, 0.0)) for k in COMPONENT_KEYS}
    return {
        "id": config_id(weights),
        "name": name,
        "weights": weights,
        "notes": notes,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "context": context or {},
    }


def save_config(record, directory=CONFIG_DIR):
    """Writes `record` to <directory>/<id>.json and returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2, default=_json_default))
    return path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def parse_configs(text):
    """
    Configurations in a JSON text: one configuration, a list of them, or
    {"configs": [...]}. Each one needs at least "weights"; unknown weight
    keys are ignored and missing ones are 0.

    Returns:
        list[dict]: {'name', 'weights', 'id'} per configuration.
    """
    data = json.loads(text)
    if isinstance(data, dict) and "configs" in data:
        data = data["configs"]
    if isinstance(data, dict):
        data = [data]
    configs = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("weights"), dict):
            raise ValueError("every configuration needs a 'weights' object")
        weights = {k: min(10.0, max(0.0, float(item["weights"].get(k, 0.0))))
                   for k in COMPONENT_KEYS}
        configs.append({"name": str(item.get("name") or config_id(weights)),
                        "weights": weights, "id": config_id(weights)})
    return configs


def library():
    """Saved configurations in CONFIG_DIR: list of (path, record), newest first."""
    entries = []
    for path in sorted(CONFIG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            entries.append((path, json.loads(path.read_text())))
        except (OSError, ValueError):
            continue
    return entries


# ---------------------------------------------------------------------------
# Random parameter ensemble (keeps every simulated series)
# ---------------------------------------------------------------------------

_DATA = {}


def _init_worker(data):
    _DATA.update(data)
    warnings.filterwarnings("ignore")  # bad parameter sets overflow all the time


def _simulate_one(model_num, params):
    """Evaluation-period flow of one parameter set, or (None, error)."""
    try:
        S0 = np.full(marrmot.MODEL_REGISTRY[model_num]["n_stores"], INITIAL_STORAGE)
        result = marrmot.simulate(model_num, _DATA["P"], _DATA["Ep"], params, S0=S0,
                                  dt=1.0, T=_DATA["T"], t=_DATA["t"])
        Q = np.asarray(result["Q"], dtype=float)[_DATA["eval_mask"]]
        if not np.all(np.isfinite(Q)):
            raise FloatingPointError("simulated flow is not finite")
        return np.maximum(Q, Q_FLOOR), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _simulate_chunk(model_num, start, param_rows):
    return start, [_simulate_one(model_num, params) for params in param_rows]


def _worker_data(data):
    return {k: data[k] for k in ("P", "Ep", "T", "t", "eval_mask")}


def ensemble_pool(data, n_procs):
    """A process pool whose workers hold `data`, to pass to `run_ensemble`
    when it is called many times on the same data (e.g. once per NSGA-II
    generation). Use it as a context manager."""
    return ProcessPoolExecutor(max_workers=n_procs, initializer=_init_worker,
                               initargs=(_worker_data(data),))


def run_ensemble(model_num, data, samples, n_procs=1, progress=None, pool=None):
    """
    Simulates every parameter set in `samples` and keeps the series.

    Parameters:
        model_num (int): MARRMoT model number.
        data (dict): output of `find_best_model.prepare_data`.
        samples (pd.DataFrame): one parameter set per row.
        n_procs (int): parallel processes (1 = run here, in this process).
        progress (callable, optional): called as progress(done, total).
        pool (ProcessPoolExecutor, optional): from `ensemble_pool(data,
            n_procs)`; used instead of starting a new one.

    Returns:
        tuple[np.ndarray, list]: (Q, errors). Q is (len(samples), T) over the
            evaluation period, NaN rows for the failed realizations; errors
            holds None or the error message of each realization.
    """
    worker_data = _worker_data(data)
    records = samples.to_dict("records")
    n = len(records)
    Q = np.full((n, int(np.sum(data["eval_mask"]))), np.nan)
    errors = [None] * n

    def store(start, results):
        for i, (q, error) in enumerate(results, start):
            if q is None:
                errors[i] = error
            else:
                Q[i] = q

    if n_procs <= 1 and pool is None:
        _DATA.update(worker_data)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for i, params in enumerate(records):
                store(i, [_simulate_one(model_num, params)])
                if progress:
                    progress(i + 1, n)
        return Q, errors

    chunk =max(1, min(MAX_CHUNK, math.ceil(n / n_procs)))
    done = 0
    with contextlib.nullcontext(pool) if pool is not None else ensemble_pool(data, n_procs) as pool:
        futures = {pool.submit(_simulate_chunk, model_num, s, records[s:s + chunk]):
                   (s, len(records[s:s + chunk])) for s in range(0, n, chunk)}
        for fut in as_completed(futures):
            start, size = futures[fut]
            try:
                store(*fut.result())
            except Exception as exc:  # the worker itself died
                store(start, [(None, f"worker crashed: {type(exc).__name__}: {exc}")] * size)
            done += size
            if progress:
                progress(done, n)
    return Q, errors
