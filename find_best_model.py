"""
Search for the best MARRMoT model structure for a USGS watershed.

For a given USGS gauge, this script:

1. Downloads daily streamflow with `get_usgs_gauge_data.py` into `data/`,
   plus the forcing products the run needs, all kept in one table,
   `data/usgs_<gauge>_forcing.parquet` (columns prefixed by product):
     - ET and mean air temperature (needed by the 12 snow models) from NASA
       POWER, either at the gauge location (--nasa-sampling point, default)
       or averaged over the drainage basin (--nasa-sampling basin);
     - precipitation from AORC averaged over the basin (--precip-source
       aorc, default; hourly, summed to local days) or from NASA POWER
       (--precip-source nasa, same sampling as the ET).
   A product already in the table for the requested years is reused; a
   missing one is downloaded and added to the table.
2. Runs N Latin-hypercube parameter realizations of every MARRMoT model
   (the same random search as `run_param_sweep` in Example 5), in parallel.
3. Saves one table per model, with each realization's parameters and its
   NSE, KGE (+ KGE_r, KGE_alpha, KGE_beta), PBIAS, correlation, peak-flow difference,
   pdi_ave, pdi_ave2, pdi_ave3, pdi_low, pdi_high and log_NSE, to
   `performance/<gauge>/model_NN_<name>.parquet`, plus a `summary.parquet`
   with each model's best scores.

A model is removed (no table written) when every one of its realizations
fails. Each model's first year(s) are a warm-up period and are not scored.

Usage (anything not passed on the command line is asked for interactively):

    python find_best_model.py
    python find_best_model.py --gauge 02317620 --n 200 --start-year 2021 --end-year 2025 --n-procs 8

Add --skip-snow for watersheds without snow: the 12 models that need
temperature are left out and, with point sampling, a missing NASA POWER
product is not downloaded (the gauge file's forcing is used). If that
download fails anyway, the run continues the same way instead of stopping.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # all helper modules live next to this script

import get_aorc_basin_rainfall as aorc  # noqa: E402
import get_usgs_gauge_data as usgs  # noqa: E402
import marrmot_models as marrmot  # noqa: E402
import performance_metrics as pm  # noqa: E402
from model_parameter_space import check_parameter_space, sample_parameters  # noqa: E402

DATA_DIR = HERE / "data"
PERFORMANCE_DIR = HERE / "performance"

INITIAL_STORAGE = 1.0  # [mm] every store starts nearly empty; the warm-up fills it
Q_FLOOR = 1e-4         # [mm/day] same floor as Example 5, keeps log metrics defined
TEMPERATURE_COL = "T2M_MEAN (°C)"
PRECIP_COL = "Precipitation (mm/day)"
MAX_MISSING_AORC_DAYS = 30  # more than this and AORC doesn't cover the period
SUMMARY_METRICS = ["NSE", "KGE", "log_NSE", "pdi_ave", "pdi_low"]
MAX_CHUNK = 20         # realizations per parallel task


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
#
# All forcing products of a gauge live in one table,
# data/usgs_<gauge>_forcing.parquet, with every column prefixed by its source:
#
#   nasa_point: ...  NASA POWER at the gauge location (one grid cell)
#   nasa_basin: ...  NASA POWER averaged over the drainage basin
#   aorc: ...        AORC precipitation averaged over the drainage basin
#
# A product is only downloaded when it is missing (or doesn't cover the
# requested years); it is then merged into the table next to the others.

FORCING_SOURCES = ("nasa_point", "nasa_basin", "aorc")

# NASA POWER grids (degrees). Meteorology and precipitation come from MERRA-2,
# with cells centered on multiples of 0.5 deg lat x 0.625 deg lon; solar
# radiation comes from a 1 x 1 deg grid with edges on whole degrees.
MERRA2_DLAT, MERRA2_DLON = 0.5, 0.625
SOLAR_DDEG = 1.0


def _covers(path, first_year, last_year):
    """True if the parquet at `path` exists and spans first_year..last_year."""
    if not path.exists():
        return False
    index = pd.read_parquet(path, columns=[]).index
    return len(index) > 0 and index.min().year <= first_year and index.max().year >= last_year


def _forcing_path(gauge):
    return DATA_DIR / f"usgs_{gauge}_forcing.parquet"


def _source_frame(table, source):
    """The columns of one source, with the 'source: ' prefix removed."""
    prefix = f"{source}: "
    cols = [c for c in table.columns if c.startswith(prefix)]
    return table[cols].rename(columns=lambda c: c[len(prefix):])


def _source_covers(table, source, start_date, end_date):
    """True if `source` has a value for every column and day in the range."""
    frame = _source_frame(table, source)
    if frame.empty:
        return False
    frame = frame.reindex(pd.date_range(start_date, end_date))
    return not frame.isna().any().any()


def _add_source(table, gauge, source, frame):
    """Merges a freshly downloaded product into the forcing table and saves it."""
    new = frame.add_prefix(f"{source}: ")
    table = new.combine_first(table) if not table.empty else new
    table.to_parquet(_forcing_path(gauge))
    print(f"  added '{source}' to {_forcing_path(gauge).relative_to(HERE)}")
    return table


def _load_forcing_table(gauge):
    """
    Reads data/usgs_<gauge>_forcing.parquet (empty table if missing).

    Files written by earlier versions of this script are migrated in place:
    the old forcing file (NASA POWER at the gauge, no prefixes) becomes the
    'nasa_point' columns, and the rain of an old ..._forcing_aorc.parquet
    becomes the 'aorc' column (that file is then removed, since its ET and
    temperature were a copy of the NASA POWER ones).
    """
    path = _forcing_path(gauge)
    table = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    migrated = False

    is_legacy = len(table.columns) > 0 and not any(
        c.startswith(f"{s}: ") for c in table.columns for s in FORCING_SOURCES)
    if is_legacy:
        table = table.add_prefix("nasa_point: ")
        migrated = True

    legacy_aorc = DATA_DIR / f"usgs_{gauge}_forcing_aorc.parquet"
    if legacy_aorc.exists():
        old = pd.read_parquet(legacy_aorc)
        parts = [old[[PRECIP_COL]].add_prefix("aorc: ")]
        if _source_frame(table, "nasa_point").empty and TEMPERATURE_COL in old:
            parts.append(old.drop(columns=PRECIP_COL).add_prefix("nasa_point: "))
        new = pd.concat(parts, axis=1)
        table = new.combine_first(table) if not table.empty else new
        migrated = True

    if migrated:
        table.to_parquet(path)
        legacy_aorc.unlink(missing_ok=True)
        print(f"Migrated the old forcing file(s) of {gauge} into {path.relative_to(HERE)}")
    return table


def _nasa_power_point(latitude, longitude, start_date, end_date, attempts=3):
    """
    Daily NASA POWER forcing at one location: the raw variables, the five ET
    estimates and the mean temperature (needed by the snow models).
    """
    for attempt in range(1, attempts + 1):
        try:
            met = usgs.download_nasa_power_data(
                latitude, longitude, start_date.replace("-", ""), end_date.replace("-", "")
            )
            break
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  attempt {attempt} failed ({type(exc).__name__}); retrying...")
            time.sleep(5 * attempt)
    missing = met.eq(-999).any(axis=1).sum()  # NASA POWER's fill value
    if missing:
        print(f"  NASA POWER has {missing} day(s) with missing values; interpolating them")
        met = met.replace(-999, np.nan).interpolate(limit_direction="both")
    met = usgs._add_et_columns(met)
    met[TEMPERATURE_COL] = (met["T2M_MAX (°C)"] + met["T2M_MIN (°C)"]) / 2
    return met


def _grid_edges(lo, hi, step, offset):
    """Cell edges `offset + k * step` from just below `lo` to just above `hi`."""
    k = np.arange(math.floor((lo - offset) / step), math.ceil((hi - offset) / step) + 1)
    return offset + k * step


def nasa_power_basin_pieces(polygon):
    """
    Cuts the basin along the cell edges of both NASA POWER grids (MERRA-2
    meteorology and the 1-degree solar grid). Each piece lies inside exactly
    one cell of each grid, so a query at the piece's center returns the right
    cell for every variable.

    Returns:
        list[tuple[float, float, float]]: (lat, lon, weight) per piece; the
            weight is the piece's area (deg^2 scaled by cos(lat)).
    """
    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    lat_edges = np.unique(np.round(np.concatenate([
        _grid_edges(lat_min, lat_max, MERRA2_DLAT, MERRA2_DLAT / 2),
        _grid_edges(lat_min, lat_max, SOLAR_DDEG, 0.0)]), 6))
    lon_edges = np.unique(np.round(np.concatenate([
        _grid_edges(lon_min, lon_max, MERRA2_DLON, MERRA2_DLON / 2),
        _grid_edges(lon_min, lon_max, SOLAR_DDEG, 0.0)]), 6))

    pieces = []
    for s, n in zip(lat_edges[:-1], lat_edges[1:]):
        for w, e in zip(lon_edges[:-1], lon_edges[1:]):
            area = polygon.intersection(aorc.shapely.box(w, s, e, n)).area
            if area > 0:
                lat, lon = (s + n) / 2, (w + e) / 2
                pieces.append((lat, lon, area * math.cos(math.radians(lat))))
    return pieces


def _nasa_power_basin(gauge, start_date, end_date):
    """
    Basin-averaged NASA POWER forcing ("cookie cutter"): every variable,
    including each ET estimate, is computed per grid cell and then averaged
    with the area of the basin that falls in that cell as weight.
    """
    pieces = nasa_power_basin_pieces(aorc.basin_polygon(gauge))
    if not pieces:
        raise RuntimeError(f"The basin of USGS {gauge} doesn't overlap the NASA POWER grid")
    total = sum(w for _, _, w in pieces)
    print(f"  basin covers {len(pieces)} NASA POWER grid piece(s)")

    average = None
    for i, (lat, lon, w) in enumerate(pieces, 1):
        print(f"  piece {i}/{len(pieces)} ({lat:.3f}, {lon:.3f}), "
              f"{100 * w / total:.1f}% of the basin")
        part = _nasa_power_point(lat, lon, start_date, end_date) * (w / total)
        average = part if average is None else average + part
    return average


def _download_nasa(gauge, sampling, start_date, end_date):
    if sampling == "basin":
        return _nasa_power_basin(gauge, start_date, end_date)
    latitude, longitude, _ = usgs.get_usgs_site_info(gauge)
    return _nasa_power_point(latitude, longitude, start_date, end_date)


def _forcing_from_gauge_file(gauge_df, start_date, end_date):
    """
    Fallback forcing without temperature, taken from the gauge parquet itself
    (NASA POWER at the gauge, only on days with observed flow). Days missing
    there get zero rain and interpolated ET.
    """
    forcing = gauge_df.drop(columns=["Drainage Area (km2)", "Flow(cms)"])
    forcing = forcing.reindex(pd.date_range(start_date, min(pd.Timestamp(end_date), forcing.index.max())))
    missing = int(forcing[PRECIP_COL].isna().sum())
    if missing:
        print(f"  {missing} day(s) missing from the gauge file: rain set to 0, ET interpolated")
    forcing[PRECIP_COL] = forcing[PRECIP_COL].fillna(0.0)
    return forcing.interpolate(limit_direction="both")


def _download_aorc_precip(gauge, start_date, end_date):
    """
    Daily basin-averaged AORC precipitation [mm/day] for start_date..end_date.

    AORC is hourly in UTC while USGS daily flows are local-time days, so the
    hours are shifted to the gauge's local standard time (offset estimated
    from its longitude, e.g. -5 h in Florida) before summing them per day.
    Each AORC value is stamped at the end of its hour, hence the extra -1 h.
    """
    _, longitude, _ = usgs.get_usgs_site_info(gauge)
    utc_offset = pd.Timedelta(hours=round(longitude / 15))
    first, last = pd.Timestamp(start_date), pd.Timestamp(end_date)

    # One extra day on each side so the local days at the edges are complete
    fetch_start = max(first - pd.Timedelta(days=1), aorc.AORC_FIRST_TIME)
    hourly = aorc.get_aorc_basin_rainfall(
        gauge, fetch_start.strftime("%Y-%m-%d"),
        (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )["precip_mm"]
    hourly.index = hourly.index - pd.Timedelta(hours=1) + utc_offset

    # min_count=24: a day missing any hour is NaN instead of an undercount
    daily = hourly.resample("D").sum(min_count=24).reindex(pd.date_range(first, last))
    missing = int(daily.isna().sum())
    if missing > MAX_MISSING_AORC_DAYS:
        raise RuntimeError(
            f"AORC has no data for {missing} day(s) of {first.date()}..{last.date()} "
            f"(AORC v1.1 starts {aorc.AORC_FIRST_TIME.date()}); pick other years "
            "or use --precip-source nasa"
        )
    if missing:
        print(f"  {missing} day(s) missing from AORC: rain set to 0")
    return daily.fillna(0.0)


def needs_temperature(model_num):
    return "T" in marrmot.MODEL_REGISTRY[model_num]["forcing_vars"]


def prepare_data(gauge, start_year, end_year, warmup_years, et_method, skip_snow=False,
                 precip_source="aorc", nasa_sampling="point"):
    """
    Downloads (or reuses) the gauge data and returns the arrays every
    simulation needs, covering the warm-up and evaluation periods.

    ET and temperature come from NASA POWER, sampled at the gauge location
    (nasa_sampling="point") or averaged over the basin ("basin").
    Precipitation comes from AORC (precip_source="aorc", basin average) or
    from the same NASA POWER product ("nasa"). Products already in
    data/usgs_<gauge>_forcing.parquet for these years are not downloaded
    again; new ones are added to that file.

    With `skip_snow=True` and point sampling, a missing NASA POWER product is
    not downloaded: the forcing stored in the gauge file is used instead
    (also the fallback if that download fails). It has no temperature, so
    the returned "T" is None and the snow models have to be skipped.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    first_year = start_year - warmup_years
    start_date, end_date = f"{first_year}-01-01", f"{end_year}-12-31"

    gauge_path = DATA_DIR / f"usgs_{gauge}_daily.parquet"
    if _covers(gauge_path, first_year, end_year):
        print(f"Using existing {gauge_path.relative_to(HERE)}")
    else:
        print(f"Downloading USGS {gauge} streamflow and forcings ({start_date} to {end_date})...")
        usgs.download_usgs_gauge_data(gauge, start_date, end_date, output_dir=str(DATA_DIR))

    gauge_df = pd.read_parquet(gauge_path)
    table = _load_forcing_table(gauge)
    table_name = _forcing_path(gauge).relative_to(HERE)

    # ET and temperature (and the rain, with precip_source="nasa")
    nasa_source = f"nasa_{nasa_sampling}"
    nasa = None
    if _source_covers(table, nasa_source, start_date, end_date):
        print(f"Using '{nasa_source}' from {table_name}")
        nasa = _source_frame(table, nasa_source)
    elif skip_snow and nasa_sampling == "point":
        print("Skipping the NASA POWER download (--skip-snow); using the gauge file's forcing")
    else:
        print(f"Downloading NASA POWER forcing '{nasa_source}' ({start_date} to {end_date})...")
        try:
            nasa = _download_nasa(gauge, nasa_sampling, start_date, end_date)
            table = _add_source(table, gauge, nasa_source, nasa)
        except Exception as exc:
            if nasa_sampling == "basin":
                raise
            print(f"  WARNING: NASA POWER download failed ({type(exc).__name__}: {exc})")
            print("  Continuing with the gauge file's forcing; models that need temperature are skipped")
    if nasa is None:
        nasa = _forcing_from_gauge_file(gauge_df, start_date, end_date)

    forcing = nasa.loc[start_date:end_date].copy()

    if precip_source == "aorc":
        if _source_covers(table, "aorc", start_date, end_date):
            print(f"Using 'aorc' from {table_name}")
        else:
            print(f"Computing basin-averaged AORC rainfall ({start_date} to {end_date})...")
            precip = _download_aorc_precip(gauge, start_date, end_date)
            table = _add_source(table, gauge, "aorc", precip.to_frame(PRECIP_COL))
        forcing[PRECIP_COL] = _source_frame(table, "aorc")[PRECIP_COL].reindex(forcing.index)

    has_temperature = TEMPERATURE_COL in forcing and not skip_snow

    area = gauge_df["Drainage Area (km2)"].iloc[0] if len(gauge_df) else np.nan
    if not np.isfinite(area):
        raise RuntimeError(f"USGS {gauge} has no drainage area on record; cannot convert flow to mm/day")

    dates = forcing.index
    qobs = (gauge_df["Flow(cms)"] * 86.4 / area).reindex(dates)  # cms -> mm/day
    eval_mask = np.asarray(dates >= pd.Timestamp(f"{start_year}-01-01"))

    n_obs = int(qobs[eval_mask].notna().sum())
    if n_obs == 0:
        raise RuntimeError(f"USGS {gauge} has no streamflow between {start_year} and {end_year}")

    rain_source = "aorc" if precip_source == "aorc" else nasa_source
    print(f"Drainage area: {area:.1f} km2")
    print(f"Forcing:       rain '{rain_source}', ET/temperature '{nasa_source}'")
    print(f"Simulation:    {dates[0].date()} to {dates[-1].date()} ({len(dates)} days, "
          f"{warmup_years} warm-up year(s))")
    print(f"Evaluation:    {dates[eval_mask][0].date()} to {dates[-1].date()} "
          f"({n_obs} days with observed flow)")

    return {
        "P": forcing[PRECIP_COL].to_numpy(dtype=float),
        "Ep": forcing[f"ET {et_method} (mm/day)"].to_numpy(dtype=float),
        "T": forcing[TEMPERATURE_COL].to_numpy(dtype=float) if has_temperature else None,
        "t": dates.dayofyear.to_numpy(dtype=float),
        "Qobs": qobs.to_numpy(dtype=float)[eval_mask],
        "dates": dates[eval_mask],
        "eval_mask": eval_mask,
    }


# ---------------------------------------------------------------------------
# Simulation (runs inside the worker processes)
# ---------------------------------------------------------------------------

_DATA = {}


def _init_worker(data):
    _DATA.update(data)
    # Bad parameter sets overflow all the time; that is reported per realization.
    warnings.filterwarnings("ignore")


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return np.nan


def compute_metrics(sim):
    """
    Same indexes as `pm.summary` + log_NSE (Example 5), but each one is
    computed on its own so a single undefined index (e.g. pdi_low on a
    stream with zero low flows) is NaN instead of failing the realization.
    """
    try:
        kge = pm.kge(sim, "Q", "Qobs")
    except Exception:
        kge = dict.fromkeys(("KGE", "r", "alpha", "beta"), np.nan)
    # KGE components are prefixed: several MARRMoT models have parameters
    # named r, alpha or beta, which share the table with the metrics.
    return {
        "NSE": _safe(pm.nse, sim, "Q", "Qobs"),
        "KGE": kge["KGE"],
        "KGE_r": kge["r"],
        "KGE_alpha": kge["alpha"],
        "KGE_beta": kge["beta"],
        "PBIAS": _safe(pm.pbias, sim, "Q", "Qobs"),
        "correlation": _safe(pm.correlation, sim, "Q", "Qobs"),
        "mean_annual_peak_flow_diff": _safe(pm.mean_annual_peak_flow_difference, sim, "Q", "Qobs"),
        "pdi_ave": _safe(pm.percentiles_index, sim, [5, 25, 50, 75, 95], "Q", "Qobs"),
        "pdi_ave2": _safe(pm.percentiles_index, sim, [1,5, 25, 75, 95,99], "Q", "Qobs"),
        "pdi_ave3": _safe(pm.percentiles_index, sim, [10,30,60,90], "Q", "Qobs"),
        "pdi_low": _safe(pm.percentiles_index, sim, [1, 5, 10, 15], "Q", "Qobs"),
        "pdi_high": _safe(pm.percentiles_index, sim, [85, 90, 95, 99], "Q", "Qobs"),
        "log_NSE": _safe(pm.nse, sim, "Q", "Qobs", log=True),
    }


def _run_one(model_num, params):
    """Simulate one parameter set; returns (metrics, None) or (None, error)."""
    try:
        S0 = np.full(marrmot.MODEL_REGISTRY[model_num]["n_stores"], INITIAL_STORAGE)
        result = marrmot.simulate(model_num, _DATA["P"], _DATA["Ep"], params, S0=S0,
                                  dt=1.0, T=_DATA["T"], t=_DATA["t"])
        Q = np.asarray(result["Q"], dtype=float)[_DATA["eval_mask"]]
        if not np.all(np.isfinite(Q)):
            raise FloatingPointError("simulated flow is not finite")
        sim = pd.DataFrame({"Q": np.maximum(Q, Q_FLOOR), "Qobs": _DATA["Qobs"]},
                           index=_DATA["dates"])
        return compute_metrics(sim), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_chunk(model_num, start, param_rows):
    return model_num, start, [_run_one(model_num, params) for params in param_rows]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _model_filename(model_num):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", marrmot.MODEL_REGISTRY[model_num]["name"]).strip("_")
    return f"model_{model_num:02d}_{slug}.parquet"


def _build_table(samples, outcomes):
    """Parameters, error, then metrics -- the layout of run_param_sweep's table."""
    rows = []
    for (_, params), (metrics, error) in zip(samples.iterrows(), outcomes):
        row = dict(params)
        row["error"] = error
        if metrics is not None:
            clash = row.keys() & metrics.keys()
            if clash:  # a metric would overwrite a parameter of the same name
                raise ValueError(f"metric name(s) {sorted(clash)} clash with model parameters")
            row.update(metrics)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.index.name = "realization"
    return table


def run_search(gauge, models, n, n_procs, data, seed):
    out_dir = PERFORMANCE_DIR / gauge
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = {m: sample_parameters(m, n, seed=seed + m) for m in models}
    outcomes = {m: [None] * n for m in models}
    chunk = max(1, min(MAX_CHUNK, math.ceil(n / n_procs)))
    pending = {m: math.ceil(n / chunk) for m in models}

    summary, removed = [], {}
    with ProcessPoolExecutor(max_workers=n_procs, initializer=_init_worker,
                             initargs=(data,)) as pool:
        futures = {}
        for m in models:
            records = samples[m].to_dict("records")
            for start in range(0, n, chunk):
                fut = pool.submit(_run_chunk, m, start, records[start:start + chunk])
                futures[fut] = (m, start, len(records[start:start + chunk]))

        for fut in as_completed(futures):
            m, start, size = futures[fut]
            try:
                _, _, results = fut.result()
            except Exception as exc:  # the worker itself died
                results = [(None, f"worker crashed: {type(exc).__name__}: {exc}")] * size
            outcomes[m][start:start + size] = results
            pending[m] -= 1
            if pending[m]:
                continue

            info = marrmot.MODEL_REGISTRY[m]
            label = f"[{len(summary) + len(removed) + 1:2d}/{len(models)}] Model {m:2d} {info['name'][:38]:<38}"
            path = out_dir / _model_filename(m)
            table = _build_table(samples[m], outcomes[m])
            n_ok = int(table["error"].isna().sum())

            if n_ok == 0:
                removed[m] = table["error"].iloc[0]
                path.unlink(missing_ok=True)  # don't leave a table from an older run
                print(f"{label} REMOVED - every realization failed ({removed[m][:80]})")
                continue

            table.to_parquet(path)
            best = {k: table[k].max() for k in SUMMARY_METRICS}
            summary.append({"model": m, "name": info["name"], "n_ok": n_ok,
                            "n_failed": n - n_ok, **{f"best_{k}": v for k, v in best.items()},
                            "file": path.name})
            print(f"{label} {n_ok:>5}/{n} ok   best NSE {best['NSE']:7.3f}   "
                  f"KGE {best['KGE']:7.3f}   log_NSE {best['log_NSE']:7.3f}")

    summary = pd.DataFrame(summary).set_index("model").sort_index() if summary else pd.DataFrame()
    if not summary.empty:
        summary.to_parquet(out_dir / "summary.parquet")
    return summary, removed


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

def _ask(prompt, cast, default=None, check=None, error="invalid value"):
    """Prompt until `cast(answer)` succeeds and passes `check`."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if not answer and default is not None:
            answer = str(default)
        try:
            value = cast(answer)
            if check is None or check(value):
                return value
        except ValueError:
            pass
        print(f"  {error}")


def _parse_models(text):
    """'7,8,10-12' -> [7, 8, 10, 11, 12]"""
    models = set()
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        models.update(range(int(lo), int(hi or lo) + 1))
    unknown = models - set(marrmot.MODEL_REGISTRY)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown MARRMoT model(s): {sorted(unknown)}")
    return sorted(models)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gauge", help="USGS site number, e.g. 02317620")
    parser.add_argument("--n", type=int, help="random realizations per model")
    parser.add_argument("--start-year", type=int, help="first year scored")
    parser.add_argument("--end-year", type=int, help="last year scored")
    parser.add_argument("--n-procs", type=int, help="number of parallel processes")
    parser.add_argument("--et-method", default="Turc", choices=usgs.ET_METHODS,
                        help="potential ET estimate used as Ep (default: Turc, as in Example 5)")
    parser.add_argument("--precip-source", default="aorc", choices=["aorc", "nasa"],
                        help="precipitation: basin-averaged AORC (default) or NASA POWER "
                             "(sampled as set by --nasa-sampling)")
    parser.add_argument("--nasa-sampling", default="point", choices=["point", "basin"],
                        help="NASA POWER (ET, temperature, NASA rain): the grid cell at the "
                             "gauge (default) or the area-weighted average of every cell "
                             "the basin overlaps")
    parser.add_argument("--warmup-years", type=int, default=1,
                        help="years simulated before start-year but not scored (default: 1)")
    parser.add_argument("--models", type=_parse_models, default=sorted(marrmot.MODEL_REGISTRY),
                        help="subset of models, e.g. '7,8,10-12' (default: all 46)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--skip-snow", action="store_true",
                        help="skip the 12 snow models that need temperature "
                             "(e.g. for watersheds without snow)")
    args = parser.parse_args()

    cpus = os.cpu_count() or 1
    if args.gauge is None:
        args.gauge = _ask("USGS gauge number", str, check=lambda s: s.isdigit(),
                          error="use the numeric site number, e.g. 02317620")
    if args.n is None:
        args.n = _ask("Number of random realizations per model (N)", int, 100,
                      check=lambda v: v > 0, error="must be a positive integer")
    if args.start_year is None:
        args.start_year = _ask("Starting year", int, check=lambda v: 1982 <= v,
                               error="NASA POWER forcing starts in 1981; use 1982 or later")
    if args.end_year is None:
        args.end_year = _ask("Ending year", int, check=lambda v: v >= args.start_year,
                             error=f"must be {args.start_year} or later")
    if args.n_procs is None:
        args.n_procs = _ask(f"Number of processors (this machine has {cpus})", int, cpus,
                            check=lambda v: v > 0, error="must be a positive integer")

    if not args.gauge.isdigit():
        parser.error("--gauge must be a numeric USGS site number")
    if args.end_year < args.start_year:
        parser.error("--end-year must not be before --start-year")
    return args


def main():
    args = parse_args()
    check_parameter_space()

    data = prepare_data(args.gauge, args.start_year, args.end_year,
                        args.warmup_years, args.et_method, skip_snow=args.skip_snow,
                        precip_source=args.precip_source, nasa_sampling=args.nasa_sampling)

    skipped = []
    if data["T"] is None:
        skipped = [m for m in args.models if needs_temperature(m)]
        args.models = [m for m in args.models if not needs_temperature(m)]
        if skipped:
            print(f"No temperature: skipping snow models {', '.join(map(str, skipped))}")
        if not args.models:
            sys.exit("Nothing left to evaluate: every selected model needs temperature")

    print(f"\nEvaluating {len(args.models)} models x {args.n} realizations "
          f"on {args.n_procs} processes...\n")
    t0 = time.time()
    summary, removed = run_search(args.gauge, args.models, args.n, args.n_procs, data, args.seed)
    elapsed = time.time() - t0

    out_dir = PERFORMANCE_DIR / args.gauge
    run_info = {
        "gauge": args.gauge,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "warmup_years": args.warmup_years,
        "n_realizations": args.n,
        "n_procs": args.n_procs,
        "precip_source": args.precip_source,
        "nasa_sampling": args.nasa_sampling,
        "et_method": args.et_method,
        "seed": args.seed,
        "initial_storage_mm": INITIAL_STORAGE,
        "temperature_available": data["T"] is not None,
        "models_evaluated": [int(m) for m in summary.index] if not summary.empty else [],
        "models_removed": {str(m): err for m, err in removed.items()},
        "models_skipped_no_temperature": skipped,
        "elapsed_seconds": round(elapsed, 1),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    print(f"\nDone in {elapsed / 60:.1f} min. {len(summary)} model(s) kept, {len(removed)} removed.")
    if not summary.empty:
        cols = [f"best_{k}" for k in SUMMARY_METRICS]
        print("\nTop 10 models by best KGE:")
        print(summary.sort_values("best_KGE", ascending=False)[["name"] + cols]
              .head(10).to_string(float_format="{:.3f}".format))
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
