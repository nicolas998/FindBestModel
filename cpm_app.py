"""
Streamlit app to design a composite performance metric (CPM) that scores both
the timing error and the flow distribution of a simulation.

For a USGS gauge it downloads the streamflow and forcing data (same sources
and cache as `find_best_model.py`), runs N random parameter sets of one
MARRMoT model (CRB3, model 11, by default) and keeps every simulated series.
Up to five CPM configurations (component weights, see `composite_metric.py`)
can be edited side by side: each picks its best realization, which is drawn
over the observed hydrograph with its relative error and the difference
between its flow PDF and the observed one. Configurations are saved to and
loaded from JSON files in `cpm_configs/`.

Usage:

    streamlit run cpm_app.py
"""

import contextlib
import io
import json
import os
import time
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import composite_metric as cm
import find_best_model as fbm
import get_usgs_gauge_data as usgs
import marrmot_models as marrmot
from model_parameter_space import PARAM_SPACE, Derived, Frac, sample_parameters

MAX_CONFIGS = 5
DEFAULT_MODEL = 11  # Collie River Basin 3 (CRB3)
MAX_ENSEMBLE_SHOWN = 50

# Categorical slots 1-5 of the reference palette, light and dark steps
CONFIG_COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
INK = {
    "light": {"obs": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"obs": "#f0efec", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}

PRESETS = {
    "Equal weights": cm.DEFAULT_WEIGHTS,
    "KGE (r, mean, deviation)": {"pearson": 10.0, "mean_ratio": 10.0, "std_ratio": 10.0},
    "Timing only": {"NSE": 10.0, "pearson": 10.0, "spearman": 10.0},
    "Distribution only": {"mean_ratio": 10.0, "std_ratio": 10.0,
                          **{f"P{p}": 10.0 for p in cm.PERCENTILES}},
}

SLIDER_GROUPS = {
    "Timing / error": ["NSE", "pearson", "spearman"],
    "Moments": ["mean_ratio", "std_ratio"],
    "Low-flow percentiles": ["P5", "P10", "P25"],
    "High-flow percentiles": ["P75", "P90", "P95"],
}

st.set_page_config(page_title="CPM Explorer", page_icon=":material/water:", layout="wide")
ss = st.session_state


def _theme():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


# ---------------------------------------------------------------------------
# Configurations in session state
# ---------------------------------------------------------------------------
#
# Every configuration has a stable uid (for its widget keys and its color)
# and, separately, the CPM identifier computed from its weights.

def _weights(uid):
    return {k: ss[f"w_{uid}_{k}"] for k in cm.COMPONENT_KEYS}


def _name(uid):
    return ss[f"name_{uid}"].strip() or cm.config_id(_weights(uid))


def _label(uid):
    return f"{_name(uid)} ({cm.config_id(_weights(uid))})"


def _color(uid):
    return CONFIG_COLORS[_theme()][ss[f"color_{uid}"]]


def _add_config(name, weights):
    if len(ss.configs) >= MAX_CONFIGS:
        return False
    uid = uuid.uuid4().hex[:8]
    for k in cm.COMPONENT_KEYS:
        ss[f"w_{uid}_{k}"] = float(weights.get(k, 0.0))
    ss[f"name_{uid}"] = name
    # The color belongs to the configuration, not to its position
    used = {ss[f"color_{u}"] for u in ss.configs}
    ss[f"color_{uid}"] = next(i for i in range(MAX_CONFIGS) if i not in used)
    ss.configs.append(uid)
    return True


def _remove_config(uid):
    ss.configs.remove(uid)


def _duplicate_config(uid):
    if not _add_config(f"{_name(uid)} (copy)", _weights(uid)):
        ss.messages.append(("warning", f"At most {MAX_CONFIGS} configurations can be loaded"))


def _add_preset():
    preset = ss.preset
    if not _add_config(preset, PRESETS[preset]):
        ss.messages.append(("warning", f"At most {MAX_CONFIGS} configurations can be loaded"))


def _load_configs(configs, source):
    loaded = [c for c in configs if _add_config(c["name"], c["weights"])]
    if loaded:
        ss.messages.append(("success", f"Loaded {', '.join(c['id'] for c in loaded)} from {source}"))
    if len(loaded) < len(configs):
        ss.messages.append(("warning", f"{len(configs) - len(loaded)} configuration(s) not loaded: "
                                       f"at most {MAX_CONFIGS} at a time"))


def _load_from_library():
    entries = dict((str(p), r) for p, r in cm.library())
    configs = []
    for path in ss.library_pick:
        try:
            configs += cm.parse_configs(json.dumps(entries[path]))
        except (KeyError, ValueError) as exc:
            ss.messages.append(("error", f"Could not read {path}: {exc}"))
    _load_configs(configs, "the library")
    ss.library_pick = []


def _load_from_upload():
    files = ss[f"upload_{ss.upload_n}"] or []
    for file in files:
        try:
            _load_configs(cm.parse_configs(file.getvalue().decode("utf-8")), file.name)
        except (ValueError, UnicodeDecodeError) as exc:
            ss.messages.append(("error", f"{file.name} is not a CPM configuration file: {exc}"))
    ss.upload_n += 1  # a fresh uploader, so the same file can be loaded again


def _record(uid):
    """JSON record of a configuration, with its result on the current run."""
    weights = _weights(uid)
    context = {}
    run = ss.get("run")
    if run is not None:
        context = {k: run[k] for k in ("gauge", "model", "model_name", "start_year", "end_year",
                                        "warmup_years", "precip_source", "et_method",
                                        "n_realizations", "seed")}
        try:
            scores = cm.cpm_score(run["components"], weights)
            best = scores.idxmax() if scores.notna().any() else None
        except ValueError:
            best = None
        if best is not None:
            context.update({
                "best_realization": int(best),
                "best_cpm": float(scores[best]),
                "best_components": {k: float(v) for k, v in run["components"].loc[best].items()},
                "best_parameters": {k: float(v) for k, v in run["params"].loc[best].items()},
            })
    return cm.config_record(_name(uid), weights, context=context, notes=ss.get(f"notes_{uid}", ""))


def _save_config(uid):
    if max(_weights(uid).values()) <= 0:
        ss.messages.append(("error", "Set at least one weight above 0 before saving"))
        return
    path = cm.save_config(_record(uid))
    ss.messages.append(("success", f"Saved {path.stem} to {path.relative_to(cm.HERE)}"))


ss.setdefault("configs", [])
ss.setdefault("messages", [])
ss.setdefault("upload_n", 0)
if not ss.configs:
    _add_config("Equal weights", cm.DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Experiment: data + random ensemble
# ---------------------------------------------------------------------------

def _range_table(model):
    rows = []
    for name, spec in PARAM_SPACE[model].items():
        if isinstance(spec, tuple):
            rows.append({"parameter": name, "sampled as": "range", "low": spec[0], "high": spec[1]})
        elif isinstance(spec, Frac):
            ref = spec.ref if isinstance(spec.ref, str) else "a function of others"
            rows.append({"parameter": name, "sampled as": f"fraction of {ref}",
                         "low": spec.lo, "high": spec.hi})
        else:
            rows.append({"parameter": name, "sampled as": "derived", "low": None, "high": None})
    return pd.DataFrame(rows)


def _space_from_table(model, table):
    space = {}
    for row, (name, spec) in zip(table.itertuples(index=False), PARAM_SPACE[model].items()):
        if isinstance(spec, Derived):
            space[name] = spec
            continue
        lo, hi = float(row.low), float(row.high)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError(f"parameter {name}: low must be a number <= high")
        space[name] = (lo, hi) if isinstance(spec, tuple) else Frac(spec.ref, lo, hi)
    return space


def _clean_log(text):
    """Keeps the last state of lines rewritten with carriage returns."""
    return "\n".join(line.split("\r")[-1] for line in text.splitlines())


def run_experiment(settings, space):
    model = settings["model"]
    with st.status("Preparing the gauge data...", expanded=True) as status:
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                data = fbm.prepare_data(
                    settings["gauge"], settings["start_year"], settings["end_year"],
                    settings["warmup_years"], settings["et_method"],
                    skip_snow=not fbm.needs_temperature(model),
                    precip_source=settings["precip_source"])
        finally:
            st.code(_clean_log(log.getvalue()) or "(no output)", language=None)
        if fbm.needs_temperature(model) and data["T"] is None:
            raise RuntimeError(f"Model {model} needs temperature, which is not available")

        n = settings["n_realizations"]
        status.update(label=f"Running {n} realizations of model {model}...")
        samples = sample_parameters(model, n, seed=settings["seed"], space=space)
        bar = st.progress(0.0)
        t0 = time.time()
        Q, errors = cm.run_ensemble(
            model, data, samples, settings["n_procs"],
            progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total} realizations"))
        elapsed = time.time() - t0

        n_failed = sum(e is not None for e in errors)
        if n_failed == n:
            raise RuntimeError(f"Every realization failed, e.g. {errors[0]}")
        status.update(label=f"{n - n_failed}/{n} realizations ran in {elapsed:.1f} s",
                      state="complete", expanded=False)

    return {
        **settings,
        "model_name": marrmot.MODEL_REGISTRY[model]["name"],
        "dates": data["dates"],
        "qobs": data["Qobs"],
        "Q": Q,
        "errors": errors,
        "params": samples,
        "components": cm.compute_components(Q, data["Qobs"]),
        "elapsed": elapsed,
    }


with st.sidebar:
    st.header("Experiment")
    gauge = st.text_input("USGS gauge", "02249500")
    c1, c2 = st.columns(2)
    start_year = c1.number_input("First year", 1982, 2100, 2016)
    end_year = c2.number_input("Last year", 1982, 2100, 2020)
    warmup_years = st.number_input("Warm-up years (not scored)", 0, 10, 1)
    precip_source = st.selectbox("Precipitation", ["aorc", "nasa"],
                                 format_func={"aorc": "AORC, basin average",
                                              "nasa": "NASA POWER, at the gauge"}.get)
    et_method = st.selectbox("Potential ET", usgs.ET_METHODS, index=usgs.ET_METHODS.index("Turc"))
    models = sorted(marrmot.MODEL_REGISTRY)
    model = st.selectbox("MARRMoT model", models, index=models.index(DEFAULT_MODEL),
                         format_func=lambda m: f"{m:02d} · {marrmot.MODEL_REGISTRY[m]['name']}")
    with st.expander("Parameter ranges"):
        st.caption("Latin-hypercube sampling between low and high. "
                   "Fractions multiply the parameter they refer to.")
        ranges = st.data_editor(_range_table(model), key=f"ranges_{model}", hide_index=True,
                                disabled=["parameter", "sampled as"], width="stretch")
    n_realizations = st.number_input("Random realizations", 10, 5000, 100, step=10)
    c1, c2 = st.columns(2)
    seed = c1.number_input("Seed", 0, 1_000_000, 1)
    cpus = os.cpu_count() or 1
    n_procs = c2.number_input("Processes", 1, cpus, 1,
                              help="1 runs in the app's own process. Starting processes takes "
                                   "a few seconds on Windows, so more only pays off for slow "
                                   "models or many realizations.")
    run_clicked = st.button("Download data and run", type="primary", width="stretch")

st.title("Composite performance metric explorer")
st.caption("Weight timing and distribution components into one score, and see which "
           "simulation each weighting picks as the best.")

if run_clicked:
    gauge = gauge.strip()
    if not gauge.isdigit():
        st.error("The USGS gauge must be a numeric site number, e.g. 02249500")
    elif end_year < start_year:
        st.error("The last year must not be before the first year")
    else:
        try:
            space = _space_from_table(model, ranges)
            ss.run = run_experiment({
                "gauge": gauge, "start_year": int(start_year), "end_year": int(end_year),
                "warmup_years": int(warmup_years), "precip_source": precip_source,
                "et_method": et_method, "model": int(model),
                "n_realizations": int(n_realizations), "seed": int(seed), "n_procs": int(n_procs),
            }, space)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

for kind, text in ss.messages:
    getattr(st, kind)(text)
ss.messages = []


# ---------------------------------------------------------------------------
# Configuration editor
# ---------------------------------------------------------------------------

st.header("CPM configurations")
with st.expander("How the CPM is computed"):
    st.latex(r"\mathrm{CPM} = 1 - \sqrt{\sum_i \left(\frac{w_i}{\max_j w_j}\,(x_i - 1)\right)^2}")
    st.markdown(
        "Every component $x_i$ is 1 for a perfect simulation: NSE (capped to 0–1), the mean "
        "and standard-deviation ratios, the sim/obs ratios of the 5, 10, 25, 75, 90 and 95th "
        "percentiles, and the Pearson and Spearman correlations. Weights are relative "
        "(divided by the largest one), so equal weights on Pearson r, mean ratio and "
        "deviation ratio only give exactly KGE. The identifier (CPM-XXXXXX) is computed "
        "from the weights: the same weights always have the same identifier.")

c1, c2, c3 = st.columns([2, 3, 3], vertical_alignment="bottom")
with c1:
    st.selectbox("Preset", list(PRESETS), key="preset")
    st.button("Add configuration", on_click=_add_preset, width="stretch",
              disabled=len(ss.configs) >= MAX_CONFIGS)
with c2:
    saved = cm.library()
    st.multiselect("Saved configurations", [str(p) for p, _ in saved], key="library_pick",
                   format_func=lambda p: next(f"{r.get('id')} · {r.get('name')}"
                                              for q, r in saved if str(q) == p),
                   placeholder="cpm_configs/ is empty" if not saved else "Choose up to 5")
    st.button("Load selected", on_click=_load_from_library, width="stretch",
              disabled=not ss.get("library_pick"))
with c3:
    st.file_uploader("Load JSON file(s)", type="json", accept_multiple_files=True,
                     key=f"upload_{ss.upload_n}", on_change=_load_from_upload)

tabs = st.tabs([_name(uid) for uid in ss.configs])
for uid, tab in zip(list(ss.configs), tabs):
    with tab:
        weights = _weights(uid)
        cid = cm.config_id(weights)
        c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
        c1.text_input("Name", key=f"name_{uid}")
        c2.markdown(f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
                    f"background:{_color(uid)};vertical-align:middle;margin-right:8px'></span>"
                    f"**{cid}**", unsafe_allow_html=True)
        columns = st.columns(len(SLIDER_GROUPS))
        for col, (group, keys) in zip(columns, SLIDER_GROUPS.items()):
            with col:
                st.markdown(f"**{group}**")
                for k in keys:
                    label, help_text = cm.COMPONENTS[k]
                    st.slider(label, 0.0, 10.0, step=0.5, key=f"w_{uid}_{k}", help=help_text)
        if max(weights.values()) <= 0:
            st.warning("Every weight is 0: this configuration has no score.")
        st.text_input("Notes (saved with the configuration)", key=f"notes_{uid}")
        b1, b2, b3, b4 = st.columns(4)
        b1.button("Save to library", key=f"save_{uid}", on_click=_save_config, args=(uid,),
                  width="stretch", icon=":material/save:")
        b2.download_button("Download JSON", json.dumps(_record(uid), indent=2), f"{cid}.json",
                           mime="application/json", key=f"dl_{uid}", width="stretch",
                           icon=":material/download:")
        b3.button("Duplicate", key=f"dup_{uid}", on_click=_duplicate_config, args=(uid,),
                  width="stretch", disabled=len(ss.configs) >= MAX_CONFIGS,
                  icon=":material/content_copy:")
        b4.button("Remove", key=f"rm_{uid}", on_click=_remove_config, args=(uid,),
                  width="stretch", disabled=len(ss.configs) == 1, icon=":material/delete:")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

run = ss.get("run")
if run is None:
    st.info("Pick a gauge and a model in the sidebar and press **Download data and run**. "
            "Downloaded data is cached in `data/`, so later runs on the same gauge are fast.")
    st.stop()

theme = _theme()
ink = INK[theme]
dates, qobs, Q = run["dates"], run["qobs"], run["Q"]
components = run["components"]
n_ok = int(np.isfinite(Q).all(axis=1).sum())

st.header(f"USGS {run['gauge']} · model {run['model']:02d} {run['model_name']}")
st.caption(f"{run['start_year']}–{run['end_year']} scored ({run['warmup_years']} warm-up year(s)), "
           f"rain {run['precip_source'].upper()}, ET {run['et_method']}, "
           f"{n_ok}/{run['n_realizations']} realizations ran (seed {run['seed']}), "
           "flows in mm/day")

skipped = cm.undefined_components(components)
if skipped:
    st.warning(f"Left out of every CPM because an observed percentile is 0 at this gauge: "
               f"{', '.join(cm.COMPONENTS[k][0] for k in skipped)}")

# Score every realization with every configuration
results = []
for uid in ss.configs:
    try:
        scores = cm.cpm_score(components, _weights(uid))
    except ValueError:
        continue
    if scores.notna().any():
        results.append({"uid": uid, "id": cm.config_id(_weights(uid)), "scores": scores,
                        "best": int(scores.idxmax())})
if not results:
    st.warning("No configuration has a score: set at least one weight above 0.")
    st.stop()


def _style(fig, height, ytitle, ylog=False, xtitle=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor=ink["grid"], linecolor=ink["axis"], title_text=xtitle)
    fig.update_yaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"],
                     title_text=ytitle, type="log" if ylog else "linear")
    return fig


# Hydrograph
st.subheader("Hydrograph")
c1, c2, c3 = st.columns([3, 3, 1], vertical_alignment="bottom")
by_label = {f"{i}. {_label(r['uid'])}": r for i, r in enumerate(results, 1)}
reference = by_label[c1.selectbox("Ensemble ranked by", list(by_label))]
top_k = c2.slider("Simulations shown (top by that CPM)", 0, min(MAX_ENSEMBLE_SHOWN, n_ok),
                  min(20, n_ok))
ylog = c3.toggle("Log scale", value=True)

fig = go.Figure()
ranked = reference["scores"].dropna().sort_values(ascending=False).index[:top_k]
for i, r in enumerate(ranked):
    fig.add_trace(go.Scatter(
        x=dates, y=Q[r], mode="lines", line=dict(color=ink["muted"], width=1), opacity=0.35,
        hoverinfo="skip", legendgroup="ensemble", showlegend=i == 0,
        name=f"Top {top_k} by {reference['id']}"))
fig.add_trace(go.Scatter(x=dates, y=qobs, mode="lines", name="Observed",
                         line=dict(color=ink["obs"], width=1.5)))
for res in results:
    fig.add_trace(go.Scatter(x=dates, y=Q[res["best"]], mode="lines",
                             name=f"{res['id']} best (#{res['best']})",
                             line=dict(color=_color(res["uid"]), width=2)))
st.plotly_chart(_style(fig, 460, "Discharge (mm/day)", ylog), width="stretch")

# Scores table
grid = cm.log_flow_grid(qobs, *[Q[r["best"]] for r in results])
pdf_obs = cm.log_flow_pdf(qobs, grid)
rows = []
for res in results:
    best = res["best"]
    pdf_sim = cm.log_flow_pdf(Q[best], grid)
    same = [o["id"] for o in results if o is not res and o["best"] == best]
    rows.append({
        "CPM ID": res["id"], "name": _name(res["uid"]), "best run": best,
        "CPM": res["scores"][best],
        "rank by KGE": int((components["KGE"] > components["KGE"][best]).sum()) + 1,
        "KGE": components["KGE"][best], "NSE": components["NSE_raw"][best],
        **{cm.COMPONENTS[k][0]: components[k][best] for k in cm.COMPONENT_KEYS if k != "NSE"},
        "PDF L1 distance": float(np.trapezoid(np.abs(pdf_sim - pdf_obs), grid)),
        "same run as": ", ".join(same),
    })
st.dataframe(pd.DataFrame(rows).set_index("CPM ID"), width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in rows[0] if isinstance(rows[0][c], float)})
st.caption("Component values of each configuration's best run. PDF L1 distance is the area "
           "between the simulated and observed flow PDFs (0 = identical, 2 = no overlap). "
           "Rank by KGE is that run's place in the ensemble ranked by KGE.")

# Relative error
st.subheader("Relative error of the best simulations")
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
window = c1.slider("Smoothing window (days, centered mean)", 1, 91, 1, step=2)
clip = c2.toggle("Clip axis to 1–99th pct", value=True)
fig = go.Figure()
errors = []
for res in results:
    error = cm.relative_error(Q[res["best"]], qobs, window)
    errors.append(error)
    fig.add_trace(go.Scatter(x=dates, y=error, mode="lines", name=res["id"],
                             line=dict(color=_color(res["uid"]), width=1.5)))
fig.add_hline(y=0, line=dict(color=ink["axis"], width=1))
_style(fig, 380, "100 · (sim − obs) / obs  (%)")
if clip:
    lo, hi = np.nanpercentile(np.concatenate(errors), [1, 99])
    pad = 0.05 * (hi - lo or 1.0)
    fig.update_yaxes(range=[lo - pad, hi + pad])
st.plotly_chart(fig, width="stretch")

# PDF and its difference
st.subheader("Flow distribution")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
flows = 10 ** grid
fig.add_trace(go.Scatter(x=flows, y=pdf_obs, mode="lines", name="Observed",
                         line=dict(color=ink["obs"], width=2)), row=1, col=1)
for res in results:
    pdf_sim = cm.log_flow_pdf(Q[res["best"]], grid)
    color = _color(res["uid"])
    fig.add_trace(go.Scatter(x=flows, y=pdf_sim, mode="lines", name=res["id"], legendgroup=res["id"],
                             line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=flows, y=pdf_sim - pdf_obs, mode="lines", name=res["id"],
                             legendgroup=res["id"], showlegend=False,
                             line=dict(color=color, width=2)), row=2, col=1)
fig.add_hline(y=0, line=dict(color=ink["axis"], width=1), row=2, col=1)
_style(fig, 520, None)
fig.update_xaxes(type="log")
fig.update_xaxes(title_text="Discharge (mm/day)", row=2, col=1)
fig.update_yaxes(title_text="PDF of log10 Q", row=1, col=1)
fig.update_yaxes(title_text="sim − obs", row=2, col=1)
st.plotly_chart(fig, width="stretch")
st.caption("Kernel density of log10(discharge) on the scored days. Above zero in the lower "
           "panel, the simulation has those flows more often than observed.")

# Flow duration curve and Q-Q plot, on the days with an observed flow
observed = np.isfinite(qobs)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Flow duration curve")
    fdc_log = st.toggle("Log scale", value=True, key="fdc_log")
    fig = go.Figure()
    exceedance, flows = cm.flow_duration(qobs)
    fig.add_trace(go.Scatter(x=exceedance, y=flows, mode="lines", name="Observed",
                             line=dict(color=ink["obs"], width=2)))
    for res in results:
        exceedance, flows = cm.flow_duration(Q[res["best"]][observed])
        fig.add_trace(go.Scatter(x=exceedance, y=flows, mode="lines", name=res["id"],
                                 line=dict(color=_color(res["uid"]), width=2)))
    _style(fig, 460, "Discharge (mm/day)", fdc_log, xtitle="Exceedance probability (%)")
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Q-Q plot")
    qq_log = st.toggle("Log scale", value=True, key="qq_log")
    fig = go.Figure()
    quantiles = [cm.qq_quantiles(Q[res["best"]], qobs) for res in results]
    for res, (q_obs, q_sim) in zip(results, quantiles):
        fig.add_trace(go.Scatter(
            x=q_obs, y=q_sim, mode="markers", name=res["id"],
            marker=dict(color=_color(res["uid"]), size=8, line=dict(width=0)),
            customdata=np.linspace(0, 100, q_obs.size),
            hovertemplate="P%{customdata:.0f}: obs %{x:.3g}, sim %{y:.3g}<extra>%{fullData.name}</extra>"))
    values = np.concatenate([np.concatenate(pair) for pair in quantiles])
    values = values[values > 0] if qq_log else values
    lo, hi = values.min(), values.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                             line=dict(color=ink["muted"], width=1, dash="dash"), hoverinfo="skip"))
    _style(fig, 460, "Simulated quantile (mm/day)", qq_log, xtitle="Observed quantile (mm/day)")
    fig.update_xaxes(type="log" if qq_log else "linear")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
st.caption("Both use only the days with an observed flow. The Q-Q plot compares the 0, 1, ..., "
           "100th percentiles: points above the 1:1 line are flows the simulation overestimates.")

with st.expander("Parameters of the best simulations"):
    st.dataframe(pd.DataFrame({r["id"]: run["params"].loc[r["best"]] for r in results}).T,
                 width="stretch")

failed = pd.Series([e for e in run["errors"] if e is not None], dtype=object)
if len(failed):
    with st.expander(f"{len(failed)} failed realization(s)"):
        st.dataframe(failed.value_counts().rename("count"), width="stretch")
