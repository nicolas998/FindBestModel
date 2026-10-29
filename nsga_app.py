"""
Streamlit app to calibrate a MARRMoT model with NSGA-II against one to three
composite performance metrics (CPMs).

Pick a USGS gauge, a model, the potential-ET method, the scored years and a
spin-up in days (optionally narrower parameter ranges), then one to three
CPMs from the presets or from the configurations saved by `cpm_app.py` in
`cpm_configs/`. NSGA-II (`nsga_calibration.py`) searches for the parameter
sets that trade those CPMs off best. The results show the Pareto fronts in
objective space (with three CPMs, also the front of every pair of them), and
the hydrograph, flow duration curve and Q-Q plot of the front members.

Every run is saved to `nsga_calib/<date>_usgs<gauge>_m<model>/` and can be
loaded again from the sidebar.

Usage:

    streamlit run nsga_app.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import composite_metric as cm
import get_usgs_gauge_data as usgs
import marrmot_models as marrmot
import nsga_calibration as nc
from model_parameter_space import PARAM_SPACE, free_parameters, space_from_table

DEFAULT_MODEL = 11  # Collie River Basin 3 (CRB3)
DEFAULT_OBJECTIVES = ["Timing only", "Distribution only"]
MAX_LINES = 40  # front members drawn per front in the series plots

# Categorical slots 1-5 of the reference palette, light and dark steps (same
# as cpm_app.py). Fronts take slots 1-4 in front_definitions order, the
# highlighted evaluation slot 5.
SLOTS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
INK = {
    "light": {"obs": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"obs": "#f0efec", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
DASHES = ["solid", "dash", "dot"]

cm.skip_main_in_workers()
st.set_page_config(page_title="NSGA-II Calibration", page_icon=":material/tune:", layout="wide")
ss = st.session_state


def _theme():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def _clean_log(text):
    """Keeps the last state of lines rewritten with carriage returns."""
    return "\n".join(line.split("\r")[-1] for line in text.splitlines())


def _weights_text(weights):
    used = [(cm.COMPONENTS[k][0], w) for k, w in weights.items() if w > 0]
    return " · ".join(f"{label} {w:g}" for label, w in used)


def _objective_choices():
    """{key: objective}: uploaded configurations first, then the ones saved
    in cpm_configs/, then the presets that are not there under the same
    weights."""
    choices = dict(ss.uploaded_cpms)
    for path, record in cm.library():
        try:
            config = cm.parse_configs(json.dumps(record))[0]
        except ValueError:
            continue
        choices.setdefault(config["id"], {**config, "source": path.name})
    for name, weights in cm.PRESETS.items():
        cid = cm.config_id(weights)
        choices.setdefault(cid, {"id": cid, "name": name, "source": "preset",
                                 "weights": {k: float(weights.get(k, 0.0)) for k in cm.COMPONENT_KEYS}})
    return choices


def _upload_cpms():
    """Adds the uploaded CPM files to the choices and selects them while
    fewer than three CPMs are picked."""
    for file in ss[f"cpm_upload_{ss.upload_n}"] or []:
        try:
            configs = cm.parse_configs(file.getvalue().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            ss.upload_messages.append(("error", f"{file.name} is not a CPM configuration file: {exc}"))
            continue
        for config in configs:
            ss.uploaded_cpms[config["id"]] = {**config, "source": f"upload: {file.name}"}
            if config["id"] in ss.objectives:
                continue
            if len(ss.objectives) < 3:
                ss.objectives.append(config["id"])
            else:
                ss.upload_messages.append(("warning", f"{config['id']} added to the list but not "
                                                      "selected: three CPMs are already picked"))
    ss.upload_n += 1  # a fresh uploader, so the same file can be uploaded again


ss.setdefault("uploaded_cpms", {})
ss.setdefault("upload_messages", [])
ss.setdefault("upload_n", 0)
ss.setdefault("objectives", [cm.config_id(cm.PRESETS[name]) for name in DEFAULT_OBJECTIVES])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Watershed and period")
    gauge = st.text_input("USGS gauge", "02249500")
    c1, c2 = st.columns(2)
    start_year = c1.number_input("First year", 1982, 2100, 2016)
    end_year = c2.number_input("Last year", 1982, 2100, 2020)
    spinup_days = st.number_input("Spin-up (days, not scored)", 0, 3650, 365, step=30,
                                  help="The simulation starts this many days before January 1st "
                                       "of the first year.")
    precip_source = st.selectbox("Precipitation", ["aorc", "nasa"],
                                 format_func={"aorc": "AORC, basin average",
                                              "nasa": "NASA POWER, at the gauge"}.get)
    et_method = st.selectbox("Potential ET", usgs.ET_METHODS, index=usgs.ET_METHODS.index("Turc"))

    st.header("Model")
    models = sorted(marrmot.MODEL_REGISTRY)
    model = st.selectbox("MARRMoT model", models, index=models.index(DEFAULT_MODEL),
                         format_func=lambda m: f"{m:02d} · {marrmot.MODEL_REGISTRY[m]['name']}")
    with st.expander("Parameter search ranges (optional)"):
        st.caption("NSGA-II starts searching between low and high. With dynamic ranges (NSGA-II "
                   "section) a range can widen up to min/max allowed: by default maximum "
                   "storages up to 4000 mm, every other parameter only up to its default range. "
                   "Fractions multiply the parameter they refer to; derived parameters are "
                   "computed from the others.")
        ranges = st.data_editor(nc.range_table(model), key=f"ranges_{model}", hide_index=True,
                                disabled=["parameter", "sampled as"], width="stretch")
    n_free = len(free_parameters(PARAM_SPACE[model]))

    st.header("Objectives")
    st.file_uploader("Upload CPM configuration(s) (JSON)", type="json", accept_multiple_files=True,
                     key=f"cpm_upload_{ss.upload_n}", on_change=_upload_cpms,
                     help="Files saved or downloaded with the CPM explorer (cpm_app.py): one "
                          "configuration, a list of them, or {\"configs\": [...]}.")
    for kind, text in ss.upload_messages:
        getattr(st, kind)(text)
    ss.upload_messages = []
    choices = _objective_choices()
    ss.objectives = [i for i in ss.objectives if i in choices]  # e.g. a deleted library file
    picked = st.multiselect(
        "CPMs to maximize (1–3)", list(choices), key="objectives", max_selections=3,
        format_func=lambda i: f"{i} · {choices[i]['name']}" + {
            "preset": " (preset)", "upload": " (uploaded)"}.get(choices[i]["source"].split(":")[0], ""),
        help="Uploaded configurations, the ones saved with the CPM explorer (cpm_app.py) in "
             "cpm_configs/, and the presets.")
    for i in picked:
        st.caption(f"**{i}** ({choices[i]['source']}): {_weights_text(choices[i]['weights'])}")

    st.header("NSGA-II")
    c1, c2 = st.columns(2)
    pop_size = c1.number_input("Calibrations per generation", 4, 500, 50, step=2,
                               help="Population size: parameter sets simulated per generation.")
    generations = c2.number_input("Generations", 1, 1000, 30)
    c1, c2 = st.columns(2)
    crossover_prob = c1.number_input("Crossover probability", 0.0, 1.0, 0.9, step=0.05,
                                     help="Chance that a pair of parents is recombined (SBX).")
    crossover_eta = c2.number_input("Crossover index ηc", 1.0, 100.0, 15.0, step=1.0,
                                    help="SBX distribution index: larger keeps children closer "
                                         "to their parents.")
    c1, c2 = st.columns(2)
    mutation_prob = c1.number_input("Mutation rate", 0.0, 1.0, value=None, step=0.01,
                                    key=f"mutation_{model}", placeholder=f"1/{n_free} = {1 / n_free:.3f}",
                                    help="Chance that each parameter of a child is mutated. "
                                         f"Empty = 1 / number of parameters (1/{n_free}).")
    mutation_eta = c2.number_input("Mutation index ηm", 1.0, 100.0, 20.0, step=1.0,
                                   help="Polynomial-mutation distribution index: larger makes "
                                        "smaller steps.")
    c1, c2 = st.columns(2)
    seed = c1.number_input("Seed", 0, 1_000_000, 1)
    cpus = os.cpu_count() or 1
    n_procs = c2.number_input("Processes", 1, cpus, min(4, cpus),
                              help="Simulations run in parallel in this many processes, started "
                                   "once for the whole calibration (a few seconds on Windows).")
    st.caption(f"{pop_size * (generations + 1):,} simulations: {pop_size} initial + "
               f"{pop_size} × {generations} generations.")

    adapt = nc.ADAPT_DEFAULTS
    adapt_ranges = st.toggle("Dynamic search ranges", value=True,
                             help="Widen a parameter's search range when the Pareto front "
                                  "crowds against one of its bounds, up to the min/max allowed "
                                  "in the parameter table.")
    if adapt_ranges:
        c1, c2 = st.columns(2)
        adapt_every = c1.number_input("Check every (generations)", 1, 100, adapt["adapt_every"])
        adapt_step = c2.number_input("Widen by (% of range)", 5, 200, int(100 * adapt["adapt_step"]),
                                     step=5, help="How far a bound moves each time.")
        c1, c2 = st.columns(2)
        adapt_edge = c1.number_input("Edge zone (% of range)", 1, 25, int(100 * adapt["adapt_edge"]),
                                     help="How close to a bound a member counts as at the bound.")
        adapt_share = c2.number_input("Front at the edge (%)", 5, 100, int(100 * adapt["adapt_share"]),
                                      step=5, help="Share of the front in the edge zone that "
                                                   "makes the bound move.")
    run_clicked = st.button("Calibrate", type="primary", width="stretch", disabled=not picked)

    st.header("Saved calibrations")
    saved = nc.list_runs()
    saved_pick = st.selectbox(
        "nsga_calib/", [str(p) for p, _ in saved], index=None,
        placeholder="nsga_calib/ is empty" if not saved else "Choose a run",
        format_func=lambda p: next(
            f"{os.path.basename(p)} · {', '.join(o['id'] for o in s['objectives'])}"
            + ("" if s.get("status") == "complete" else f" ({s.get('status')})")
            for q, s in saved if str(q) == p))
    load_clicked = st.button("Load", width="stretch", disabled=saved_pick is None)


st.title("NSGA-II calibration")
st.caption("Calibrate a MARRMoT model against up to three composite performance metrics and "
           "see the trade-offs between them.")

if run_clicked:
    gauge = gauge.strip()
    if not gauge.isdigit():
        st.error("The USGS gauge must be a numeric site number, e.g. 02249500")
    elif end_year < start_year:
        st.error("The last year must not be before the first year")
    else:
        settings = {
            "gauge": gauge, "start_year": int(start_year), "end_year": int(end_year),
            "spinup_days": int(spinup_days), "precip_source": precip_source,
            "et_method": et_method, "model": int(model), "pop_size": int(pop_size),
            "generations": int(generations), "crossover_prob": float(crossover_prob),
            "crossover_eta": float(crossover_eta),
            "mutation_prob": None if mutation_prob is None else float(mutation_prob),
            "mutation_eta": float(mutation_eta), "seed": int(seed), "n_procs": int(n_procs),
            "adapt_ranges": adapt_ranges,
        }
        if adapt_ranges:
            settings.update(adapt_every=int(adapt_every), adapt_step=adapt_step / 100,
                            adapt_edge=adapt_edge / 100, adapt_share=adapt_share / 100)
        objectives = [{k: choices[i][k] for k in ("id", "name", "weights")} for i in picked]
        try:
            space = space_from_table(model, ranges)
            limits = nc.limits_from_table(model, ranges, space)
            with st.status("Preparing the gauge data...", expanded=True) as status:
                data, log = nc.prepare_data(gauge, settings["start_year"], settings["end_year"],
                                            settings["spinup_days"], et_method, model, precip_source)
                st.code(_clean_log(log) or "(no output)", language=None)
                status.update(label=f"Calibrating model {model} with NSGA-II...")
                st.caption("Changing a control while this runs stops it. Every finished "
                           "generation is already saved in nsga_calib/ and can be loaded.")
                bar = st.progress(0.0)
                t0 = time.time()

                def progress(generation, row):
                    best = " · ".join(f"{o['id']} {row['best ' + o['id']]:.3f}" for o in objectives)
                    bar.progress((generation + 1) / (settings["generations"] + 1),
                                 text=f"Generation {generation}/{settings['generations']} · "
                                      f"best {best}")

                ss.run = nc.calibrate(settings, data, space, objectives, limits, progress=progress)
                status.update(label=f"{len(ss.run['evaluations'])} simulations in "
                                    f"{time.time() - t0:.0f} s, saved to "
                                    f"nsga_calib/{ss.run['path'].name}",
                              state="complete", expanded=False)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

if load_clicked or ("run" in st.query_params and "run" not in ss):
    # ?run=<folder in nsga_calib/> opens that run directly
    saved_pick = saved_pick if load_clicked else str(nc.OUT_DIR / st.query_params["run"])
    try:
        with st.spinner("Loading (an interrupted run is finished first: its fronts are simulated)..."):
            ss.run = nc.load_run(saved_pick)
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

run = ss.get("run")
if run is None:
    st.info("Pick a gauge, a model and one to three CPMs in the sidebar and press **Calibrate**, "
            "or load a saved calibration. Downloaded data is cached in `data/`.")
    st.stop()

theme = _theme()
ink, slots = INK[theme], SLOTS[theme]
settings, evaluations, series = run["settings"], run["evaluations"], run["series"]
objectives = settings["objectives"]
ids = [o["id"] for o in objectives]
fronts = nc.front_definitions(objectives)
front_color = {key: slots[i] for i, key in enumerate(fronts)}
highlight_color = slots[4]
dates = pd.to_datetime(series.index)
qobs = series["observed"].to_numpy()
ok = evaluations[ids].notna().all(axis=1)
members = {key: evaluations.index[evaluations[f"front_{key}"]] for key in fronts}

st.header(f"USGS {settings['gauge']} · model {settings['model']:02d} {settings['model_name']}")
mutation = settings["mutation_prob"]
st.caption(
    f"{settings['start_year']}–{settings['end_year']} scored after a {settings['spinup_days']}-day "
    f"spin-up (from {settings['simulation_start']}), rain {settings['precip_source'].upper()}, "
    f"ET {settings['et_method']}. NSGA-II: {settings['pop_size']} per generation × "
    f"{settings['generations']} generations, crossover {settings['crossover_prob']:g} "
    f"(ηc {settings['crossover_eta']:g}), mutation {mutation:.3g} (ηm {settings['mutation_eta']:g}), "
    f"seed {settings['seed']}. "
    + (f"Dynamic search ranges: {len(settings['range_changes'])} bound(s) widened. "
       if settings.get("adapt_ranges") else "")
    + f"Saved in `nsga_calib/{run['path'].name}`.")
if settings.get("status") == "interrupted":
    st.warning(f"This run was interrupted after generation {int(evaluations['generation'].max())} "
               f"of {settings['generations']}; the fronts use the evaluations it finished.")

c = st.columns(2 + len(fronts))
c[0].metric("Simulations", f"{len(evaluations):,}")
c[1].metric("Failed", f"{int((~ok).sum()):,}")
for col, (key, (label, _)) in zip(c[2:], fronts.items()):
    col.metric(f"On front · {label}", len(members[key]))

st.dataframe(pd.DataFrame([{"CPM": o["id"], "name": o["name"], "weights": _weights_text(o["weights"]),
                            "best": evaluations[o["id"]].max()} for o in objectives]).set_index("CPM"),
             width="stretch", column_config={"best": st.column_config.NumberColumn(format="%.3f")})


def _style(fig, height, ytitle=None, ylog=False, xtitle=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"],
                     title_text=xtitle)
    fig.update_yaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"],
                     title_text=ytitle, type="log" if ylog else "linear")
    return fig


def _hover(rows):
    """customdata + hovertemplate showing an evaluation and its CPMs."""
    data = np.column_stack([rows.index, rows["generation"], rows[ids]])
    text = "<br>".join(f"{i}: %{{customdata[{j + 2}]:.3f}}" for j, i in enumerate(ids))
    return data, f"#%{{customdata[0]}} (generation %{{customdata[1]}})<br>{text}<extra></extra>"


def _member_label(e):
    row = evaluations.loc[e]
    return f"#{e} · " + " · ".join(f"{i} {row[i]:.3f}" for i in ids)


# Selection shared by every plot below
c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
shown = c1.multiselect("Fronts shown", list(fronts), format_func=lambda k: fronts[k][0],
                       default=[k for k in fronts if k != nc.ALL] or [nc.ALL],
                       key=f"shown_{run['path'].name}")
candidates = sorted(set().union(*(members[k] for k in shown))) if shown else []
highlight = c2.selectbox("Highlight one front member", candidates, index=None,
                         format_func=_member_label, key=f"hl_{run['path'].name}",
                         placeholder="None")

# Objective space
st.subheader("Objective space")
if len(ids) == 1:
    st.caption("With a single CPM NSGA-II works as a single-objective genetic algorithm: the "
               "front is the best simulation.")
else:
    front_values = evaluations.loc[evaluations.filter(like="front_").any(axis=1), ids]
    floor = st.number_input(
        "Lower axis limit (CPM)", -100.0, 0.9,
        max(-1.0, float(np.floor(front_values.min().min() * 10) / 10 - 0.1)), step=0.1,
        key=f"floor_{run['path'].name}",
        help="Bad simulations, and the far ends of a front, can have very negative CPMs that "
             "squeeze the rest of the plot. Lower this to see them.")
    cloud = evaluations[ok & (evaluations[ids] >= floor).all(axis=1)]
    top = evaluations[ids].max() + 0.03
    cut = int((front_values < floor).any(axis=1).sum())
    pairs = [(fronts[k][1], k) for k in fronts if k != nc.ALL] or [(ids, nc.ALL)]
    tab_2d, tab_3d = st.tabs(["Pairs", "3-D"]) if len(ids) == 3 else (st.container(), None)
    with tab_2d:
        fig = make_subplots(rows=1, cols=len(pairs), horizontal_spacing=0.07)
        for col, ((a, b), key) in enumerate(pairs, 1):
            first = col == 1
            data, template = _hover(cloud)
            fig.add_trace(go.Scattergl(
                x=cloud[a], y=cloud[b], mode="markers", name="Simulations", legendgroup="cloud",
                showlegend=first, marker=dict(color=ink["muted"], size=5, opacity=0.35),
                customdata=data, hovertemplate=template), row=1, col=col)
            if key != nc.ALL:
                rows = evaluations.loc[members[nc.ALL]]
                data, template = _hover(rows)
                fig.add_trace(go.Scatter(
                    x=rows[a], y=rows[b], mode="markers", name=fronts[nc.ALL][0],
                    legendgroup=nc.ALL, showlegend=first,
                    marker=dict(color=front_color[nc.ALL], size=9, symbol="circle-open",
                                line=dict(width=2)),
                    customdata=data, hovertemplate=template), row=1, col=col)
            rows = evaluations.loc[members[key]].sort_values(a)
            data, template = _hover(rows)
            fig.add_trace(go.Scatter(
                x=rows[a], y=rows[b], mode="lines+markers", name=fronts[key][0], legendgroup=key,
                line=dict(color=front_color[key], width=2, shape="vh"),
                marker=dict(color=front_color[key], size=8),
                customdata=data, hovertemplate=template), row=1, col=col)
            if highlight is not None:
                row = evaluations.loc[[highlight]]
                fig.add_trace(go.Scatter(
                    x=row[a], y=row[b], mode="markers", name=f"#{highlight}", legendgroup="hl",
                    showlegend=first, hoverinfo="skip",
                    marker=dict(color=highlight_color, size=18, symbol="circle-open",
                                line=dict(width=3))), row=1, col=col)
        _style(fig, 440)
        for col, ((a, b), _) in enumerate(pairs, 1):
            fig.update_xaxes(title_text=a, range=[floor, top[a]], row=1, col=col)
            fig.update_yaxes(title_text=b, range=[floor, top[b]], row=1, col=col)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each point is one simulation; up and right is better on both CPMs. "
                   + ("Every panel draws the front of its pair of CPMs (a stepped line) and, as "
                      "open circles, the front over all three CPMs. " if len(ids) == 3 else "")
                   + (f"{cut} front member(s) have a CPM below the lower axis limit and are "
                      "outside the axes." if cut else ""))
    if tab_3d is not None:
        with tab_3d:
            fig = go.Figure()
            data, template = _hover(cloud)
            fig.add_trace(go.Scatter3d(
                x=cloud[ids[0]], y=cloud[ids[1]], z=cloud[ids[2]], mode="markers",
                name="Simulations", marker=dict(color=ink["muted"], size=2, opacity=0.35),
                customdata=data, hovertemplate=template))
            rows = evaluations.loc[members[nc.ALL]]
            data, template = _hover(rows)
            fig.add_trace(go.Scatter3d(
                x=rows[ids[0]], y=rows[ids[1]], z=rows[ids[2]], mode="markers",
                name=fronts[nc.ALL][0], marker=dict(color=front_color[nc.ALL], size=4),
                customdata=data, hovertemplate=template))
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=30, b=0),
                              scene=dict(xaxis_title=ids[0], yaxis_title=ids[1], zaxis_title=ids[2]),
                              legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
            st.plotly_chart(fig, width="stretch")


def _drawn(key):
    """Members of a front drawn in the series plots: all of them, or
    MAX_LINES spread along the first objective."""
    rows = evaluations.loc[members[key]].sort_values(fronts[key][1][0])
    if len(rows) > MAX_LINES:
        rows = rows.iloc[np.linspace(0, len(rows) - 1, MAX_LINES).round().astype(int)]
    return list(rows.index)


drawn = {key: _drawn(key) for key in shown}
if any(len(members[k]) > MAX_LINES for k in shown):
    st.caption(f"Fronts with more than {MAX_LINES} members are drawn with {MAX_LINES} of them, "
               "spread along the first CPM.")


def _series_traces(fig, make_trace):
    """One legend entry per front; `make_trace(member, **style)` builds a trace."""
    for key in shown:
        for n, e in enumerate(drawn[key]):
            fig.add_trace(make_trace(e, name=fronts[key][0], legendgroup=key, showlegend=n == 0,
                                     line=dict(color=front_color[key], width=1), opacity=0.55))
    if highlight is not None:
        fig.add_trace(make_trace(highlight, name=f"#{highlight}", legendgroup="hl",
                                 line=dict(color=highlight_color, width=2.5), opacity=1.0))


# Hydrograph
st.subheader("Hydrograph")
ylog = st.toggle("Log scale", value=True, key="hydro_log")
fig = go.Figure()
_series_traces(fig, lambda e, **style: go.Scattergl(
    x=dates, y=series[str(e)], mode="lines", hoverinfo="skip", **style))
fig.add_trace(go.Scattergl(x=dates, y=qobs, mode="lines", name="Observed",
                           line=dict(color=ink["obs"], width=1.5)))
_style(fig, 480, "Discharge (mm/day)", ylog)
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# Flow duration curve and Q-Q plot, on the days with an observed flow
observed = np.isfinite(qobs)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Flow duration curve")
    fdc_log = st.toggle("Log scale", value=True, key="fdc_log")
    fig = go.Figure()

    def fdc_trace(e, **style):
        exceedance, flows = cm.flow_duration(series[str(e)].to_numpy()[observed])
        return go.Scattergl(x=exceedance, y=flows, mode="lines", hoverinfo="skip", **style)

    _series_traces(fig, fdc_trace)
    exceedance, flows = cm.flow_duration(qobs)
    fig.add_trace(go.Scattergl(x=exceedance, y=flows, mode="lines", name="Observed",
                               line=dict(color=ink["obs"], width=2)))
    _style(fig, 460, "Discharge (mm/day)", fdc_log, xtitle="Exceedance probability (%)")
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Q-Q plot")
    qq_log = st.toggle("Log scale", value=True, key="qq_log")
    fig = go.Figure()
    quantiles = {}

    def qq_trace(e, **style):
        quantiles[e] = cm.qq_quantiles(series[str(e)].to_numpy(), qobs)
        q_obs, q_sim = quantiles[e]
        return go.Scatter(x=q_obs, y=q_sim, mode="lines", hoverinfo="skip", **style)

    _series_traces(fig, qq_trace)
    values = np.concatenate([np.concatenate(pair) for pair in quantiles.values()]
                            or [qobs[observed]])
    values = values[values > 0] if qq_log else values
    lo, hi = values.min(), values.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                             line=dict(color=ink["muted"], width=1, dash="dash"), hoverinfo="skip"))
    _style(fig, 460, "Simulated quantile (mm/day)", qq_log, xtitle="Observed quantile (mm/day)")
    fig.update_xaxes(type="log" if qq_log else "linear")
    st.plotly_chart(fig, width="stretch")
st.caption("Both use only the days with an observed flow. The Q-Q plot compares the 0, 1, ..., "
           "100th percentiles: above the 1:1 line the simulation overestimates those flows.")

# Parameter distributions: one panel per parameter, one row per front
st.subheader("Parameter distributions")
param_names = marrmot.MODEL_REGISTRY[settings["model"]]["param_names"]
initial = {r["parameter"]: r for r in settings["parameter_ranges"]}
# Runs from before dynamic ranges have no final ranges: they never changed
search = {r["parameter"]: r for r in settings.get("parameter_ranges_final", settings["parameter_ranges"])}
with_all = st.toggle("Show every simulation as a reference", value=True, key="params_all")
groups = [(fronts[k][0], evaluations.loc[members[k]], front_color[k]) for k in shown]
if with_all:
    groups.append(("All simulations", evaluations[ok], ink["muted"]))
if not groups:
    st.info("Select at least one front above to see its parameters.")
else:
    n_cols = 4
    n_rows = -(-len(param_names) // n_cols)
    panel_height = 70 + 24 * len(groups)
    height = n_rows * panel_height + 40
    fig = make_subplots(
        rows=n_rows, cols=n_cols, horizontal_spacing=0.04, vertical_spacing=60 / height,
        subplot_titles=[p if search[p]["sampled as"] == "range" else f"{p} ({search[p]['sampled as']})"
                        for p in param_names])
    for n, p in enumerate(param_names):
        row, col = n // n_cols + 1, n % n_cols + 1
        # Added bottom-up: the reference sits at the bottom, the fronts above it in legend order
        for label, rows, color in reversed(groups):
            reference = label == "All simulations"
            fig.add_trace(go.Box(
                x=rows[p], name=label, orientation="h", legendgroup=label, showlegend=n == 0,
                line=dict(color=color, width=1.5), fillcolor="rgba(0,0,0,0)",
                boxpoints=False if reference else "all", jitter=0.5, pointpos=0,
                marker=dict(color=color, size=5), hoveron="boxes" if reference else "points",
                text=[f"#{e}" for e in rows.index],
                hovertemplate=None if reference else f"%{{text}}: %{{x:.4g}}<extra>{label}</extra>"),
                row=row, col=col)
        if search[p]["sampled as"] == "range":
            lo, hi = search[p]["low"], search[p]["high"]
        else:
            lo, hi = evaluations.loc[ok, p].min(), evaluations.loc[ok, p].max()
        pad = 0.03 * (hi - lo or 1.0)
        fig.update_xaxes(range=[lo - pad, hi + pad], row=row, col=col)
        if search[p]["sampled as"] == "range":
            for side in ("low", "high"):
                if initial[p][side] != search[p][side]:  # where the search started
                    fig.add_vline(x=initial[p][side], row=row, col=col,
                                  line=dict(color=ink["muted"], width=1.5, dash="dot"))
        if highlight is not None:
            fig.add_vline(x=evaluations.loc[highlight, p], row=row, col=col,
                          line=dict(color=highlight_color, width=2))
    _style(fig, height)
    fig.update_yaxes(showticklabels=False)
    # Legend above the first row's panel titles
    fig.update_layout(legend=dict(traceorder="reversed", y=1 + 34 / (height - 90)),
                      margin=dict(t=80), hovermode="closest")
    fig.update_annotations(font_size=13)
    st.plotly_chart(fig, width="stretch")
    widened = bool(settings.get("range_changes"))
    st.caption("Each row is a front (in legend order, top to bottom): a box from the 25th to the "
               "75th percentile with the median, and every member as a dot. Parameters searched "
               "over a range span exactly that range"
               + (" (after widening; a dotted line marks a bound where the search started)"
                  if widened else "")
               + ", so members piled against an edge suggest widening it; the others (fractions "
                 "and derived) span the values simulated."
               + (f" The vertical line is #{highlight}." if highlight is not None else ""))

if settings.get("adapt_ranges"):
    changes = pd.DataFrame(settings["range_changes"])
    with st.expander(f"Search range changes ({len(changes)})"):
        if changes.empty:
            st.caption("No bound was widened: the front never crowded against a bound that "
                       "could still move.")
        else:
            st.dataframe(changes, hide_index=True, width="stretch")
        final = pd.DataFrame(settings["parameter_ranges_final"])
        limits = final["parameter"].map(lambda p: settings["parameter_limits"].get(p, [None, None]))
        final["min allowed"], final["max allowed"] = limits.str[0], limits.str[1]
        st.dataframe(final, hide_index=True, width="stretch")
        st.caption(f"Final search ranges. Checked every {settings['adapt_every']} generations: a "
                   f"bound moved by {settings['adapt_step']:.0%} of its range when at least "
                   f"{settings['adapt_share']:.0%} of the population's front was within "
                   f"{settings['adapt_edge']:.0%} of it.")

# Convergence
st.subheader("Convergence")
history = run["history"]
fig = go.Figure()
for i, dash in zip(ids, DASHES):
    fig.add_trace(go.Scatter(x=history["generation"], y=history[f"best {i}"], mode="lines",
                             name=i, line=dict(color=ink["obs"], width=2, dash=dash)))
    fig.add_annotation(x=history["generation"].iloc[-1], y=history[f"best {i}"].iloc[-1], text=i,
                       showarrow=False, xanchor="left", xshift=6, font=dict(color=ink["muted"]))
_style(fig, 320, "Best CPM so far", xtitle="Generation")
fig.update_layout(hovermode="x unified", margin=dict(r=110))
st.plotly_chart(fig, width="stretch")

# Front members
st.subheader("Front members")
on_any = evaluations.filter(like="front_").any(axis=1)
table = evaluations[on_any].copy()
table.insert(1, "fronts", [", ".join(fronts[k][0] for k in fronts if row[f"front_{k}"])
                           for _, row in table.iterrows()])
table = table[["generation", "fronts", *ids, "KGE", "NSE_raw", *param_names]]
st.dataframe(table, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in [*ids, "KGE", "NSE_raw"]})
st.download_button("Download front members (CSV)", table.to_csv(),
                   f"{run['path'].name}_fronts.csv", mime="text/csv", icon=":material/download:")
st.caption(f"Every simulation, with its parameters, CPMs, components and front flags, is in "
           f"`nsga_calib/{run['path'].name}/evaluations.csv`.")

failed = evaluations.loc[~ok, "error"].dropna()
if len(failed):
    with st.expander(f"{len(failed)} failed simulation(s)"):
        st.dataframe(failed.value_counts().rename("count"), width="stretch")
