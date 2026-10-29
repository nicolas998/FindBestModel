"""
NSGA-II calibration of a MARRMoT model against one to three composite
performance metrics (CPMs, see `composite_metric.py`).

The search runs in the unit cube of the model's free parameters (see
`model_parameter_space.params_from_unit`), so fractions such as field
capacity / maximum storage always stay valid. Every objective is 1 - CPM,
minimized; a simulation that fails gets FAILED_OBJECTIVE on every objective.

NSGA-II (Deb et al., 2002): a Latin-hypercube initial population, then per
generation binary tournament selection on (front rank, crowding distance),
simulated binary crossover (SBX), polynomial mutation and elitist survival of
the best `pop_size` of parents + offspring.

Every evaluation is appended to <run>/evaluations.csv as soon as its
generation finishes, so an interrupted run keeps what it computed. When the
run ends, the Pareto fronts of the whole archive (every objective together
and, with three objectives, every pair) are flagged in that table, written to
pareto_fronts.csv, and the series of their members to front_series.parquet.

Output folder: nsga_calib/<YYYYmmdd-HHMMSS>_usgs<gauge>_m<model>/

    settings.json         gauge, period, model, objectives, ranges, NSGA-II setup
    evaluations.csv       every parameter set: generation, parameters,
                          CPM of each objective, components, front flags
    history.csv           per generation: best CPM so far per objective
    pareto_fronts.csv     the evaluations that are on any front
    front_series.parquet  observed flow and the flow of every front member

Used by the Streamlit app `nsga_app.py`.
"""

import contextlib
import io
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import composite_metric as cm
import find_best_model as fbm
import marrmot_models as marrmot
from model_parameter_space import free_parameters, latin_hypercube, params_from_unit, space_table

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "nsga_calib"

FAILED_OBJECTIVE = 1e10  # 1 - CPM of a failed simulation
ALL = "all"              # key of the front over every objective


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data(gauge, start_year, end_year, spinup_days, et_method, model,
                 precip_source="aorc"):
    """
    `find_best_model.prepare_data` with a spin-up given in days: the
    simulation starts `spinup_days` before January 1st of `start_year`, and
    only start_year..end_year is scored.

    Returns:
        tuple[dict, str]: (data, log printed while preparing it).
    """
    warmup_years = math.ceil(spinup_days / 365)  # every year has >= 365 days
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        data = fbm.prepare_data(gauge, start_year, end_year, warmup_years, et_method,
                                skip_snow=not fbm.needs_temperature(model),
                                precip_source=precip_source)
    if fbm.needs_temperature(model) and data["T"] is None:
        raise RuntimeError(f"Model {model} needs temperature, which is not available")

    n_warm = int((~data["eval_mask"]).sum())
    keep = slice(n_warm - spinup_days, None)
    data = {**data, **{k: data[k][keep] for k in ("P", "Ep", "t", "eval_mask")},
            "T": None if data["T"] is None else data["T"][keep]}
    first = data["dates"][0] - pd.Timedelta(days=spinup_days)
    print(f"Spin-up:       {spinup_days} day(s), simulation starts {first.date()}", file=log)
    return data, log.getvalue()


# ---------------------------------------------------------------------------
# Pareto utilities (every objective minimized)
# ---------------------------------------------------------------------------

def pareto_mask(F):
    """True for the rows of `F` (n, k) that no other row dominates."""
    F = np.asarray(F, dtype=float)
    keep = np.ones(len(F), dtype=bool)
    # In lexicographic order a point can only be dominated by earlier ones,
    # so a point still kept when it is reached is on the front.
    for i in np.lexsort(F.T[::-1]):
        if keep[i]:
            dominated = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
            keep &= ~dominated
    return keep


def non_dominated_sort(F):
    """Front rank of every row of `F` (0 = non-dominated)."""
    F = np.asarray(F, dtype=float)
    leq = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominates = leq & lt                       # [i, j]: i dominates j
    n_dominators = dominates.sum(axis=0)
    rank = np.full(len(F), -1)
    current, r = np.flatnonzero(n_dominators == 0), 0
    while current.size:
        rank[current] = r
        n_dominators = n_dominators - dominates[current].sum(axis=0)
        n_dominators[rank >= 0] = -1
        current, r = np.flatnonzero(n_dominators == 0), r + 1
    return rank


def crowding_distance(F):
    """Crowding distance of every row of `F` (one front); inf at the ends."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for column in F.T:
        order = np.argsort(column, kind="stable")
        span = column[order[-1]] - column[order[0]]
        distance[order[[0, -1]]] = np.inf
        if span > 0:
            distance[order[1:-1]] += (column[order[2:]] - column[order[:-2]]) / span
    return distance


def rank_and_crowding(F):
    rank = non_dominated_sort(F)
    crowd = np.empty(len(F))
    for r in np.unique(rank):
        members = rank == r
        crowd[members] = crowding_distance(F[members])
    return rank, crowd


# ---------------------------------------------------------------------------
# NSGA-II operators on the unit cube
# ---------------------------------------------------------------------------

def tournament(rank, crowd, n, rng):
    """`n` winners of binary tournaments: lower rank, then larger crowding."""
    a, b = rng.integers(len(rank), size=(2, n))
    better_a = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    better_a = np.where(tie, rng.random(n) < 0.5, better_a)
    return np.where(better_a, a, b)


def sbx(p1, p2, eta, prob, rng):
    """Simulated binary crossover of the rows of `p1` and `p2` (bounded to
    [0, 1]); each pair is crossed with probability `prob`, each variable of a
    crossed pair with probability 0.5."""
    n, k = p1.shape
    y1, y2 = np.minimum(p1, p2), np.maximum(p1, p2)
    cross = ((rng.random(n) < prob)[:, None] & (rng.random((n, k)) < 0.5)
             & (y2 - y1 > 1e-14))
    u = rng.random((n, k))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        def spread(beta):
            alpha = 2.0 - beta ** -(eta + 1.0)
            return np.where(u <= 1.0 / alpha, (u * alpha) ** (1.0 / (eta + 1.0)),
                            (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0)))
        c1 = 0.5 * (y1 + y2 - spread(1.0 + 2.0 * y1 / (y2 - y1)) * (y2 - y1))
        c2 = 0.5 * (y1 + y2 + spread(1.0 + 2.0 * (1.0 - y2) / (y2 - y1)) * (y2 - y1))
    c1, c2 = np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)
    swap = rng.random((n, k)) < 0.5
    c1, c2 = np.where(swap, c2, c1), np.where(swap, c1, c2)
    return np.where(cross, c1, p1), np.where(cross, c2, p2)


def polynomial_mutation(X, eta, prob, rng):
    """Bounded polynomial mutation: every variable mutates with probability
    `prob`."""
    mutate = rng.random(X.shape) < prob
    u = rng.random(X.shape)
    power = 1.0 / (eta + 1.0)
    low = u < 0.5
    delta = np.where(low, X, 1.0 - X)  # distance to the bound on the side it moves to
    with np.errstate(invalid="ignore"):
        dq_low = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta) ** (eta + 1.0)) ** power - 1.0
        dq_high = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta) ** (eta + 1.0)) ** power
    mutated = np.clip(X + np.where(low, dq_low, dq_high), 0.0, 1.0)
    return np.where(mutate, mutated, X)


def make_offspring(X, F, n, crossover_prob, crossover_eta, mutation_prob, mutation_eta, rng):
    rank, crowd = rank_and_crowding(F)
    m = 2 * math.ceil(n / 2)
    parents = X[tournament(rank, crowd, m, rng)]
    c1, c2 = sbx(parents[0::2], parents[1::2], crossover_eta, crossover_prob, rng)
    children = np.vstack([c1, c2])[:n]
    return polynomial_mutation(children, mutation_eta, mutation_prob, rng)


def survivors(F, n):
    """Indices of the `n` best rows of `F`: whole fronts first, the last one
    cut by crowding distance."""
    rank, crowd = rank_and_crowding(F)
    return np.lexsort((-crowd, rank))[:n]


# ---------------------------------------------------------------------------
# Objectives and fronts
# ---------------------------------------------------------------------------

def check_objectives(objectives, qobs):
    """Raises if an objective has no score at this gauge (e.g. only weights
    on percentiles that are 0 here)."""
    perfect = cm.compute_components(qobs[None, np.isfinite(qobs)], qobs[np.isfinite(qobs)])
    for obj in objectives:
        try:
            cm.cpm_score(perfect, obj["weights"])
        except ValueError as exc:
            raise ValueError(f"{obj['id']} ({obj['name']}): {exc}") from None


def score(Q, qobs, objectives):
    """(components table, CPM matrix (R, K)); NaN for failed simulations."""
    components = cm.compute_components(Q, qobs)
    ok = np.isfinite(Q).all(axis=1)
    cpm = np.full((len(Q), len(objectives)), np.nan)
    if ok.any():
        for j, obj in enumerate(objectives):
            cpm[ok, j] = cm.cpm_score(components[ok], obj["weights"]).to_numpy()
    return components, cpm


def to_objectives(cpm):
    return np.where(np.isfinite(cpm), 1.0 - cpm, FAILED_OBJECTIVE)


def front_definitions(objectives):
    """{key: (label, objective ids)}: every objective together and, with
    three objectives, each pair."""
    ids = [o["id"] for o in objectives]
    fronts = {ALL: ("All objectives" if len(ids) > 1 else f"Best {ids[0]}", ids)}
    if len(ids) == 3:
        for a, b in itertools.combinations(ids, 2):
            fronts[f"{a}_{b}"] = (f"{a} × {b}", [a, b])
    return fronts


def front_flags(evaluations, objectives):
    """One boolean column `front_<key>` per front of `front_definitions`,
    over every successful evaluation. A child can be an unchanged copy of
    its parent: repeated CPM values are only flagged on their first
    evaluation."""
    cpm = evaluations[[o["id"] for o in objectives]]
    ok = (cpm.notna().all(axis=1) & ~cpm.duplicated()).to_numpy()
    flags = {}
    for key, (_, ids) in front_definitions(objectives).items():
        flag = np.zeros(len(evaluations), dtype=bool)
        if ok.any():
            flag[ok] = pareto_mask(1.0 - evaluations.loc[ok, ids].to_numpy())
        flags[f"front_{key}"] = flag
    return pd.DataFrame(flags, index=evaluations.index)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _run_dir(settings):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUT_DIR / f"{stamp}_usgs{settings['gauge']}_m{settings['model']:02d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_settings(path, settings):
    (path / "settings.json").write_text(json.dumps(settings, indent=2, default=cm._json_default))


def calibrate(settings, data, space, objectives, progress=None):
    """
    Runs NSGA-II and saves every result under nsga_calib/.

    Parameters:
        settings (dict): gauge, start_year, end_year, spinup_days, et_method,
            precip_source, model, pop_size, generations, crossover_prob,
            crossover_eta, mutation_prob, mutation_eta, seed, n_procs.
        data (dict): output of `prepare_data`.
        space (dict): parameter space (layout of PARAM_SPACE[model]).
        objectives (list[dict]): 1-3 CPMs as {'id', 'name', 'weights'}.
        progress (callable, optional): progress(generation, history_row)
            after every generation (0 = the initial population).

    Returns:
        dict: the run, as `load_run` returns it.
    """
    model = settings["model"]
    ids = [o["id"] for o in objectives]
    if not 1 <= len(objectives) <= 3 or len(set(ids)) < len(ids):
        raise ValueError("Pick one to three different CPMs")
    qobs = data["Qobs"]
    check_objectives(objectives, qobs)

    names = free_parameters(space)
    n_pop, n_gen = settings["pop_size"], settings["generations"]
    settings = {
        **settings,
        "model_name": marrmot.MODEL_REGISTRY[model]["name"],
        "mutation_prob": (1.0 / len(names) if settings.get("mutation_prob") is None
                          else settings["mutation_prob"]),
        "objectives": objectives,
        "parameter_ranges": space_table(model, space).to_dict("records"),
        "free_parameters": names,
        "simulation_start": str((data["dates"][0] - pd.Timedelta(days=settings["spinup_days"])).date()),
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _run_dir(settings)
    _write_settings(path, settings)

    rng = np.random.default_rng(settings["seed"])
    history, tables = [], []
    best = np.full(len(objectives), -np.inf)
    t0 = time.time()

    def evaluate(X, generation, pool):
        params = params_from_unit(model, X, space)
        Q, errors = cm.run_ensemble(model, data, params, settings["n_procs"], pool=pool)
        components, cpm = score(Q, qobs, objectives)
        start = sum(len(t) for t in tables)
        table = pd.concat([
            pd.DataFrame({"generation": generation}, index=range(len(X))),
            params.reset_index(drop=True),
            pd.DataFrame(cpm, columns=ids),
            components.reset_index(drop=True),
            pd.DataFrame({"error": errors}),
        ], axis=1)
        table.index = pd.RangeIndex(start, start + len(table), name="evaluation")
        table.to_csv(path / "evaluations.csv", mode="a", header=not tables)
        tables.append(table)

        best[:] = np.maximum(best, np.where(np.isfinite(cpm), cpm, -np.inf).max(axis=0))
        history.append({"generation": generation, "evaluations": start + len(table),
                        "failed": int(sum(e is not None for e in errors)),
                        **{f"best {i}": b if np.isfinite(b) else np.nan for i, b in zip(ids, best)},
                        "elapsed_s": round(time.time() - t0, 2)})
        pd.DataFrame(history).to_csv(path / "history.csv", index=False)
        if progress:
            progress(generation, history[-1])
        return to_objectives(cpm)

    n_procs = settings["n_procs"]
    with cm.ensemble_pool(data, n_procs) if n_procs > 1 else contextlib.nullcontext() as pool:
        X = latin_hypercube(n_pop, len(names), rng)
        F = evaluate(X, 0, pool)
        for generation in range(1, n_gen + 1):
            children = make_offspring(X, F, n_pop, settings["crossover_prob"],
                                      settings["crossover_eta"], settings["mutation_prob"],
                                      settings["mutation_eta"], rng)
            F_children = evaluate(children, generation, pool)
            X, F = np.vstack([X, children]), np.vstack([F, F_children])
            keep = survivors(F, n_pop)
            X, F = X[keep], F[keep]

    evaluations = pd.concat(tables)
    settings.update(status="complete", elapsed_s=round(time.time() - t0, 2),
                    finished_at=datetime.now().isoformat(timespec="seconds"))
    return finish_run(path, settings, evaluations, data)


def finish_run(path, settings, evaluations, data):
    """Flags the fronts, simulates their members again (the series are not
    kept during the search) and writes the remaining output files."""
    objectives = settings["objectives"]
    evaluations = evaluations.drop(columns=[c for c in evaluations if c.startswith("front_")])
    evaluations = pd.concat([evaluations, front_flags(evaluations, objectives)], axis=1)
    on_front = evaluations.filter(like="front_").any(axis=1)
    evaluations.to_csv(path / "evaluations.csv")
    evaluations[on_front].to_csv(path / "pareto_fronts.csv")

    members = evaluations[on_front]
    params = members[marrmot.MODEL_REGISTRY[settings["model"]]["param_names"]]
    Q, _ = cm.run_ensemble(settings["model"], data, params)
    series = pd.DataFrame(Q.T, index=data["dates"], columns=members.index.astype(str))
    series.insert(0, "observed", data["Qobs"])
    series.index.name = "date"
    series.to_parquet(path / "front_series.parquet")
    _write_settings(path, settings)
    return {"path": path, "settings": settings, "evaluations": evaluations, "series": series,
            "history": pd.read_csv(path / "history.csv")}


# ---------------------------------------------------------------------------
# Saved runs
# ---------------------------------------------------------------------------

def list_runs():
    """Saved runs, newest first: list of (path, settings)."""
    runs = []
    for path in sorted(OUT_DIR.glob("*/settings.json"), reverse=True):
        try:
            runs.append((path.parent, json.loads(path.read_text())))
        except (OSError, ValueError):
            continue
    return runs


def load_run(path):
    """
    A saved run: {'path', 'settings', 'evaluations', 'series', 'history'}.

    A run that was interrupted has no fronts or series yet: they are computed
    from the evaluations it saved (the gauge data is prepared again, from the
    cache in data/ when possible).
    """
    path = Path(path)
    settings = json.loads((path / "settings.json").read_text())
    if not (path / "evaluations.csv").exists():
        raise FileNotFoundError(f"{path.name} stopped before its first generation finished")
    evaluations = pd.read_csv(path / "evaluations.csv", index_col="evaluation")
    history = pd.read_csv(path / "history.csv")
    series_path = path / "front_series.parquet"
    if settings.get("status") == "complete" and series_path.exists():
        return {"path": path, "settings": settings, "evaluations": evaluations,
                "series": pd.read_parquet(series_path), "history": history}

    data, _ = prepare_data(settings["gauge"], settings["start_year"], settings["end_year"],
                           settings["spinup_days"], settings["et_method"], settings["model"],
                           settings["precip_source"])
    settings["status"] = "interrupted"
    return finish_run(path, settings, evaluations, data)

