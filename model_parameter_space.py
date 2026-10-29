"""
Parameter sampling ranges for all 46 MARRMoT models in `marrmot_models.py`.

The registry in `marrmot_models.py` quotes ranges from the MARRMoT catalogue
(Table S3), but ~60 parameters are listed there as "not specified", and a few
threshold parameters are documented as fractions while the Python port uses
them as absolute storages (e.g. `Sfc` in models 3, 4, 9, 11, 12, 16). This
module fills those gaps so every model can be sampled automatically:

- a plain `(lo, hi)` tuple is sampled uniformly (Latin hypercube);
- `Frac(ref, lo, hi)` is sampled as a fraction of another parameter (or of a
  function of the other parameters), so thresholds always stay below the
  capacity they belong to (e.g. field capacity < maximum storage);
- `Derived(fn)` is not sampled at all, it is computed from the others
  (e.g. SACRAMENTO's PBASE).

Ranges follow the registry where it gives one. Where it does not, they follow
MARRMoT's own conventions (rates [0, 1] d^-1, storages [1, 2000] mm,
interception [0, 5] mm, ...). Edit them freely: they are a starting point,
not a calibrated prior.
"""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd

import marrmot_models as marrmot


@dataclass(frozen=True)
class Frac:
    """Value = U(lo, hi) * base, where base is another parameter's value
    (`ref` as a name) or a function of the sampled parameters."""
    ref: Union[str, Callable]
    lo: float
    hi: float


@dataclass(frozen=True)
class Derived:
    """Value computed from the other (already sampled) parameters."""
    fn: Callable


STORE = (1, 2000)        # generic maximum storage [mm]
RATE = (0, 1)            # generic linear rate coefficient [1/d]
FRACTION = (0.05, 0.95)  # generic fraction [-]
T_THRESH = (-3, 5)       # snow/rain threshold temperature [degC]
DDF = (0, 20)            # degree-day factor [mm/degC/d]
INTERCEPTION = (0, 5)    # interception capacity / loss [mm, mm/d]
TANK_THRESH = (0, 200)   # Tank-model outlet heights [mm]

# How far a calibration that widens its search ranges may take a maximum
# storage (see `default_limits`).
STORE_LIMIT = 4000  # [mm]

# (a * max(0, S - Sfc)) ** b form used by models 4, 11 and 16 in this port:
# `a` has to be small or the flux explodes for b > 1.
SUBSURFACE_A = (0.001, 0.1)
SUBSURFACE_B = (1, 3)


PARAM_SPACE = {
    1: {'Smax': STORE},
    2: {'Dw': INTERCEPTION, 'Sw_max': STORE, 'beta_w': (0, 10), 'Kw': RATE},
    3: {'Smax': STORE, 'Sfc': Frac('Smax', *FRACTION), 'a': RATE, 'M': FRACTION},
    4: {'Smax': STORE, 'Sfc': Frac('Smax', *FRACTION), 'M': FRACTION,
        'a': SUBSURFACE_A, 'b': SUBSURFACE_B, 'tc_bf': RATE},
    5: {'lp': STORE, 'd': STORE, 'p': (0, 10), 'alpha': (0, 1),
        'tau_q': (1, 700), 'tau_s': (1, 700), 'tau_d': (0, 119)},
    6: {'Tt': T_THRESH, 'ddf': DDF, 'Smax': STORE, 'tc': RATE},
    7: {'x1': STORE, 'x2': (-10, 15), 'x3': (1, 300), 'x4': (1, 15)},
    8: {'alpha_ei': (0, 1), 'M': FRACTION, 'S_max': STORE, 'fc': FRACTION,
        'alpha_ss': RATE},
    9: {'S_b': STORE, 'S_fc': Frac('S_b', *FRACTION), 'M': FRACTION,
        'a': (1, 50), 'b': (1, 5), 'r': (0, 1)},
    10: {'S_b': STORE, 'phi': FRACTION, 'fc': FRACTION, 'r': (0, 1),
         'c': RATE, 'd': (1, 5)},
    11: {'S_max': STORE, 'S_fc': Frac('S_max', *FRACTION), 'a': SUBSURFACE_A,
         'M': FRACTION, 'b': SUBSURFACE_B, 'lam': (0, 1)},
    12: {'T_t': T_THRESH, 'ddf': DDF, 'S_max': STORE,
         'S_fc': Frac('S_max', *FRACTION), 't_c_in': RATE, 't_c_bf': RATE},
    13: {'D_w': INTERCEPTION, 'S_h_max': STORE, 'beta_h': (0, 10), 'a': (0, 1),
         'T_h': (1, 120), 'C_param': (0, 4), 'K_h': RATE},
    14: {'S_UZmax': STORE, 'S_t': FRACTION, 'K_d': RATE, 'q0': (0.1, 200),
         'f': (0, 1), 'chi': (1, 7.5), 'phi': (0.1, 5)},
    15: {'F_max': (0, 200), 'D_p': INTERCEPTION, 'S_umax': STORE, 'lp': FRACTION,
         'p': (0, 1), 'T_p': (1, 120), 'C_rate': (0, 4), 'K_p': RATE},
    16: {'I_max': INTERCEPTION, 'S_max': STORE, 'S_fc': Frac('S_max', *FRACTION),
         'M': FRACTION, 'a': SUBSURFACE_A, 'b': SUBSURFACE_B, 't_c_bf': RATE,
         'd_uh': (1, 120)},
    17: {'S_max': STORE, 'phi': (0, 1), 'gamma': (0, 1), 'k1': RATE},
    18: {'INSC': INTERCEPTION, 'COEFF': (0, 600), 'SQ': (0, 15), 'SMSC': STORE,
         'SUB': (0, 1), 'CRAK': (0, 1), 'K': RATE},
    19: {'S_b': STORE, 'phi': FRACTION, 'fc': FRACTION, 'alpha_SS': RATE,
         'beta_SS': (1, 5), 'K_deep': RATE, 'alpha_BF': RATE, 'beta_BF': (1, 5)},
    20: {'C': RATE, 'NDC': FRACTION, 'S_max': STORE, 'E_max': (0, 20),
         'F_rate': (0, 200), 'B': (0, 1), 'DPF': RATE, 'SDR_max': (1, 300)},
    21: {'UR_max': STORE, 'beta': (0, 10), 'D': (0, 1), 'Perc_max': (0, 20),
         'Lp': FRACTION, 'Nlag_f': (1, 5), 'Nlag_s': (1, 15), 'Kf': RATE, 'Ks': RATE},
    22: {'I_bar': INTERCEPTION, 'I_delta': (0, 1), 'I_s': (1, 365),
         'S_sm_max': STORE, 'b': (0, 10), 'k1': RATE, 'c1': (1, 5),
         'S_gw_max': STORE, 'k2': RATE, 'c2': (1, 5)},
    23: {'alpha_f': (0, 200), 'beta_f': (0, 5), 'B_max': STORE, 'F_max': STORE,
         'alpha_c': (0, 5), 'beta_c': (0, 1), 'A_max': STORE,
         'A_min': Frac('A_max', 0.01, 0.99), 'alpha_ss': (0, 5), 'beta_ss': (0, 1),
         'c': (0, 200), 'alpha_g': (-5, 0), 'beta_g': (0, 1), 'gamma_f': (0, 1),
         'delta_f': (0, 10), 't_d': RATE, 'alpha_b': (0, 1), 'beta_b': (0.1, 200),
         'gamma_a': (0, 1), 'delta_a': (0, 10), 'alpha_a': (0.1, 200),
         'beta_a': (1, 5), 'gamma_b': (0, 1), 'delta_b': (0, 10)},
    24: {'S_b1': STORE, 't_w': RATE, 't_u': RATE, 'S_e': STORE, 't_c': RATE},
    # k2 multiplies S**2 (units mm^-1 d^-1), so it is kept well below 1.
    25: {'phi': (0, 1), 'rc': STORE, 'gamma': (0, 1), 'k1': RATE,
         'c_a': (0, 5), 'k2': (0, 0.1)},
    26: {'Imax': INTERCEPTION, 'URmax': STORE, 'beta': (0, 10), 'D': (0, 1),
         'Percmax': (0, 20), 'Lp': FRACTION, 'Nlag_f': (1, 5), 'Nlag_s': (1, 15),
         'Kf': RATE, 'Ks': RATE},
    27: {'A0': RATE, 'A1': RATE, 'A2': RATE, 't1': TANK_THRESH, 't2': TANK_THRESH,
         'B0': RATE, 'B1': RATE, 't3': TANK_THRESH, 'C0': RATE, 'C1': RATE,
         't4': TANK_THRESH, 'D1': RATE},
    # |a| < 0.5 keeps (0.5 - a) ** (1 - b) away from 0 ** negative.
    28: {'Aim': (0, 1), 'a': (-0.49, 0.49), 'b': (0, 10), 'Wmax': STORE,
         'LM': Frac('Wmax', *FRACTION), 'c': (0, 1), 'Smax': STORE, 'Ex': (0, 10),
         'kI': RATE, 'kG': RATE, 'cI': RATE, 'cG': RATE},
    29: {'Smax': STORE, 'b': (0, 10), 'a': (0, 1), 'kf': RATE, 'ks': RATE},
    30: {'Tcrit': T_THRESH, 'ddf': DDF, 'Sb1': STORE, 'tw': RATE, 'tu': RATE,
         'Se': STORE, 'tc': RATE},
    31: {'T_crit': T_THRESH, 'ddf': DDF, 'S_b1': STORE, 't_w': RATE,
         'S_b2': STORE, 't_u': RATE, 'S_e': STORE, 't_c': RATE},
    32: {'T_crit': T_THRESH, 'ddf': DDF, 'S_b1': STORE, 't_w': RATE,
         'I_alpha': (0, 1), 'I_s': (1, 365), 'S_b2': STORE, 't_u': RATE,
         'S_e': STORE, 't_c': RATE},
    # PBASE is MARRMoT's derived percolation demand, not a free parameter.
    33: {'PCTIM': (0, 1), 'UZTWM': STORE, 'UZFWM': STORE, 'k_uz': RATE,
         'PBASE': Derived(lambda p: p['LZFWPM'] * p['k_lzp'] + p['LZFWSM'] * p['k_lzs']),
         'ZPERC': (0, 100), 'REXP': (0, 10), 'LZTWM': STORE, 'LZFWPM': STORE,
         'LZFWSM': STORE, 'PFREE': (0, 1), 'k_lzp': RATE, 'k_lzs': RATE},
    34: {'TT': T_THRESH, 'ddf': DDF, 'Imax': INTERCEPTION, 'URmax': STORE,
         'beta': (0, 10), 'Lp': FRACTION, 'Percmax': (0, 20), 'D': (0, 1),
         'Nlag_f': (1, 5), 'Nlag_s': (1, 15), 'Kf': RATE, 'Ks': RATE},
    35: {'Tcrit': T_THRESH, 'ddf': DDF, 'Sb1': STORE, 'tw': RATE,
         'Ialpha': (0, 1), 'Is': (1, 365), 'Tmin': (-10, 0), 'Tmax': (1, 20),
         'Sb2': STORE, 'tu': RATE, 'Se': STORE, 'tc': RATE},
    36: {'INSC': INTERCEPTION, 'COEFF': (0, 600), 'SQ': (0, 15), 'SMSC': STORE,
         'SUB': (0, 1), 'CRAK': (0, 1), 'EM': (0, 20), 'DSC': (0, 50),
         'ADS': (0, 1), 'MD': (0.99, 1), 'VCOND': (0, 0.5), 'DLEV': (-10, 10),
         'k1': RATE, 'k2': RATE, 'k3': (0, 100)},
    37: {'TT': T_THRESH, 'TTI': (0, 17), 'CFR': (0, 1), 'CFMAX': DDF,
         'TTM': (-3, 3), 'WHC': (0, 1), 'CFLUX': (0, 4), 'FC': STORE,
         'LP': FRACTION, 'beta': (0, 10), 'K0': RATE, 'alpha': (1, 5),
         'c': (0, 20), 'K1': RATE, 'MAXBAS': (1, 120)},
    38: {'sm1': STORE, 'sm2': STORE, 'k1': (0, 4), 'k2': (0, 4),
         'A0': RATE, 'A1': RATE, 'A2': RATE, 't1': TANK_THRESH, 't2': TANK_THRESH,
         'B0': RATE, 'B1': RATE, 't3': TANK_THRESH, 'C0': RATE, 'C1': RATE,
         't4': TANK_THRESH, 'D1': RATE},
    # c1 sits inside exp(c1 * D); keep it small enough not to overflow.
    39: {'Smax': INTERCEPTION, 'cmax': (0, 1), 'c0': (0, 1), 'c1': (0, 0.5),
         'ce': (0, 1), 'Dsurp': STORE, 'kd': RATE, 'gamma_d': (1, 5),
         'qp_max': (0, 20), 'kg': RATE, 'tau': (1, 120), 'Sbf': (1, 300),
         'kcr': RATE, 'gamma_cr': (1, 5), 'kor': RATE, 'gamma_or': (1, 5)},
    40: {'H': (0, 1), 'Y': (0, 200), 'Smax': STORE, 'C': (0, 1), 'G': (0, 1),
         'KG': RATE, 'N': (1, 10), 'K': (1, 120)},
    41: {'Cs': DDF, 'Cif': RATE, 'Lstar': STORE, 'CL1': (0, 0.99),
         'Ustar': (1, 100), 'Cof': RATE, 'CL2': (0, 0.99), 'K0': RATE,
         'K1': RATE, 'Kb': RATE},
    42: {'C': (0, 1), 'I1max': INTERCEPTION, 'alpha': (0, 1),
         'I2max': Frac('I1max', 0.01, 0.99), 'kin': RATE, 'D50': STORE,
         'D16': Frac('D50', 0.01, 0.99), 'Sbc': STORE, 'kb': RATE, 'pb': (1, 5),
         'kh': RATE, 'kc': RATE},
    43: {'fice': (0, 1), 'T0': T_THRESH, 'asnow': DDF, 'Tm': (-3, 3), 'ks': RATE,
         'aice': DDF, 'ki': RATE, 'A': STORE, 'x': (0, 10), 'y': (0, 5),
         'ksl': RATE, 'beta': (0, 1)},
    # sw (wilting point) < sm (field capacity) < Smax, all absolute mm.
    44: {'rho': INTERCEPTION, 'Ts': T_THRESH, 'Tm': (-3, 3), 'as_': DDF,
         'af': (0, 1), 'Gmax': (0, 2), 'theta': (0, 1), 'phi': (0, 200),
         'Smax': STORE, 'sw': Frac('Smax', 0.05, 0.5), 'sm': Frac('Smax', 0.5, 0.95),
         'Ksat': RATE, 'c': (1, 5), 'Lmax': (0, 20), 'kf': RATE, 'ks': RATE},
    45: {'TT': T_THRESH, 'ddf': DDF, 'alpha': (0, 1), 'beta': (0, 1),
         'STOR': INTERCEPTION, 'RETIP': (0, 50), 'SCN': Frac('SCX', 0, 1),
         'SCX': (0, 1), 'REMX': Frac('SMAX', *FRACTION), 'SMAX': STORE,
         'c_gw': (0, 20), 'RESMAX': (1, 300), 'k1': RATE, 'k2': (1, 5),
         'k3': RATE, 'k4': RATE, 'k5': RATE, 'k6': RATE},
    # f_ap + f_as <= 1 so the impermeable fraction is never negative.
    46: {'f_ap': (0, 1), 'f_dp': (0.01, 0.99), 'd_p': STORE, 'c_q': RATE,
         'd_1': (0, 1), 'f_as': Frac(lambda p: 1 - p['f_ap'], 0, 1),
         'f_ds': (0.01, 0.99), 'd_s': STORE, 'd_2': (0, 1), 'c_xq': RATE,
         'c_xs': RATE, 'c_u': RATE},
}


def check_parameter_space():
    """Raise if any model's parameter names don't match MODEL_REGISTRY."""
    problems = []
    for model_num, info in marrmot.MODEL_REGISTRY.items():
        space = PARAM_SPACE.get(model_num)
        if space is None:
            problems.append(f"model {model_num}: no parameter space defined")
            continue
        expected, defined = set(info['param_names']), set(space)
        if expected != defined:
            problems.append(
                f"model {model_num}: missing {sorted(expected - defined)}, "
                f"unknown {sorted(defined - expected)}"
            )
    if problems:
        raise ValueError("Parameter space does not match MODEL_REGISTRY:\n  " + "\n  ".join(problems))


def free_parameters(space):
    """Names of the parameters that are sampled (everything but Derived)."""
    return [name for name, spec in space.items() if not isinstance(spec, Derived)]


def spec_bounds(spec):
    """(low, high) of a range or of a fraction."""
    return (spec[0], spec[1]) if isinstance(spec, tuple) else (spec.lo, spec.hi)


def with_bounds(spec, lo, hi):
    """The same range or fraction with new bounds."""
    return (lo, hi) if isinstance(spec, tuple) else Frac(spec.ref, lo, hi)


def default_limits(model_num):
    """
    {name: (min, max)}: how far a calibration that widens its search ranges
    (nsga_calibration.py) may take each sampled parameter by default.

    Maximum storages (the generic STORE range) may grow up to STORE_LIMIT.
    Every other parameter stays within its default range: rates, fractions,
    exponents, temperatures and days of the year have physical or numerical
    limits (e.g. a field capacity equal to the maximum storage divides by
    zero in some fluxes) that no general rule captures.
    """
    space = PARAM_SPACE[model_num]
    return {name: (STORE[0], STORE_LIMIT) if space[name] == STORE else spec_bounds(space[name])
            for name in free_parameters(space)}


def latin_hypercube(n, k, rng):
    """(n, k) Latin-hypercube sample of the unit cube."""
    edges = np.linspace(0.0, 1.0, n + 1)
    unit = np.empty((n, k))
    for j in range(k):
        pts = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(pts)
        unit[:, j] = pts
    return unit


def sample_parameters(model_num, n, seed=None, space=None):
    """
    Latin-hypercube sample of `n` parameter sets for `model_num` (same scheme
    as `run_param_sweep` in Example 5).

    `space` replaces `PARAM_SPACE[model_num]` (same layout), e.g. to sample
    narrower ranges than the defaults.

    Returns:
        pd.DataFrame: one row per realization, columns in the order of
            `MODEL_REGISTRY[model_num]['param_names']`.
    """
    space = PARAM_SPACE[model_num] if space is None else space
    unit = latin_hypercube(n, len(free_parameters(space)), np.random.default_rng(seed))
    return params_from_unit(model_num, unit, space)


def params_from_unit(model_num, unit, space=None):
    """
    Parameter sets from points of the unit cube, one column per entry of
    `free_parameters(space)`: 0 is the low end of a range (or fraction), 1 the
    high end. Used by `sample_parameters` and by the NSGA-II calibration,
    which searches in this cube so thresholds always stay below their
    capacity.

    Returns:
        pd.DataFrame: as `sample_parameters`.
    """
    space = PARAM_SPACE[model_num] if space is None else space
    sampled = free_parameters(space)
    unit = np.atleast_2d(unit)

    # Plain ranges first, so Frac/Derived can reference them.
    table = pd.DataFrame({
        name: space[name][0] + unit[:, j] * (space[name][1] - space[name][0])
        for j, name in enumerate(sampled)
        if isinstance(space[name], tuple)
    })
    for j, name in enumerate(sampled):
        spec = space[name]
        if isinstance(spec, Frac):
            base = table[spec.ref] if isinstance(spec.ref, str) else spec.ref(table)
            table[name] = base * (spec.lo + unit[:, j] * (spec.hi - spec.lo))
    for name, spec in space.items():
        if isinstance(spec, Derived):
            table[name] = spec.fn(table)

    return table[marrmot.MODEL_REGISTRY[model_num]['param_names']]


# ---------------------------------------------------------------------------
# Editable range tables (used by the Streamlit apps)
# ---------------------------------------------------------------------------

def space_table(model_num, space=None):
    """The parameter space as a table: parameter, how it is sampled, low, high."""
    space = PARAM_SPACE[model_num] if space is None else space
    rows = []
    for name, spec in space.items():
        if isinstance(spec, tuple):
            rows.append({"parameter": name, "sampled as": "range", "low": spec[0], "high": spec[1]})
        elif isinstance(spec, Frac):
            ref = spec.ref if isinstance(spec.ref, str) else "a function of others"
            rows.append({"parameter": name, "sampled as": f"fraction of {ref}",
                         "low": spec.lo, "high": spec.hi})
        else:
            rows.append({"parameter": name, "sampled as": "derived", "low": None, "high": None})
    return pd.DataFrame(rows)


def space_from_table(model_num, table):
    """Inverse of `space_table`: the default space with the table's low/high
    values (Derived parameters are kept as they are)."""
    space = {}
    for row, (name, spec) in zip(table.itertuples(index=False), PARAM_SPACE[model_num].items()):
        if isinstance(spec, Derived):
            space[name] = spec
            continue
        lo, hi = float(row.low), float(row.high)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError(f"parameter {name}: low must be a number <= high")
        space[name] = with_bounds(spec, lo, hi)
    return space
