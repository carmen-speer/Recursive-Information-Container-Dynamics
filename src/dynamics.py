"""
Core deterministic RICD equations for the higher-education instantiation
(RICD-BSSM v0.1, per RICD_BSSM_v01_HigherEd spec).

Every function here is pure numpy so it can be used identically for:
  (a) synthetic data generation, and
  (b) inside the PyMC model (via pytensor-compatible ops in model.py).

Omega has 5 categories (see spec Section 1):
    0 Thriving, 1 Stable, 2 Strained, 3 Structural Crisis, 4 Reorganizing

IMPORTANT, STATED SIMPLIFICATIONS FOR v0.1 (not silently invented --
flagged here and in the accompanying README):
  - The history-dependent metric g_{C,t} = g(M_{C,t}) (RICD Sec 4.10-4.11)
    is instantiated as the IDENTITY metric for v0.1. The manuscript leaves
    the concrete functional form of g(M) adapter-supplied; choosing a
    genuine higher-ed-specific metric is deferred to a later phase. This
    means grad_{g_C,t} reduces to the ordinary gradient here.
  - O_o(P) and O_p(P) are implemented as fixed linear scoring vectors over
    Omega (a concrete adapter choice, Sec 2 of the spec), not derived from
    first principles.
"""

import numpy as np

N_OMEGA = 5
OMEGA_LABELS = ["Thriving", "Stable", "Strained", "Structural Crisis", "Reorganizing"]

# --- Adapter: O_o, O_p scoring -------------------------------------------
# O_p (operational) is an honest, instantaneous scoring of true P -- the
# harder-to-game signal, full dynamic range.
O_P_SCORE = np.array([1.00, 0.70, 0.35, 0.05, 0.02])

# O_o (official) is NOT a second instantaneous scoring of the same P.
# An earlier version used a second fixed vector re-scoring P directly;
# testing (World 1, the "stable" synthetic world) showed this creates a
# spurious global attractor wherever the two vectors happen to agree
# exactly (here, the pure "Thriving" corner, where both scored 1.0),
# pulling the gradient flow toward a degenerate corner state unrelated to
# genuine health. That is a real dynamical bug, not a coding slip, and is
# the reason O_o is instead modeled as its own separately-evolving,
# lagged tracking of true operational reality (step_O_o below) -- which
# is also the more honest mechanism for what "official messaging lags or
# suppresses reality" actually means.


def O_p(P):
    """Operational representation: scalar projection of true P (shape (...,5))."""
    return P @ O_P_SCORE


def step_O_o(O_o_prev, O_p_current, kappa_lag):
    """
    Official representation as its own lagged state, not an instantaneous
    re-scoring of P: O_o(t+1) = O_o(t) + kappa_lag*(O_p(t) - O_o(t)).
    kappa_lag near 1 -> official tracks operational reality closely
    (baseline/healthy). kappa_lag near 0 -> official is suppressed/shielded
    from tracking real change (the high-entropy / false-stability
    mechanism) -- this is now a controllable, structural knob rather than
    an emergent accident of two competing scoring vectors.
    """
    return O_o_prev + kappa_lag * (O_p_current - O_o_prev)


def project_simplex(v):
    """
    Euclidean projection of a vector onto the probability simplex Delta^4.
    Standard algorithm (Wang & Carreira-Perpinan 2013 / Duchi et al. 2008).
    v: shape (..., 5)
    """
    v = np.asarray(v, dtype=float)
    orig_shape = v.shape
    v2 = v.reshape(-1, orig_shape[-1])
    n = v2.shape[1]
    u = np.sort(v2, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1)
    idx = np.arange(1, n + 1)
    cond = u - (css - 1) / idx > 0
    rho = cond.sum(axis=1) - 1  # last True index, 0-indexed
    rho = np.clip(rho, 0, n - 1)
    theta = (css[np.arange(v2.shape[0]), rho] - 1) / (rho + 1)
    w = np.clip(v2 - theta[:, None], 0, None)
    return w.reshape(orig_shape)


def js_divergence(p, q, eps=1e-9):
    """Jensen-Shannon divergence between two categorical distributions."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def scalar_to_dummy_dist(x, spread=0.15):
    """
    Turn a scalar health score in [0,1] into a soft categorical distribution
    over Omega, peaked near the category whose O_p score is closest to x.
    Used to construct P_o, P_p, P_m as *distributions* (needed for the JS
    divergence machinery) from scalar official/operational/meta readings,
    since D_op/D_om/D_pm are defined between distributions, not scalars.
    """
    x = np.atleast_1d(x)
    centers = np.array([1.00, 0.70, 0.35, 0.05, 0.02])
    d = -0.5 * ((x[..., None] - centers) / spread) ** 2
    d = d - d.max(axis=-1, keepdims=True)
    w = np.exp(d)
    return w / w.sum(axis=-1, keepdims=True)


def divergence_potential(P, O_o_scalar, P_m_scalar):
    """
    Phi_t(P | P_m, M) ~ D_op + D_om + D_pm (spec Sec 4/5).
    P: current state, shape (...,5)
    O_o_scalar: current official reading -- an independently-evolving
                lagged state (step_O_o), NOT a function of P at this
                timestep, consistent with Sec 6.1's independence patch
                extended to the official channel's own inertia.
    P_m_scalar: current meta-representation reading, scalar in [0,1],
                likewise treated as an independently-evolving
                contemporaneous input, NOT derived from P.
    """
    P_o = scalar_to_dummy_dist(O_o_scalar)
    P_p = scalar_to_dummy_dist(O_p(P))
    P_m = scalar_to_dummy_dist(P_m_scalar)
    D_op = js_divergence(P_o, P_p)
    D_om = js_divergence(P_o, P_m)
    D_pm = js_divergence(P_p, P_m)
    return D_op + D_om + D_pm, D_op, D_om, D_pm


def grad_phi_identity_metric(P, O_o_scalar, P_m_scalar, eps=1e-4):
    """
    Numerical gradient of Phi w.r.t. P under the IDENTITY metric (v0.1
    simplification, see module docstring), holding O_o and P_m fixed as
    contemporaneous inputs (they are not functions of P). Central finite
    differences.
    """
    P = np.atleast_2d(P)
    n = P.shape[-1]
    grad = np.zeros_like(P)
    for i in range(n):
        dP = np.zeros(n)
        dP[i] = eps
        phi_plus, *_ = divergence_potential(P + dP, O_o_scalar, P_m_scalar)
        phi_minus, *_ = divergence_potential(P - dP, O_o_scalar, P_m_scalar)
        grad[..., i] = (phi_plus - phi_minus) / (2 * eps)
    return grad


def step_P(P_t, O_o_t, P_m_t, pi_f_t, gamma_F, xi_t, dt=1.0):
    """
    P_{t+1} = Proj_simplex[ P_t - dt*gamma_F*pi_f*grad Phi + xi_t ]
    (spec Sec 5, identity-metric v0.1 simplification).
    """
    grad = grad_phi_identity_metric(P_t, O_o_t, P_m_t)
    raw = P_t - dt * gamma_F * pi_f_t[..., None] * grad + xi_t
    return project_simplex(raw)


def step_S(S_t, D_t, pi_f_t, lam, dt=1.0):
    """
    S_{t+dt} = S_t*exp(-lam*dt) + (1-exp(-lam*dt))*(1-pi_f)*D_t
    -- the corrected exact recursion from RICD 6.6 (spec Sec 5).
    """
    decay = np.exp(-lam * dt)
    return S_t * decay + (1 - decay) * (1 - pi_f_t) * D_t


def step_R(R_t, E_exch_t, M_maint, W_raw_t):
    """
    Maintenance-first clipped resource update (spec Sec 5, RICD Sec 7.3).
    """
    M_maint_actual = np.minimum(M_maint, R_t)
    W_adaptive = np.minimum(W_raw_t, np.maximum(R_t - M_maint_actual, 0))
    W_t = W_adaptive + M_maint_actual
    R_next = R_t + E_exch_t - W_t
    return R_next, W_t


def step_pi_f(pi_f_t, pi_f_target, kappa_pi, eta_t):
    """Mean-reverting permeability update (spec Sec 5)."""
    return pi_f_t + kappa_pi * (pi_f_target - pi_f_t) + eta_t


def causal_normalize(series, fallback_lo=0.0, fallback_hi=1.0, min_history=3,
                      min_range=0.05, window=8):
    """
    Causal (trailing-only) min-max normalization (spec Sec 6). For the
    first `min_history` periods of a series, uses a fixed fallback range
    instead of that series' own (too-thin) history, avoiding both future
    leakage and a degenerate early range.

    `min_range` is a floor on the trailing (hi - lo) width, not just a
    check for exact degeneracy -- fixes the earlier-discovered pathology
    where a small, flat true series gets amplified into full-scale [0,1]
    swings by ordinary noise.

    `window` bounds the trailing range to the most recent `window`
    points, rather than the entire history since t=0. This is a second,
    later fix, distinct from the min_range one above: testing on real
    institution data (and confirmed on a generic synthetic series with
    no genuine crisis in it at all) showed that full-history trailing
    min-max normalization saturates to 1.0 and stays there for ANY
    persistently trending series, because a monotonically drifting
    series is always setting a new all-time high, and "distance from
    the all-time high" is trivially near-zero for a trending series at
    almost every point. A bounded window lets the effective ceiling
    itself drift with the series, so the normalized value reflects
    genuine relative severity within a recent horizon, not permanent
    distance from a single historical extremum that a trending series
    will always be close to. This remains fully causal -- only points up
    to and including t ever enter the computation -- window bounding is
    about which trailing points are used, not about looking forward.
    """
    series = np.asarray(series, dtype=float)
    out = np.zeros_like(series)
    for t in range(len(series)):
        if t < min_history:
            lo, hi = fallback_lo, fallback_hi
        else:
            win_start = max(0, t + 1 - window)
            wnd = series[win_start: t + 1]
            lo, hi = wnd.min(), wnd.max()
            if hi - lo < min_range:
                # widen symmetrically around the window's own mean rather
                # than falling back to an unrelated fixed range
                mid = (hi + lo) / 2
                lo, hi = mid - min_range / 2, mid + min_range / 2
        out[t] = np.clip((series[t] - lo) / (hi - lo + 1e-12), 0, 1)
    return out


def rolling_causal_variance(series, window):
    """
    Trailing-window (causal) variance of the FIRST DIFFERENCES of
    `series`, per spec Sec 4's sigma_o/sigma_p -- deliberately not the
    variance of the raw level.

    Testing (World 2, false-stability) surfaced a real conflation: a
    smoothly, monotonically *trending* series (official slowly lagging
    toward a declining reality) can show substantial windowed variance
    of its raw level purely from the trend, while a series that has
    already settled to a new level shows low level-variance even if it
    arrived there erratically. That is the wrong notion of "volatility"
    for regime classification, which is about how erratically a channel
    is moving, not how far it has drifted. Differencing first (standard
    practice for measuring volatility, as with financial returns rather
    than price levels) fixes this: a steady trend has small, consistent
    differences and low differenced-variance regardless of level drift;
    genuine erratic movement does not.
    """
    series = np.asarray(series, dtype=float)
    diffs = np.diff(series, prepend=series[0])
    out = np.zeros_like(series)
    for t in range(len(series)):
        lo = max(0, t - window + 1)
        chunk = diffs[lo : t + 1]
        out[t] = chunk.var() if len(chunk) > 1 else 0.0
    return out


def classify_regime(sigma_o, sigma_p, ratio_threshold=2.0, floor_fraction=0.02):
    """
    Minimal regime classification (spec Sec 4): compares causal variance
    of the official vs operational channel.
      - sigma_p >> sigma_o  -> 'high-entropy' (official-shielded)
      - sigma_o >> sigma_p  -> 'medium-entropy' (official-exposed)
      - comparable          -> 'baseline'

    Floors sigma_o and sigma_p before computing their ratio, but --
    corrected from a first attempt -- the floor is RELATIVE to each
    series' own trailing scale, not a fixed absolute constant. A fixed
    constant (tried first, e.g. 1e-4) fixed the real-institution case it
    was built for but broke synthetic World 2 entirely, flipping it
    from consistently high-entropy back to baseline, because synthetic
    data's true variances sit at a genuinely different absolute scale
    than real fitted-model data. The floor at each t is instead
    max(sigma_o[:t+1].max(), sigma_p[:t+1].max()) * floor_fraction --
    trailing-only, consistent with 6.8's causal-normalization
    requirement -- so what counts as "meaningfully near zero" scales
    with whatever this specific series' own real variance range has
    been, rather than assuming one absolute number applies everywhere.
    """
    sigma_o = np.asarray(sigma_o, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)
    regime = np.full(sigma_o.shape, "baseline", dtype=object)
    for t in range(len(sigma_o)):
        trailing_scale = max(sigma_o[: t + 1].max(), sigma_p[: t + 1].max())
        floor = floor_fraction * trailing_scale if trailing_scale > 0 else 1e-9
        so = max(sigma_o[t], floor)
        sp = max(sigma_p[t], floor)
        if sp / so > ratio_threshold:
            regime[t] = "high-entropy"
        elif so / sp > ratio_threshold:
            regime[t] = "medium-entropy"
    return regime


def hull_breach(I_t, S_tilde_t, pi_f_t, theta_hull):
    """B_t indicator (spec Sec 4 / RICD Sec 10.5a)."""
    return ((1 - I_t) + S_tilde_t) > (theta_hull * pi_f_t)


def confirmed_onset(breach_series, tau_confirm):
    """
    Collapse onset requires breach to hold for tau_confirm consecutive
    periods (spec Sec 4 / 9's persistence rule) -- a single spike is not
    onset.
    """
    breach_series = np.asarray(breach_series, dtype=bool)
    onset = np.zeros_like(breach_series)
    run = 0
    for t in range(len(breach_series)):
        run = run + 1 if breach_series[t] else 0
        if run >= tau_confirm:
            onset[t] = True
    return onset
