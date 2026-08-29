"""
RICD Part 15 addendum: shock-type observable modeling, drift-plus-jump
rather than a single homogeneous Gaussian random walk.

Confirmed by direct test this session: modeling a raw debt trajectory
(genuinely shock-type -- long stable stretches punctuated by a rare,
large escalation) as an ordinary GaussianRandomWalk understated a real,
confirmed debt spike by nearly an order of magnitude (posterior mean
~0.08 against ~0.65 computed directly from raw data). The random walk's
own homogeneous-step assumption pulled the genuine large step toward
consistency with the surrounding small ones -- the model doing exactly
what a Gaussian random walk is built to do, applied to the wrong class
of process.

This module builds the fix: a Student-t distributed step size instead
of Gaussian for a shock-type latent process. A Student-t's heavy tails
tolerate an occasional large step without requiring a discrete
jump-indicator variable per timestep, which would be far harder for
NUTS to sample efficiently than a fully continuous distribution. This
is the same jump-diffusion structure already licensed by Drive_B(j,t)
and the resolution-kick machinery (§8.8a, §8.8b) for a single-container
observable, not a new mechanism -- a single-container shock is the same
phenomenon without requiring a coupled neighbor as its source.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_shock_type_latent(name, T, sigma_drift=0.05, nu=3.0,
                              init_mu=0.3, init_sigma=0.2):
    """
    Returns a latent sequence appropriate for a shock-type observable:
    Student-t distributed steps (heavy-tailed, allowing rare large
    jumps) rather than the Gaussian steps a GaussianRandomWalk uses.

    nu (degrees of freedom) controls tail weight directly: nu=3 gives
    substantially heavier tails than a Gaussian (nu=infinity) while
    still having a well-defined variance; lower nu tolerates larger
    rare jumps at the cost of more diffuse ordinary-step behavior.
    A domain adapter with a specific, real reason to expect more or
    fewer extreme jumps should fit nu rather than leave it at this
    default.

    Returns the latent Deterministic sequence directly; the caller is
    responsible for adding the observation likelihood against whatever
    sparse real data is available, exactly as with any other latent
    process in this codebase.
    """
    init = pm.Normal.dist(init_mu, init_sigma)
    steps_raw = pm.StudentT(f"{name}_steps_raw", nu=nu, mu=0, sigma=1, shape=T - 1)
    steps = steps_raw * sigma_drift
    init_val = pm.Normal(f"{name}_init", mu=init_mu, sigma=init_sigma)
    latent = pm.Deterministic(name, pt.concatenate([[init_val], init_val + pt.cumsum(steps)]))
    return latent


def compute_spike_deterministic(latent, name="debt_spike"):
    """
    Definition 11.10.2's severity signal, computed from a genuinely
    shock-type latent sequence: largest single-step rise relative to
    the typical (median) step size. Structurally identical to the
    external, raw-data computation this feature was actually validated
    on -- the fix here is in how the latent trajectory itself is
    modeled (Student-t steps), not in how spike is derived from it.
    """
    diffs = latent[1:] - latent[:-1]
    median_idx = diffs.shape[0] // 2
    sorted_abs_diffs = pt.sort(pt.abs(diffs))
    return pm.Deterministic(name, pt.max(diffs) - sorted_abs_diffs[median_idx])
