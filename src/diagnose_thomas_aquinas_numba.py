"""
Diagnostic: does compiling the MCMC sampler through PyTensor's Numba
backend (compile_mode="NUMBA"), instead of its default C-compiled path,
avoid the reproducibility instability confirmed in
diagnose_thomas_aquinas_pinned_env.py?

The pinned-environment diagnostic ruled out environment drift as the
cause: three separate job dispatches of a byte-identical Docker image
(same pinned dependency versions, same fixed random_seed=7) still split
two ways -- two runs agreeing at rhat=1.3016/frac_high_entropy=0.0000,
one landing at rhat=3.2258/153 divergences/frac_high_entropy=1.0000.
With software pinned identical, the remaining explanation is
hardware/numerics-level: PyTensor's default path here is its own
"severely degraded" non-BLAS-linked fallback (see the warning printed
on every single run in this entire investigation), which is exactly
the kind of unoptimized, less-tested math most likely to behave
differently across different underlying CPU hardware.

PyTensor's own warning names Numba and JAX as alternative backends that
"perform their own BLAS optimizations" instead of routing through that
fragile fallback. This diagnostic tests Numba specifically: a
different compiled code path, not just a different BLAS library, so if
it stays stable across separate dispatches where the default C path
did not, that's real evidence the instability lives specifically in
PyTensor's default fallback path rather than being an unavoidable
property of this model or this institution's data.

ONE evaluation per dispatch, not several replicates -- replicates
within a single job are guaranteed to agree with each other regardless
of anything, since random_seed=7 is fixed (confirmed by the extreme-
precision diagnostic's three identical replicates). The real test is
running THIS workflow several separate times and comparing results
across those separate dispatches, exactly how the pinned-environment
diagnostic was actually evaluated.
"""

from score_institution import compute_features_for_institution

UNITID = "124292"
NAME = "Thomas Aquinas College"


def main():
    features = compute_features_for_institution(UNITID, NAME, cores=1, compile_mode="NUMBA")
    if features is None:
        print("Diagnostic inconclusive: returned None (insufficient live data).")
        return
    print(f"\nfrac_high_entropy: {features.frac_high_entropy:.4f}")
    print("Compare this run's CONVERGENCE line above and this "
          "frac_high_entropy value against OTHER SEPARATE dispatches "
          "of this same workflow.")


if __name__ == "__main__":
    main()
