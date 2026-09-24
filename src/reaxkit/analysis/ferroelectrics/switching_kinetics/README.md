# Ferroelectric switching kinetics

This package implements the three global switched-fraction models used in the
supplied Jacques and Zheng papers:

- KAI: `f(t) = 1 - exp(-(t/t0)^n)`.
- NLS: the KAI response integrated over a Lorentzian distribution in
  `log(t0)`, with center `log(t1)`, width `w`, exponent `n`, and optional
  normalization `A`.
- SNNG: `f(t) = 1 - exp(-B I(t))`, where
  `B = 2*pi*d*v^2*N_infinity`,
  `I(t) = integral_0^t g'(tau)(t-tau)^2 d tau`, and
  `g'(tau) = alpha*m*tau^(m-1)*exp(-alpha*tau^m)`.

The fitter optimizes positive parameters in log space with bounds and multiple
starts. It reports standard errors from the local Jacobian plus SSE, RMSE, MAE,
R-squared, AIC, corrected AIC, and BIC. Model comparison ranks by corrected AIC.
The SNNG curve identifies only `B`; `d`, `v`, and `N_infinity` require external
information to separate them.

The corresponding workflows accept CSV or Excel data as an existing switched
fraction, a polarization trace normalized by its endpoints, or per-domain
polarity columns converted to the fraction opposite its initial sign. Use
`reaxkit fit-switching-kinetics --help` for the combined comparison and
`fit-kai-switching`, `fit-nls-switching`, or `fit-snng-switching` for one model.

Each command writes every invocation to a new
`reaxkit_workspace/other/<command-name>/run_<timestamp>_<token>/` directory.
Every model produces a best-fit plot
and a residual plot; the combined command also produces a best-model overlay.
Parameter sweeps conditionally refit all remaining free parameters at every
requested value and add the sweep curves, parameters, and metrics to the output
workbook. For example, `--sweep kai.n=1:6` produces KAI curves for every integer
`n` from 1 through 6. Observations are marker-only, sweep lines progress from
dark blue to dark red with increasing parameter value, and the unconstrained
best fit is green. Comma-separated values and `START:STOP:STEP` are also
accepted, such as `--sweep snng.m=0.5:3:0.5`.

For KAI, `--fixed t0` needs no numeric value. The workflow sorts and normalizes
each curve, applies a cumulative maximum to suppress small downward noise, and
linearly interpolates the time at which the flipped fraction reaches
`1-exp(-1)`, approximately 0.63212. In grouped input, every group receives its
own inferred `t0`. A numeric override such as `--fixed t0=12.4` remains
available when `t0` was measured independently.
