# Domain-Wall and Polarization-Switching Models for Wurtzite Ferroelectrics

A technical reference for implementing and comparing switching-kinetics analyses on atomistic trajectories (particularly AlN / AlScN) in ReaxKit.

## Scope and conventions

These models are **not interchangeable**: KAI, NLS, SNNG, and statistical distributions describe global switching transients; Merz law describes field-dependent rates or times; individual-column and fractal/neighbor-dependent approaches describe spatial mechanisms. A successful fit by itself does **not** prove its microscopic mechanism.

For an initially uniform specimen with a fixed set of switchable units, define the atomistically measured flipped fraction as

$$
f_{\mathrm{atoms}}(t)=\frac{N_{\mathrm{flipped}}(t)}{N_{\mathrm{switchable}}}.
$$

For a full reversal, a polarization-based proxy is

$$
f_P(t)=\frac{P(t)-P(0)}{P_{\mathrm{final}}-P(0)}.
$$

These are equal only when each unit's polarization magnitude is approximately unchanged (or the nonswitching contributions are removed). In an applied field, ionic displacement, charge redistribution, piezoelectric response, and other effects can change total polarization without changing the number of flipped units. For example, with 100 units starting at +1 each, after 20 flip to −1 and the other 80 increase to +1.1, the flipped fraction is 20%, but the polarization proxy is `(68−100)/(−100−100)=16%`.

**Use one switching event and a specified time origin for each fit.** For an initially mixed-domain slab, define a clear denominator (e.g., initially unswitched units eligible to reverse) and whether reverse-switching or transient zero-polarity units count. Report how the domain polarity was assigned and how switching persistence was checked.

## Quick comparison

| Model | Equation / representation | Parameters | Published numerical examples or typical values | Data needed | Interpretation and cautions |
|---|---|---|---|---|---|
| **KAI (Kolmogorov–Avrami–Ishibashi)** | `f(t)=1−exp[−(t/t0)^n]` | `t0` characteristic time; `n` effective Avrami exponent | Idealized 2D site-saturated: `n=2`; 2D constant nucleation: `n=3`; ideal 3D constant nucleation: `n=4`. Reports cited by Behrendt et al.: about `n=7` for AlScN and `n=11` for AlBN. Zheng et al. report `1–4` in their low-field AlScN MD and cite an AlScN experiment near `n≈2`. | `t`, `f(t)` or corrected normalized polarization; separate field/temperature metadata. | An empirical fitted `n` **may be fractional**; do not round it to an integer or interpret an anomalously high `n` as literal growth dimensionality. |
| **NLS (nucleation-limited switching)** | Weighted mixture of KAI curves over switching times, `f(t)=∫[1−exp{−(t/t0)^n}]F(log t0)d(log t0)` | Distribution center `t1`; Lorentzian half-width `w`; local exponent `n`; normalization `A` | Lorentzian distribution used by Huang et al.; commonly `n=2` in one 2D-growth implementation but not universally fixed; `A=1` when distribution is properly normalized. | `t`, `f(t)`, preferably sufficiently broad early and late time windows; optional field series. | Captures heterogeneity in characteristic switching times. Huang et al. fit AlScN experimental data using NLS; Jacques et al. found their Lorentzian implementation inadequate for the studied ZMO. |
| **SNNG (simultaneous nonlinear nucleation and growth)** | Nucleation-time integral with radial growth: `f=1−exp[−2π d v² N∞ ∫₀ᵗ αm τ^(m−1) exp(−ατ^m)(t−τ)²dτ]` | Film thickness `d`; DW velocity `v`; saturated nucleation density `N∞`; `α` timescale/rate parameter; `m` nucleation shape | Jacques et al. set `m=n_KAI−2` in their 2D-growth treatment; Zheng et al. report high-field AlScN MD fits by SNNG. There are no universal numerical `α`, `v`, `N∞`. | `t`, `f(t)`; `d`; independently measured `v` or `N∞` if individually identifying physical parameters. | `v²N∞` is coupled in the global transient. A fit cannot determine both `v` and `N∞` independently without additional constraints. |
| **Merz law** | `t0(E)=t00 exp(Ea/|E|)` or `v(E)=v0 exp(−Ea/|E|)` | Activation field `Ea`; time or velocity prefactor | Pure-AlN MD in Behrendt et al.: out-of-plane `Ea≈50 MV/cm`; in-plane `≈130 MV/cm` at 300 K and `≈240 MV/cm` at 200 K. | At multiple **constant** fields: switching times or tracked DW velocities, plus field and ideally temperature. | A field dependence, not a standalone model for the entire `f(t)` trace. Activation **field** is not an energy barrier in eV. |
| **Individual-column switching (ICS)** | Column-level switching events, represented statistically by fitted waiting-time distributions or microscopically by event probabilities | Column switching-time distribution, or conditional flipping rates depending on local environment | Jacques et al.: Gaussian-like early switching for `Zn₀.₇₃Mg₀.₂₇O`; bimodal behavior for `Zn₀.₅₉Mg₀.₄₁O`; associated high-field regime beyond conventional KAI behavior. | Globally: `t`,`f(t)` or switching current. Microscopically: column labels, polarity, neighboring states and event times. | ICS is a **mechanistic picture**, not a single universal closed-form `f(t)` equation. Separate new nuclei from growth at pre-existing walls. |
| **Gaussian switching-time distribution** | `f(t)=Φ[(t−μ)/σ] = [1+erf((t−μ)/(σ√2))]/2` | Mean `μ` (often `τ0`); standard deviation `σ` | Used by Jacques et al. for a ZMO switching transient; `μ` is the 50% point for the untruncated Gaussian. | Time and switched fraction, or switching current for fitting the derivative. | Phenomenological; a Gaussian on the whole real time axis may need onset/truncation treatment. |
| **Inverse-gamma switching-time distribution** | `p(t;α,β)=β^α t^(−α−1)exp(−β/t)/Γ(α)` for `t>0`; use its **CDF**, not its PDF, to model switched fraction. | Shape `α`; scale `β` | Jacques et al. use for slow/asymmetric components in higher-Mg ZMO; often combined with a Gaussian for bimodality. | Switching-time samples, or switching current/PDF; `f(t)` for a properly integrated/normalized CDF fit. | The cited supplemental Eq. S11 is a **PDF**. Do not accidentally fit a PDF directly to cumulative switched fraction. |
| **Fractal / neighbor-dependent column switching** | On hexagonal column lattice, `P(s_i:0→1 in Δt | k_i,E,T)=p_{k_i}(E,T,Δt)`, where `k_i=Σ_{j∈N(i)}s_j` | Nucleation probability for `k=0`; growth probabilities for `k>0`; field/temperature dependence; fractal boundary dimension `Df` | Behrendt et al.: MC `Df≈1.34`; estimate from previously published experimental images `Df≈1.29`; each column has three relevant neighboring columns in their simplified 2D representation. | Columnwise polarity maps across frames, local connectivity, event times, new nuclei, domain area/perimeter and optional fields. | This is a compact implementation-oriented abstraction of their Monte Carlo mechanism, **not a verbatim equation or uniquely reported numeric set of probabilities**. Fractal-domain coalescence can produce high apparent Avrami exponents even without time-dependent nucleation. |

## 1. KAI model

$$
 f(t)=1-\exp\left[-\left(\frac{t}{t_0}\right)^n\right].
$$

- At `t=t0`, `f=1−e^(−1)≈0.632`.
- Linearized diagnostic: `ln[−ln(1−f)] = n ln(t) − n ln(t0)`. Only take logs for `0<f<1` and `t>0`; inspect the chosen fitting window and residuals.
- For **idealized** `D`-dimensional growth: with site-saturated nucleation `n=D`, and with a constant nucleation rate `n=D+1`, subject to the geometric assumptions of classical Avrami theory.
- In actual regression `n` may be any suitable positive real number. **Fractional effective exponents are allowed.** `n≈2.6` should be reported as approximately 2.6, not rounded to 3 or described literally as 2.6-dimensional space.
- Wurtzite-nitride anomalous high `n` values can reflect departures from classical assumptions (fractal/irregular walls, coalescence, time-dependent nucleation, heterogeneous regions, changing field, etc.). Do not attribute a specific cause from a KAI fit alone.

## 2. NLS model

$$
 f(t)=\int_{-\infty}^{\infty}\left[1-\exp\left(-\left(\frac{t}{t_0}\right)^n\right)\right]F(\log t_0)\,d(\log t_0).
$$

For the Lorentzian distribution in `x=log(t0)`:

$$
 F(x)=\frac{A}{\pi}\frac{w}{[x-\log(t_1)]^2+w^2}.
$$

Specify the logarithm base consistently in fitting and reported `w`; if using normalized full-support distribution then `A=1`. NLS approaches KAI when the distribution collapses to a delta function. A fitted width in log-time is not directly an atomic DW width.

## 3. SNNG model

Using the Jacques et al. supplemental form, with nucleation-time variable `τ`:

$$
 f(t)=1-\exp\left[-2\pi d v^2 N_\infty\int_0^t \alpha m\tau^{m-1}\exp(-\alpha\tau^m)(t-\tau)^2\,d\tau\right].
$$

The normalized temporal nucleation-rate shape is

$$
 \dot g(\tau)=\alpha m\tau^{m-1}\exp(-\alpha\tau^m).
$$

**Implementation warning:** the exponential in the integrand depends on the integration variable `τ`, not `t`. Parsed text of the supplied supplement's S2 is ambiguous on this point; its S5 nucleation-rate equation explicitly uses `τ`. Check the PDF equation visually if reproducing exact source typography. The global curve constrains a combination involving `d v²N∞`; external measurements or fixed assumptions are required to separately identify `v` and `N∞`. Declare dimensions/units before fitting.

## 4. Merz law

$$
 t_0(E)=t_{00}\exp(E_a/|E|),\qquad
 v(E)=v_0\exp(-E_a/|E|).
$$

For positive field magnitudes, fit `ln(t0)` or `ln(v)` versus `1/|E|`; their slopes have magnitudes `Ea`. Fit separate propagation directions, temperature groups and switching regimes. Do **not** mix a ramped or sinusoidal field trace into a constant-field Merz regression without a justified time-dependent-rate model.

## 5. ICS and 6–7. Statistical time distributions

ICS considers the individual column as a switching unit, whether its event nucleates a new reversed domain or grows one adjacent to an existing wall. A measured global `f(t)` may be fitted to statistical CDFs, while a current transient corresponds approximately to the derivative `df/dt` after correcting for capacitive and leakage currents.

Gaussian:

$$
 F_G(t)=\tfrac12\left[1+\operatorname{erf}\left(\frac{t-\mu}{\sigma\sqrt2}\right)\right],\quad
 p_G(t)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{(t-\mu)^2}{2\sigma^2}\right].
$$

Inverse gamma (`t>0`):

$$
 p_{IG}(t)=\frac{\beta^\alpha}{\Gamma(\alpha)}t^{-\alpha-1}\exp(-\beta/t),\quad
 F_{IG}(t)=\frac{\Gamma(\alpha,\beta/t)}{\Gamma(\alpha)}.
$$

Here `Γ(α,x)` is the *upper* incomplete gamma function. A bimodal mixture, if justified by the data, could be `F(t)=w F_G(t)+(1−w)F_IG(t)` with `0≤w≤1`; this is an implementation option for the qualitative combined-distribution approach, **not an assertion that this exact mixture formula was printed in the attached paper**. Specify any lag/onset and ensure the CDF remains monotone and between 0 and 1.

## 8. Neighbor-dependent / fractal switching

In Behrendt et al.'s AlN mechanism, switching an axial column is fast relative to lateral growth; the next column's rate depends on how many of its local neighbors are already switched. A practical analysis:

1. Define one persistent polarity label per switchable Al-centered column per frame, including a policy for transient/undetermined labels.
2. Build basal-plane neighbor connectivity with periodic boundary handling as appropriate.
3. Record the first *persistent* flipping event and `k` previously switched neighbors at that event.
4. Distinguish nucleation (`k=0`, according to the chosen adjacency definition) from DW growth (`k>0`); track domain mergers.
5. Estimate conditional hazards per **eligible unswitched-column exposure time**, not just counts of observed switches: `rate_k ≈ events_k / total_exposure_time_k` for fixed `k` and suitably narrow field/temperature bins.
6. Track switched area, domain perimeter, and optional `Df` over time and measurement scale; do not infer a unique fractal dimension from one arbitrary resolution.
7. Compare reconstructed/predicted `f(t)` against KAI, NLS, and SNNG, while inspecting real-space switching patterns.

The `p_k` / hazard description and the analysis checklist above are suggested **implementations** of the source's qualitative neighbor-dependent model; they should not be described as direct author-reported numerical fits.

## Time-dependent field caution for the user's AlN trajectories

For `E(t)=Emax sin(ωt)`, a fit of a simple constant-field KAI/NLS/SNNG law across a whole half-cycle combines changes in driving force with intrinsic kinetics. Treat `t0`, `n`, or fitted widths as *effective, waveform-dependent* quantities; do not compare them directly with constant-field published parameters. Analyze separate reversal events, use a transparent onset convention, and if feasible run fixed-field trajectories from comparable initial structures for parameter comparisons. Spatial event statistics remain valuable under sinusoidal drive if conditioned on instantaneous field and history.

## ReaxKit implementation and workflows

The `switching_kinetics` package currently implements the three global switched-fraction models used in the supplied Jacques and Zheng papers: KAI, NLS, and SNNG. For SNNG, the fitted curve uses the combined prefactor $B=2\pi d v^2 N_\infty$ and the integral

$$
I(t)=\int_0^t g'(\tau)(t-\tau)^2\,d\tau,\qquad
g'(\tau)=\alpha m\tau^{m-1}\exp(-\alpha\tau^m),\qquad
f(t)=1-\exp[-B I(t)].
$$

The fitter optimizes positive parameters in log space with bounds and multiple starts. It reports standard errors from the local Jacobian, SSE, RMSE, MAE, $R^2$, AIC, corrected AIC, and BIC. Model comparison ranks by corrected AIC. The global SNNG curve identifies only $B$; $d$, $v$, and $N_\infty$ require external information to separate them.

The corresponding workflows accept CSV or Excel data as an existing switched fraction, a polarization trace normalized by its endpoints, or per-domain polarity columns converted to the fraction opposite their initial sign. Use `reaxkit fit-switching-kinetics --help` for combined comparison, or `fit-kai-switching`, `fit-nls-switching`, and `fit-snng-switching` for individual models.

Each invocation writes to a new `reaxkit_workspace/other/<command-name>/run_<timestamp>_<token>/` directory. Every model produces a best-fit plot and a residual plot; the combined command also produces a best-model overlay. Parameter sweeps conditionally refit all remaining free parameters at every requested value and add sweep curves, parameters, and metrics to the output workbook. For example, `--sweep kai.n=1:6` produces KAI curves for each integer $n$ from 1 through 6. Observations appear as markers, sweep lines progress from dark blue to dark red as the parameter increases, and the unconstrained best fit is green. Comma-separated values and `START:STOP:STEP` are also accepted, such as `--sweep snng.m=0.5:3:0.5`.

For KAI, `--fixed t0` needs no numeric value. The workflow sorts and normalizes each curve, applies a cumulative maximum to suppress small downward noise, and linearly interpolates the time at which the flipped fraction reaches $1-e^{-1}\approx0.63212$. In grouped input, every group receives its own inferred $t_0$. A numeric override such as `--fixed t0=12.4` remains available when $t_0$ was measured independently.

## Further implementation guidance

Implement these as separate analysis families, with explicit units and reproducible output:

- `global_kinetics`: calculate `f_atoms(t)` and optionally `f_P(t)`; fit KAI, NLS, SNNG, Gaussian CDF, inverse-gamma CDF and justified mixtures; report fit windows, initialization, constraints, residuals, confidence/identifiability warnings.
- `field_dependence`: obtain per-run characteristic times or direction-resolved DW velocities; fit Merz only for comparable constant-field runs (unless a time-dependent model is implemented).
- `spatial_mechanism`: column switching events; nucleation versus growth; `k`-dependent exposure-normalized hazards; wall position/velocity, area, perimeter and scale-dependent fractal measures.
- Preserve both fitted global data and atomistic assignments, and make it easy to compare `f_atoms` against `f_P` to expose nonswitching contributions.
- Avoid treating numerical differentiations of noisy `f(t)` as actual nucleation rates without accounting for growth and smoothing/uncertainty.
- Respect multiple cycles: fit each intended reversal separately; never assume the first frame of the whole trajectory is the relevant baseline for every later reversal.

## Sources

1. Xiangyu Zheng et al., **“Domain-Wall Mediated Polarization Switching in Ferroelectric AlScN: Strain Relief and Field-Dependent Dynamics”** (2026 preprint), especially Fig. 2–4 and supplementary discussion (KAI, NLS, SNNG and Merz).
2. Jiawei Huang et al., **“Atomistic Structure of Transient Switching States in Ferroelectric AlScN,”** *Physical Review Letters* **136**, 026801 (2026), DOI: 10.1103/lfdh-86x6; supplemental Eqs. 1–3 (KAI, NLS).
3. Drew Behrendt, Atanu Samanta and Andrew M. Rappe, **“Ferroelectric Fractals: Switching Mechanism of Wurtzite AlN,”** supplied 2024 arXiv manuscript, arXiv:2410.18816; especially Fig. 2 (Merz) and Fig. 4 (fractal/Monte Carlo switching).
4. Leonard Jacques et al., **“Switching kinetics in ferroelectric zinc magnesium oxide thin films,”** *Acta Materialia* **300** (2025), 121483, DOI: 10.1016/j.actamat.2025.121483; main-text Eqs. 1–3 and supplemental Eqs. S2–S11.

**Source note:** Reported parameter values above are specific to their respective compositions, temperatures, fields, and measurement/simulation settings. No universal numerical values are asserted for parameters the cited sources do not establish. The further implementation suggestions are proposed analysis methods, not claims that the source papers published those exact APIs or estimators.
