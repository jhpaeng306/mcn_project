"""Run the complete bifurcation, forcing, field, and amplitude analysis.

Examples
--------
Run both requested presets with the project virtual environment::

    .venv/bin/python -m simulations.pipeline

Run a faster smoke analysis::

    .venv/bin/python -m simulations.pipeline --quick
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from .amplitude import (
    classify_wave,
    integrate_amplitudes,
    project_critical_amplitudes,
    unforced_branch_amplitudes,
    zero_state_growth_rate,
)
from .fft import (
    BifurcationPoint,
    ForcingCoefficients,
    NormalFormCoefficients,
    compute_bifurcation,
    compute_forcing_coefficients,
    compute_normal_form,
    linear_symbol,
    modal_spectrum,
)
from .simulation import Forcing, ModelParameters, SimulationResult, simulate

Array = np.ndarray
SUBPLOT_WIDTH = 5.0


def subplot_figsize(columns: int, rows: int = 1, row_height: float = 4.0) -> tuple[float, float]:
    """Return a fixed panel width and twice the former per-row height."""
    return SUBPLOT_WIDTH * columns, 2.0 * row_height * rows


def _panel_text(ax, text: str) -> None:
    """Place former title metadata inside a panel as ordinary text."""
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
    )


@dataclass(frozen=True)
class RunSettings:
    p_bounds: tuple[float, float]
    max_mode: int
    p_margin: float
    drive_amplitude: float
    grid_points: int
    total_time: float
    dt: float
    save_every: int
    initial_scale: float
    drive_weights: tuple[float, ...] | None = None
    forcing_target_growth: float | None = None
    pattern_amplitude_threshold: float = 5e-4
    temporal_frequency_multiplier: float = 2.0
    spatial_forcing_epsilon_squared_coefficient: float | None = None


def pi_ei_preset() -> tuple[ModelParameters, RunSettings]:
    """PI's Wilson-Cowan coefficients, converted to variance convention.

    PI's kernels are ``exp(-(x/s)^2)``.  A normalized Gaussian written as
    ``N(0, Sigma)`` has the same exponent when ``Sigma = s^2 / 2``.
    """
    model = ModelParameters(
        K=2,
        A=np.array([[5.4, -10.0], [10.0, -8.0]]),
        A_p=np.array([[1.0, 0.0], [0.0, 0.0]]),
        Sigma=np.array([[4.0**2 / 2.0, 8.0**2 / 2.0]] * 2),
        tau=np.array([3.0, 6.6]),
        theta=np.array([0.5, 0.3]),
        gain=np.array([4.0, 4.0]),
        length=60.0,
        names=("E", "I"),
        time_unit="PI model time",
        space_unit="PI grid unit",
        metadata={
            "preset": "PI E-I",
            "source": "wc2dn.txt and wilson_cowan_ei_bifurcation.ipynb",
            "perturbation": "A_EE = 5.4 + p",
        },
    )
    settings = RunSettings((0.0, 2.2), 12, 0.08, 0.05, 96, 700.0, 0.1, 5, 0.01)
    return model, settings


def v1_epvsst_vip_preset() -> tuple[ModelParameters, RunSettings]:
    """Canonical E/PV/SST/VIP rate-field preset for mouse V1.

    Billeh et al. (2020) provide the L2/3 connection probabilities at 75 um
    and Gaussian distance scales.  The exact nonzero mask follows the reduced
    inhibitory motif of Pfeffer et al. (2013): PV inhibits E and PV; SST
    inhibits E, PV, and VIP; VIP inhibits SST; and E excites every population.
    Campagnola et al. (2022) supports treating this as a coarse cell-class
    reduction rather than a complete physiological graph.  Connection
    probabilities do not uniquely determine Wilson-Cowan gains, so the
    nonzero magnitudes remain explicitly calibrated rate parameters.
    """
    effective_A = np.array(
        [
            [6.22009, -1.45455, -6.48944, 0.0],
            [5.09798, -0.90615, -8.04066, 0.0],
            [2.67174, 0.0, 0.0, -16.0],
            [1.68844, 0.0, -3.13584, 0.0],
        ]
    )
    billeh_scale_um = np.array(
        [
            [114.0, 95.0, 85.0, 85.0],
            [92.0, 120.0, 120.0, 120.0],
            [103.0, 120.0, 120.0, 120.0],
            [103.0, 120.0, 120.0, 120.0],
        ]
    )
    probability_75um = np.array(
        [
            [0.160, 0.411, 0.424],
            [0.395, 0.451, 0.857],
            [0.182, 0.030, 0.082],
        ]
    )
    # The surviving gains were recalibrated under the exact canonical zero
    # mask.  A uniform positive source-column direction preserves every Dale
    # sign and gives a finite-wave supercritical Hopf crossing; the baseline
    # scale 0.5 is stable and onset occurs near p=0.25.
    ap_column_factors = np.ones(4)
    pattern_forming_A_p = effective_A * ap_column_factors[None, :]
    # Billeh's exp(-r^2/s^2) parameter becomes variance s^2/2.  One model
    # spatial unit is 0.1 mm = 100 um.
    Sigma = (billeh_scale_um / 100.0) ** 2 / 2.0
    model = ModelParameters(
        K=4,
        A=0.5 * effective_A,
        A_p=pattern_forming_A_p,
        Sigma=Sigma,
        tau=np.array([10.0, 12.0, 35.0, 20.0]),
        theta=np.array([0.28483799, 0.12068148, 0.40137356, 0.7]),
        gain=np.array([3.47682761, 7.34413013, 3.74650527, 6.0]),
        length=47.6023,
        names=("E", "PV", "SST", "VIP"),
        time_unit="ms",
        space_unit="0.1 mm",
        metadata={
            "preset": "mouse V1 E/PV/SST/VIP",
            "billeh_E_PV_SST_connection_probability_75um_rows_target_columns_source": probability_75um.tolist(),
            "billeh_exp_minus_r2_over_s2_scale_um": billeh_scale_um.tolist(),
            "effective_full_scale_A": effective_A.tolist(),
            "baseline_scale": 0.5,
            "canonical_target_source_sign_mask": [
                [1, -1, -1, 0],
                [1, -1, -1, 0],
                [1, 0, 0, -1],
                [1, 0, -1, 0],
            ],
            "perturbation": (
                "A(p) = 0.5 A_effective + p A_p; A_p uses a uniform "
                "Dale-preserving source-column direction"
            ),
            "A_p_column_factors_E_PV_SST_VIP": ap_column_factors.tolist(),
            "A_p_design": (
                "spectral search under the canonical zero mask plus direct-field "
                "validation; homogeneous mode remains damped"
            ),
            "evidence_sources": [
                "Billeh et al., Neuron 2020, doi:10.1016/j.neuron.2020.01.040",
                "Pfeffer et al., Nat Neurosci 2013, doi:10.1038/nn.3446",
                "Campagnola et al., Science 2022, doi:10.1126/science.abj5861",
            ],
            "VIP_reduction": (
                "VIP->SST is the only retained VIP output; SST->VIP closes the "
                "disinhibitory loop. Zero entries encode the idealized Pfeffer "
                "motif, not proof that weak biological connections are absent."
            ),
            "forcing_direction": (
                "equal additive drive to E and PV, with no direct drive to "
                "SST or VIP; temporal forcing is "
                "growth-calibrated and the spatial component is capped at "
                "epsilon^2"
            ),
            "forcing_asymptotics": (
                "epsilon^2 = p0 - p = p_margin; physical forcing is "
                "epsilon^2 times an order-one coefficient"
            ),
            "interpretation": "probabilities/widths are data-derived; rate gains and activation parameters are calibrated hypotheses",
        },
    )
    settings = RunSettings(
        (0.0, 0.5),
        12,
        0.001,
        0.01,
        96,
        60000.0,
        0.25,
        4,
        0.0003,
        drive_weights=(1.0, 1.0, 0.0, 0.0),
        forcing_target_growth=2e-4,
        pattern_amplitude_threshold=3e-4,
        temporal_frequency_multiplier=2.0,
        # E+PV forcing couples less strongly to the resonant coefficient than
        # the former PV+VIP control direction.  Six remains an order-one
        # multiplier of epsilon^2 and preserves the matched-growth cases.
        spatial_forcing_epsilon_squared_coefficient=6.0,
    )
    return model, settings


def v1_epvsst_preset() -> tuple[ModelParameters, RunSettings]:
    """Backward-compatible name for the four-population V1 preset."""
    return v1_epvsst_vip_preset()


def presets() -> dict[str, tuple[ModelParameters, RunSettings]]:
    return {"pi_ei": pi_ei_preset(), "v1_epvsst_vip": v1_epvsst_vip_preset()}


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _jsonable(value: object) -> object:
    if isinstance(value, complex):
        return _complex(value)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [_jsonable(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _frequency_label(model: ModelParameters, critical: BifurcationPoint) -> str:
    if model.time_unit == "ms":
        return f"{critical.frequency_hz:.2f} Hz"
    return f"{critical.frequency_cycles_per_time:.5f} cycles/{model.time_unit}"


def plot_bifurcation(
    model: ModelParameters,
    settings: RunSettings,
    critical: BifurcationPoint,
    output: Path,
) -> Path:
    # Plot modal dispersion curves at a small collection of perturbation
    # values around onset.  Include p0 exactly: selecting the closest member
    # of a regular p grid can leave the nominal critical curve visibly above
    # or below zero.
    # The simulation margin can be extremely small (epsilon^2 in V1), which
    # would make the plotted curves visually coincident.  Use at least four
    # percent of the scan interval for this diagnostic comparison.
    p_step = max(
        settings.p_margin, 0.04 * (settings.p_bounds[1] - settings.p_bounds[0])
    )
    p_values = np.clip(
        critical.p0
        + p_step * np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
        settings.p_bounds[0],
        settings.p_bounds[1],
    )
    p_values = np.unique(np.append(p_values, critical.p0))
    modes, spectrum = modal_spectrum(model, p_values, max_mode=settings.max_mode)
    growth = spectrum.real.max(axis=2)
    critical_index = int(np.flatnonzero(np.isclose(p_values, critical.p0))[0])
    frequency = np.abs(spectrum[critical_index, :, 0].imag) / (2.0 * np.pi)
    if model.time_unit == "ms":
        frequency = 1000.0 * frequency
        frequency_unit = "Hz"
    else:
        frequency_unit = f"cycles/{model.time_unit}"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    color_positions = np.linspace(0.12, 0.88, p_values.size)
    for index, (p, color_position) in enumerate(zip(p_values, color_positions)):
        highlight = np.isclose(p, critical.p0)
        axes[0].plot(
            modes,
            growth[index],
            marker="o",
            markersize=5.5 if highlight else 3.5,
            color="C3" if highlight else plt.colormaps["viridis"](color_position),
            lw=2.8 if highlight else 1.25,
            zorder=4 if highlight else 2,
            label=f"p = p0 = {p:.5f}" if highlight else f"p = {p:.5f}",
        )
    axes[0].axhline(0.0, color="black", lw=1.0)
    axes[0].scatter(
        [critical.mode],
        [0.0],
        marker="*",
        s=150,
        facecolor="gold",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label=f"critical maximum (k0={critical.mode})",
    )
    axes[0].set(
        xlabel="spatial frequency number k",
        ylabel=f"max real eigenvalue (1/{model.time_unit})",
        title="Modal dispersion curves near onset",
        xticks=modes,
    )
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    colors = ["C3" if mode == critical.mode else "C0" for mode in modes]
    axes[1].bar(modes, frequency, color=colors)
    axes[1].set(
        xlabel="spatial Fourier mode k",
        ylabel=f"natural frequency ({frequency_unit})",
        title=f"Spectrum at p0; selected mode {critical.mode}",
    )
    fig.suptitle(
        f"{model.metadata.get('preset', 'model')} - omega0={critical.omega0:.5f}, {_frequency_label(model, critical)}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _drive_vector(
    model: ModelParameters, settings: RunSettings, amplitude: float
) -> Array:
    """Return a validated population forcing vector at the requested scale."""
    if settings.drive_weights is None:
        weights = np.zeros(model.K)
        weights[0] = 1.0
    else:
        weights = np.asarray(settings.drive_weights, dtype=float)
        if weights.shape != (model.K,):
            raise ValueError("drive_weights must have one entry per population")
        if not np.all(np.isfinite(weights)) or not np.any(weights):
            raise ValueError("drive_weights must be finite and nonzero")
    return float(amplitude) * weights


def _forcing_case(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    p: float,
    kind: str,
) -> tuple[Forcing, Array, Array]:
    """Build one resonant case, optionally calibrating its onset strength."""
    critical = normal.critical

    def build(
        amplitude: float, *, apply_spatial_cap: bool = True
    ) -> tuple[Forcing, Array, Array]:
        drive = _drive_vector(model, settings, amplitude)
        zero = np.zeros(model.K)
        spatial_cap = settings.spatial_forcing_epsilon_squared_coefficient
        if spatial_cap is not None and spatial_cap < 0.0:
            raise ValueError(
                "spatial_forcing_epsilon_squared_coefficient must be non-negative"
            )
        maximum_spatial_amplitude = (
            None if spatial_cap is None else spatial_cap * settings.p_margin
        )
        if kind == "spatial":
            spatial_drive = drive
            if apply_spatial_cap and maximum_spatial_amplitude is not None:
                spatial_drive = _drive_vector(
                    model, settings, min(amplitude, maximum_spatial_amplitude)
                )
            return (
                Forcing(
                    spatial_amplitude=spatial_drive,
                    spatial_mode=2 * critical.mode,
                ),
                zero,
                0.5 * spatial_drive,
            )
        if kind == "temporal":
            return (
                Forcing(
                    temporal_amplitude=drive,
                    temporal_omega=(
                        settings.temporal_frequency_multiplier * critical.omega0
                    ),
                ),
                0.5 * drive,
                zero,
            )
        if kind == "both":
            half = 0.5 * drive
            spatial_half = half
            if apply_spatial_cap and maximum_spatial_amplitude is not None:
                spatial_half = _drive_vector(
                    model,
                    settings,
                    min(0.5 * amplitude, maximum_spatial_amplitude),
                )
            return (
                Forcing(
                    spatial_amplitude=spatial_half,
                    temporal_amplitude=half,
                    spatial_mode=2 * critical.mode,
                    temporal_omega=(
                        settings.temporal_frequency_multiplier * critical.omega0
                    ),
                ),
                0.25 * drive,
                0.5 * spatial_half,
            )
        raise ValueError(f"unknown forcing kind: {kind}")

    amplitude = settings.drive_amplitude
    if settings.forcing_target_growth is not None:
        target = settings.forcing_target_growth

        def objective(value: float) -> float:
            # Calibrate the uncapped resonant case first.  The V1 spatial cap
            # is then applied to the physical S component returned below.
            _, temporal, spatial = build(value, apply_spatial_cap=False)
            coefficients = compute_forcing_coefficients(
                model,
                normal,
                temporal_harmonic=temporal,
                spatial_harmonic=spatial,
            )
            return zero_state_growth_rate(normal, p=p, forcing=coefficients) - target

        lower = 0.0
        upper = max(settings.drive_amplitude, 1e-8)
        while objective(upper) <= 0.0 and upper < 100.0:
            upper *= 2.0
        if objective(upper) <= 0.0:
            raise RuntimeError(f"could not bracket the {kind} forcing threshold")
        amplitude = brentq(objective, lower, upper, xtol=1e-12)
    return build(amplitude, apply_spatial_cap=True)


def _forcing_suite(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    p: float,
) -> dict[str, tuple[Forcing, Array, Array]]:
    zero = np.zeros(model.K)
    return {
        "no forcing": (Forcing(), zero, zero),
        "spatial only": _forcing_case(model, normal, settings, p, "spatial"),
        "temporal only": _forcing_case(model, normal, settings, p, "temporal"),
        "spatial + temporal": _forcing_case(model, normal, settings, p, "both"),
    }


def _safe_limit(values: list[Array], percentile: float = 99.0) -> float:
    concatenated = np.concatenate([np.ravel(value) for value in values])
    return max(float(np.nanpercentile(np.abs(concatenated), percentile)), 1e-8)


def _display_forcing(result: SimulationResult) -> tuple[Array, int]:
    """Select the driven population with the largest displayed forcing."""
    peaks = np.max(np.abs(result.forcing), axis=(0, 2))
    population = int(np.argmax(peaks))
    return result.forcing[:, population], population


def _zero_forcing_coefficients(model: ModelParameters) -> ForcingCoefficients:
    zero = np.zeros(model.K, dtype=complex)
    return ForcingCoefficients(zero.copy(), zero.copy(), 0.0j, 0.0j)


def _off_resonance_and_subthreshold_suite(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    p: float,
) -> tuple[dict[str, tuple[Forcing, ForcingCoefficients, str]], float]:
    """Build detuned and analytically subthreshold forcing experiments."""
    critical = normal.critical
    matched_spatial, _, _ = _forcing_case(model, normal, settings, p, "spatial")
    matched_temporal, _, _ = _forcing_case(model, normal, settings, p, "temporal")
    matched_both, _, _ = _forcing_case(model, normal, settings, p, "both")
    spatial_drive = matched_spatial.vector(matched_spatial.spatial_amplitude, model.K)
    temporal_drive = matched_temporal.vector(
        matched_temporal.temporal_amplitude, model.K
    )
    both_spatial_drive = matched_both.vector(matched_both.spatial_amplitude, model.K)
    both_temporal_drive = matched_both.vector(matched_both.temporal_amplitude, model.K)
    no_projection = _zero_forcing_coefficients(model)

    cases: dict[str, tuple[Forcing, ForcingCoefficients, str]] = {
        "detuned P: temporal frequency 1.5 omega0": (
            Forcing(
                temporal_amplitude=temporal_drive,
                temporal_omega=1.5 * critical.omega0,
            ),
            no_projection,
            "off resonance: no leading autonomous eta_t term",
        ),
        "detuned S: spatial mode 3 k0": (
            Forcing(
                spatial_amplitude=spatial_drive,
                spatial_mode=3 * critical.mode,
            ),
            no_projection,
            "off resonance: no leading autonomous eta_x term",
        ),
        "detuned P + S": (
            Forcing(
                temporal_amplitude=both_temporal_drive,
                temporal_omega=1.5 * critical.omega0,
                spatial_amplitude=both_spatial_drive,
                spatial_mode=3 * critical.mode,
            ),
            no_projection,
            "both components are off the 2:1 resonances",
        ),
    }

    # Start at one tenth of the comparison amplitude.  If any resonant case
    # is still above the parametric threshold, halve all weak amplitudes until
    # every zero-amplitude critical-mode growth rate is strictly negative.
    weak_amplitude = 0.1 * settings.drive_amplitude
    unforced_growth = zero_state_growth_rate(normal, p=p)
    if unforced_growth >= 0.0:
        raise ValueError("subthreshold forcing controls require p below onset")
    # Retain at least half of the unforced decay margin instead of stopping
    # arbitrarily close to zero, where finite-time noise can look patterned.
    required_margin = 0.5 * unforced_growth
    while True:
        weak_drive = _drive_vector(model, settings, weak_amplitude)
        weak_temporal = compute_forcing_coefficients(
            model, normal, temporal_harmonic=0.5 * weak_drive
        )
        weak_spatial = compute_forcing_coefficients(
            model, normal, spatial_harmonic=0.5 * weak_drive
        )
        weak_both = compute_forcing_coefficients(
            model,
            normal,
            temporal_harmonic=0.25 * weak_drive,
            spatial_harmonic=0.25 * weak_drive,
        )
        weak_coefficients = (weak_temporal, weak_spatial, weak_both)
        if all(
            zero_state_growth_rate(normal, p=p, forcing=coefficients) <= required_margin
            for coefficients in weak_coefficients
        ):
            break
        weak_amplitude *= 0.5
        if weak_amplitude < 1e-12:
            raise RuntimeError("could not construct a stable subthreshold forcing")

    weak_drive = _drive_vector(model, settings, weak_amplitude)
    weak_half = 0.5 * weak_drive
    cases.update(
        {
            "insufficient P: resonant 2 omega0": (
                Forcing(
                    temporal_amplitude=weak_drive,
                    temporal_omega=(
                        settings.temporal_frequency_multiplier * critical.omega0
                    ),
                ),
                weak_temporal,
                "resonant, but below the critical-mode threshold",
            ),
            "insufficient S: resonant 2 k0": (
                Forcing(
                    spatial_amplitude=weak_drive,
                    spatial_mode=2 * critical.mode,
                ),
                weak_spatial,
                "resonant, but below the critical-mode threshold",
            ),
            "insufficient P + S": (
                Forcing(
                    temporal_amplitude=weak_half,
                    temporal_omega=(
                        settings.temporal_frequency_multiplier * critical.omega0
                    ),
                    spatial_amplitude=weak_half,
                    spatial_mode=2 * critical.mode,
                ),
                weak_both,
                "combined resonant input remains below threshold",
            ),
        }
    )
    return cases, weak_amplitude


def plot_forcing_comparison(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    p: float,
    results: dict[str, tuple[SimulationResult, ForcingCoefficients]],
    output: Path,
    *,
    min_amplitude: float = 5e-4,
    late_time_window: float | None = None,
) -> Path:
    def time_mask(result: SimulationResult) -> Array:
        if late_time_window is None:
            return np.ones(result.t.size, dtype=bool)
        if late_time_window <= 0.0:
            raise ValueError("late_time_window must be positive")
        return result.t >= result.t[-1] - late_time_window

    forcing_limit = _safe_limit(
        [
            _display_forcing(item[0])[0][time_mask(item[0])]
            for item in results.values()
        ]
    )
    activity_limit = _safe_limit(
        [item[0].u[time_mask(item[0]), 0] for item in results.values()]
    )
    fig, axes = plt.subplots(
        len(results),
        3,
        figsize=subplot_figsize(3, len(results), row_height=4.0),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (label, (result, coefficients)) in enumerate(results.items()):
        forcing_vectors = (
            result.forcing_spec.temporal_amplitude,
            result.forcing_spec.spatial_amplitude,
            result.forcing_spec.traveling_amplitude,
        )
        driven = np.zeros(model.K, dtype=bool)
        for value in forcing_vectors:
            driven |= np.abs(result.forcing_spec.vector(value, model.K)) > 0.0
        drive_label = "+".join(np.asarray(model.names)[driven]) or "no"
        visible = time_mask(result)
        visible_time = result.t[visible]
        displayed_forcing, _ = _display_forcing(result)
        displayed_forcing = displayed_forcing[visible]
        extent = [result.x[0], result.x[-1], visible_time[0], visible_time[-1]]
        axes[row, 0].imshow(
            displayed_forcing,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-forcing_limit,
            vmax=forcing_limit,
        )
        axes[row, 1].imshow(
            result.u[visible, 0],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-activity_limit,
            vmax=activity_limit,
        )
        A, B = project_critical_amplitudes(result, normal.critical)
        reduced = integrate_amplitudes(
            result.t,
            normal,
            p=p,
            initial=(complex(A[0]), complex(B[0])),
            forcing=coefficients,
        )
        axes[row, 2].plot(np.abs(A), result.t, color="C0", label="direct |A|")
        axes[row, 2].plot(np.abs(B), result.t, color="C1", label="direct |B|")
        if np.any(np.isfinite(reduced.A)):
            axes[row, 2].plot(
                np.abs(reduced.A),
                result.t,
                color="C0",
                ls="--",
                alpha=0.75,
                label="normal form |A|",
            )
            axes[row, 2].plot(
                np.abs(reduced.B),
                result.t,
                color="C1",
                ls="--",
                alpha=0.75,
                label="normal form |B|",
            )
        forcing_peak = float(np.max(np.abs(result.forcing)))
        temporal_peak = float(
            np.max(
                np.abs(
                    result.forcing_spec.vector(
                        result.forcing_spec.temporal_amplitude, model.K
                    )
                )
            )
        )
        spatial_peak = float(
            np.max(
                np.abs(
                    result.forcing_spec.vector(
                        result.forcing_spec.spatial_amplitude, model.K
                    )
                )
            )
        )
        epsilon_squared = normal.critical.p0 - p
        scaled_peak = forcing_peak / epsilon_squared
        component_parts = []
        if spatial_peak > 0.0:
            component_parts.append(
                f"S={spatial_peak:.4g} = {spatial_peak / epsilon_squared:.3g} epsilon^2"
            )
        if temporal_peak > 0.0:
            component_parts.append(
                f"P={temporal_peak:.4g} = {temporal_peak / epsilon_squared:.3g} epsilon^2"
            )
        if component_parts:
            component_note = "\n" + "\n".join(component_parts)
        else:
            component_note = f"\npeak={forcing_peak:.4g} ({scaled_peak:.3g} epsilon^2)"
        input_frequency_note = ""
        response_frequency_note = ""
        if result.forcing_spec.temporal_omega > 0.0:
            frequency = measure_frequency_doubling(model, normal.critical, result)
            unit = frequency["frequency_unit"]
            input_frequency_note = (
                f"\ninput={frequency['measured_input_frequency']:.2f} {unit} "
                f"= 2 f0"
            )
            response_frequency_note = (
                f"\nmode {normal.critical.mode} response="
                f"{frequency['measured_response_frequency']:.2f} {unit} = f0; "
                f"ratio={frequency['measured_input_over_response']:.4f}"
            )
        _panel_text(
            axes[row, 0],
            f"stimulus on {drive_label}\n"
            rf"$\delta p=p-p_0={p - normal.critical.p0:+.4g}$"
            f"{component_note}",
        )
        growth = zero_state_growth_rate(normal, p=p, forcing=coefficients)
        for column in (0, 1):
            axes[row, column].set(
                xlabel=f"space ({model.space_unit})", ylabel=f"time ({model.time_unit})"
            )
        axes[row, 2].set(
            xlabel="amplitude",
            ylabel=f"time ({model.time_unit})",
            ylim=(visible_time[0], visible_time[-1]),
        )
    axes[0, 2].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _frequency_spectra(
    model: ModelParameters,
    critical: BifurcationPoint,
    result: SimulationResult,
    *,
    start_fraction: float = 0.4,
) -> tuple[Array, Array, Array, Array, Array, int]:
    """Return temporal spectra of the applied input and the critical mode."""
    if result.forcing_spec.temporal_omega <= 0.0:
        raise ValueError("frequency verification requires temporal forcing")
    start = int(start_fraction * result.t.size)
    if result.t.size - start < 8:
        raise ValueError("frequency verification requires at least eight samples")
    dt = float(np.median(np.diff(result.t)))
    temporal_drive = result.forcing_spec.vector(
        result.forcing_spec.temporal_amplitude, model.K
    )
    population = int(np.argmax(np.abs(temporal_drive)))
    input_trace = result.forcing[start:, population, 0]
    critical_coefficient = (
        np.fft.fft(result.u[start:, 0], axis=1)[:, critical.mode]
        / result.u.shape[2]
    )
    response_trace = critical_coefficient.real
    window = np.hanning(input_trace.size)
    input_centered = (input_trace - np.mean(input_trace)) * window
    response_centered = (response_trace - np.mean(response_trace)) * window
    frequencies = np.fft.rfftfreq(input_trace.size, d=dt)
    input_power = np.abs(np.fft.rfft(input_centered)) ** 2
    response_power = np.abs(np.fft.rfft(response_centered)) ** 2
    return (
        frequencies,
        input_power,
        response_power,
        input_trace,
        response_trace,
        population,
    )


def measure_frequency_doubling(
    model: ModelParameters,
    critical: BifurcationPoint,
    result: SimulationResult,
) -> dict[str, float | str]:
    """Measure the temporal input-to-critical-response frequency ratio."""
    frequencies, input_power, response_power, _, _, _ = _frequency_spectra(
        model, critical, result
    )
    expected_response = critical.omega0 / (2.0 * np.pi)
    expected_input = result.forcing_spec.temporal_omega / (2.0 * np.pi)
    input_band = (frequencies >= 0.5 * expected_input) & (
        frequencies <= 1.5 * expected_input
    )
    response_band = (frequencies >= 0.5 * expected_response) & (
        frequencies <= 1.5 * expected_response
    )
    measured_input = float(
        frequencies[input_band][np.argmax(input_power[input_band])]
    )
    measured_response = float(
        frequencies[response_band][np.argmax(response_power[response_band])]
    )
    scale = 1000.0 if model.time_unit == "ms" else 1.0
    unit = "Hz" if model.time_unit == "ms" else f"cycles/{model.time_unit}"
    return {
        "frequency_unit": unit,
        "expected_input_frequency": scale * expected_input,
        "expected_response_frequency": scale * expected_response,
        "measured_input_frequency": scale * measured_input,
        "measured_response_frequency": scale * measured_response,
        "measured_input_over_response": measured_input / measured_response,
    }


def plot_frequency_doubling_verification(
    model: ModelParameters,
    critical: BifurcationPoint,
    results: dict[str, tuple[SimulationResult, ForcingCoefficients]],
    output: Path,
) -> tuple[Path, dict[str, dict[str, float | str]]]:
    """Show that the stimulus is at 2 omega0 and the response at omega0."""
    temporal_results = {
        label: result
        for label, (result, _) in results.items()
        if result.forcing_spec.temporal_omega > 0.0
    }
    if not temporal_results:
        raise ValueError("frequency verification requires a temporal forcing case")
    diagnostics = {
        label: measure_frequency_doubling(model, critical, result)
        for label, result in temporal_results.items()
    }
    rows = len(temporal_results)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(12.5, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    response_period = 2.0 * np.pi / critical.omega0
    for row, (label, result) in enumerate(temporal_results.items()):
        (
            frequencies,
            input_power,
            response_power,
            input_trace,
            response_trace,
            population,
        ) = _frequency_spectra(model, critical, result)
        diagnostic = diagnostics[label]
        samples = max(
            8,
            int(np.ceil(6.0 * response_period / np.median(np.diff(result.t)))),
        )
        samples = min(samples, input_trace.size)
        relative_time = result.t[-samples:] - result.t[-samples]
        input_zoom = input_trace[-samples:] - np.mean(input_trace[-samples:])
        response_zoom = response_trace[-samples:] - np.mean(response_trace[-samples:])
        input_zoom /= max(float(np.max(np.abs(input_zoom))), 1e-15)
        response_zoom /= max(float(np.max(np.abs(response_zoom))), 1e-15)
        axes[row, 0].plot(
            relative_time,
            input_zoom,
            label=f"{model.names[population]} input (2 omega0)",
            lw=1.3,
        )
        axes[row, 0].plot(
            relative_time,
            response_zoom,
            label=f"E mode {critical.mode} response (omega0)",
            lw=1.7,
        )
        axes[row, 0].set(
            xlabel=f"zoomed time ({model.time_unit})",
            ylabel="normalized signal",
            title=f"{label}: six response periods",
        )
        axes[row, 0].legend(frameon=False, fontsize=8)

        scale = 1000.0 if model.time_unit == "ms" else 1.0
        shown_frequency = scale * frequencies
        input_power /= max(float(np.max(input_power)), 1e-15)
        response_power /= max(float(np.max(response_power)), 1e-15)
        upper = 1.25 * float(diagnostic["expected_input_frequency"])
        visible = shown_frequency <= upper
        axes[row, 1].plot(
            shown_frequency[visible], input_power[visible], label="input spectrum"
        )
        axes[row, 1].plot(
            shown_frequency[visible],
            response_power[visible],
            label=f"mode {critical.mode} response spectrum",
        )
        axes[row, 1].axvline(
            float(diagnostic["expected_response_frequency"]),
            color="C1",
            ls="--",
            lw=1.0,
            label="expected f0",
        )
        axes[row, 1].axvline(
            float(diagnostic["expected_input_frequency"]),
            color="C0",
            ls="--",
            lw=1.0,
            label="expected 2 f0",
        )
        axes[row, 1].set(
            xlim=(0.0, upper),
            xlabel=f"frequency ({diagnostic['frequency_unit']})",
            ylabel="normalized power",
            title=(
                f"measured input/response = "
                f"{diagnostic['measured_input_over_response']:.4f}"
            ),
        )
        axes[row, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{model.metadata.get('preset', 'model')}: temporal 2:1 resonance; "
        f"input 2 omega0, critical response omega0 (mode k0={critical.mode})"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output, diagnostics


def plot_off_resonance_and_subthreshold(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    p: float,
    output: Path,
    *,
    quick: bool,
) -> tuple[Path, list[dict[str, object]]]:
    """Simulate detuned inputs and resonant inputs below threshold."""
    cases, weak_amplitude = _off_resonance_and_subthreshold_suite(
        model, normal, settings, p
    )
    rows: list[
        tuple[
            str,
            SimulationResult,
            ForcingCoefficients,
            Array,
            Array,
            float,
            int,
            str,
        ]
    ] = []
    summaries: list[dict[str, object]] = []
    for index, (label, (forcing, coefficients, explanation)) in enumerate(
        cases.items()
    ):
        result = simulate(
            model,
            p=p,
            forcing=forcing,
            seed=23,
            grid_points=settings.grid_points,
            total_time=settings.total_time * (0.55 if quick else 1.0),
            dt=settings.dt,
            save_every=settings.save_every,
            initial_scale=settings.initial_scale,
        )
        A, B = project_critical_amplitudes(result, normal.critical)
        growth = zero_state_growth_rate(normal, p=p, forcing=coefficients)
        tail = result.u[int(0.8 * result.t.size) :, 0]
        centered = tail - tail.mean(axis=1, keepdims=True)
        power = np.mean(np.abs(np.fft.rfft(centered, axis=1)) ** 2, axis=0)
        dominant_mode = int(np.argmax(power[1:]) + 1)
        rows.append(
            (
                label,
                result,
                coefficients,
                A,
                B,
                growth,
                dominant_mode,
                explanation,
            )
        )
        late = slice(int(0.8 * A.size), None)
        summaries.append(
            {
                "label": label,
                "explanation": explanation,
                "p": p,
                "critical_zero_growth_rate": growth,
                "critical_zero_stable": growth < 0.0,
                "classification": classify_wave(
                    A,
                    B,
                    min_amplitude=settings.pattern_amplitude_threshold,
                ),
                "late_mean_abs_A": float(np.mean(np.abs(A[late]))),
                "late_mean_abs_B": float(np.mean(np.abs(B[late]))),
                "late_dominant_spatial_mode": dominant_mode,
                "forcing_peak": float(np.max(np.abs(result.forcing))),
                "forcing_peak_over_epsilon_squared": float(
                    np.max(np.abs(result.forcing)) / settings.p_margin
                ),
                "eta_t": coefficients.eta_t,
                "eta_x": coefficients.eta_x,
            }
        )

    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=subplot_figsize(3, len(rows), row_height=3.25),
        squeeze=False,
        constrained_layout=True,
    )
    shared_activity_limit = _safe_limit(
        [item[1].u[:, 0] for item in rows], percentile=99.5
    )
    for row, (
        label,
        result,
        coefficients,
        A,
        B,
        growth,
        dominant_mode,
        explanation,
    ) in enumerate(rows):
        extent = [result.x[0], result.x[-1], result.t[0], result.t[-1]]
        displayed_forcing, forcing_population = _display_forcing(result)
        forcing_limit = _safe_limit([displayed_forcing])
        axes[row, 0].imshow(
            displayed_forcing,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-forcing_limit,
            vmax=forcing_limit,
        )
        axes[row, 1].imshow(
            result.u[:, 0],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-shared_activity_limit,
            vmax=shared_activity_limit,
        )
        reduced = integrate_amplitudes(
            result.t,
            normal,
            p=p,
            initial=(complex(A[0]), complex(B[0])),
            forcing=coefficients,
        )
        axes[row, 2].plot(np.abs(A), result.t, color="C0", label="direct |A|")
        axes[row, 2].plot(np.abs(B), result.t, color="C1", label="direct |B|")
        axes[row, 2].plot(
            np.abs(reduced.A),
            result.t,
            color="C0",
            ls="--",
            alpha=0.72,
            label="normal form |A|",
        )
        axes[row, 2].plot(
            np.abs(reduced.B),
            result.t,
            color="C1",
            ls="--",
            alpha=0.72,
            label="normal form |B|",
        )
        peak = float(np.max(np.abs(result.forcing)))
        _panel_text(
            axes[row, 0],
            f"peak stimulus={peak:.4g}\n"
            rf"$\delta p=p-p_0={p - normal.critical.p0:+.4g}$",
        )
        for column in (0, 1):
            axes[row, column].set(
                xlabel=f"space ({model.space_unit})",
                ylabel=f"time ({model.time_unit})",
            )
        axes[row, 2].set(
            xlabel="critical amplitude",
            ylabel=f"time ({model.time_unit})",
            ylim=(result.t[0], result.t[-1]),
        )
    axes[0, 2].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output, summaries


def plot_pattern_verification(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    output: Path,
    *,
    quick: bool,
    target_growth: float = 0.0003,
) -> tuple[Path, dict[str, object]]:
    """Long direct-field check that the designed A_p sustains a pattern."""
    if normal.alpha.real <= 0.0:
        raise ValueError("pattern verification requires Re(alpha) > 0")
    p = normal.critical.p0 + target_growth / normal.alpha.real
    # This check is above onset at a fixed 0.001/ms growth rate, so it does not
    # need the longer slow time used by the epsilon-scaled below-onset runs.
    total_time = 6000.0 if quick else 10000.0
    result = simulate(
        model,
        p=p,
        seed=31,
        grid_points=settings.grid_points,
        total_time=total_time,
        dt=settings.dt,
        save_every=settings.save_every,
        initial_scale=settings.initial_scale,
    )
    A, B = project_critical_amplitudes(result, normal.critical)
    tail_start = int(0.8 * result.t.size)
    tail = result.u[tail_start:, 0]
    centered = tail - tail.mean(axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(centered, axis=1)) ** 2, axis=0)
    dominant_mode = int(np.argmax(power[1:]) + 1)
    late_spatial_std = float(np.std(centered))
    q0_growth = float(np.max(np.linalg.eigvals(linear_symbol(model, 0.0, p)).real))
    verified = bool(
        dominant_mode == normal.critical.mode
        and late_spatial_std > 2.0 * settings.initial_scale
        and q0_growth < 0.0
    )

    fig, axes = plt.subplots(
        1, 3, figsize=subplot_figsize(3, row_height=4.2), constrained_layout=True
    )
    extent = [result.x[0], result.x[-1], result.t[0], result.t[-1]]
    activity_limit = _safe_limit([result.u[:, 0]], percentile=99.5)
    axes[0].imshow(
        result.u[:, 0],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-activity_limit,
        vmax=activity_limit,
    )
    axes[0].set(
        xlabel=f"space ({model.space_unit})",
        ylabel=f"time ({model.time_unit})",
    )
    _panel_text(
        axes[0],
        rf"stimulus peak=0; $\delta p=p-p_0={p - normal.critical.p0:+.4g}$",
    )
    shown_modes = np.arange(min(power.size, settings.max_mode + 4))
    colors = ["C3" if mode == normal.critical.mode else "C0" for mode in shown_modes]
    axes[1].bar(shown_modes, power[shown_modes], color=colors)
    axes[1].set(
        xlabel="spatial Fourier mode",
        ylabel="late E-mode power",
    )
    axes[2].plot(np.abs(A), result.t, color="C0", label="|A|")
    axes[2].plot(np.abs(B), result.t, color="C1", label="|B|")
    axes[2].set(
        xlabel="projected critical amplitude",
        ylabel=f"time ({model.time_unit})",
        ylim=(result.t[0], result.t[-1]),
    )
    axes[2].legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    summary = {
        "p": p,
        "p0": normal.critical.p0,
        "target_critical_growth": target_growth,
        "homogeneous_growth": q0_growth,
        "late_dominant_spatial_mode": dominant_mode,
        "expected_critical_mode": normal.critical.mode,
        "late_spatial_std": late_spatial_std,
        "late_mean_abs_A": float(np.mean(np.abs(A[tail_start:]))),
        "late_mean_abs_B": float(np.mean(np.abs(B[tail_start:]))),
        "classification": classify_wave(
            A,
            B,
            min_amplitude=settings.pattern_amplitude_threshold,
        ),
        "verified_sustained_pattern": verified,
        "simulation_time": total_time,
    }
    return output, summary


def plot_normal_form_branches(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    output: Path,
) -> Path:
    lower = max(settings.p_bounds[0], normal.critical.p0 - 2.2 * settings.p_margin)
    upper = min(settings.p_bounds[1], normal.critical.p0 + 3.5 * settings.p_margin)
    p_values = np.linspace(lower, upper, 300)
    branches = unforced_branch_amplitudes(normal, p_values)
    preferred = normal.preferred_pattern
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    before = p_values <= normal.critical.p0
    after = ~before
    ax.plot(
        p_values[before],
        branches["homogeneous"][before],
        color="black",
        lw=2.2,
        label="homogeneous (stable)",
    )
    ax.plot(
        p_values[after],
        branches["homogeneous"][after],
        color="black",
        lw=1.4,
        ls="--",
        label="homogeneous (unstable)",
    )
    for name, color in (("traveling", "C0"), ("standing", "C1")):
        stable = preferred.startswith(name)
        ax.plot(
            p_values,
            branches[name],
            color=color,
            lw=2.2 if stable else 1.5,
            ls="-" if stable else "--",
            label=f"{name} branch ({'stable' if stable else 'unstable'} at cubic order)",
        )
    ax.axvline(normal.critical.p0, color="C3", ls=":", lw=1.4)
    ax.set(
        xlabel="absolute connectivity parameter p",
        ylabel="predicted |A| (and |B| for standing)",
        title="Unforced cubic branches; report coordinate mu=p-p0",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_ideal_wave_patterns(
    model: ModelParameters,
    critical: BifurcationPoint,
    output: Path,
) -> Path:
    """Plot normalized leading-order E patterns for the four ideal A/B states."""
    x = np.linspace(0.0, model.length, 512, endpoint=False)
    period = 2.0 * np.pi / critical.omega0
    t = np.linspace(0.0, 6.0 * period, 721)
    phase_x = np.exp(1j * critical.q * x)[None, :]
    phase_t = np.exp(1j * critical.omega0 * t)[:, None]
    phi_e = complex(critical.phi[0])
    states = (
        ("homogeneous", 0.0j, 0.0j),
        ("standing wave", 1.0 + 0.0j, 1.0 + 0.0j),
        ("traveling wave", 1.0 + 0.0j, 0.0j),
        ("traveling wave", 0.0j, 1.0 + 0.0j),
    )
    fields = []
    for _, A, B in states:
        positive_mode = phi_e * A * phase_t + np.conj(phi_e) * np.conj(B) / phase_t
        fields.append(2.0 * np.real(positive_mode * phase_x))
    limit = _safe_limit(fields)

    fig, axes = plt.subplots(
        1,
        len(states),
        figsize=subplot_figsize(len(states), row_height=4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    extent = [x[0], x[-1], t[0], t[-1]]
    image = None
    for ax, (label, A, B), field in zip(axes, states, fields):
        image = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set(
            xlabel=f"space ({model.space_unit})",
            ylabel=f"time ({model.time_unit})",
        )
        _panel_text(
            ax,
            f"{label}; A={A.real:.0f}, B={B.real:.0f}\n"
            r"stimulus peak=0; unit ideal amplitudes",
        )
    assert image is not None
    fig.colorbar(
        image, ax=axes, shrink=0.82, label="ideal E activation (arbitrary units)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_unforced_sweep(
    model: ModelParameters,
    normal: NormalFormCoefficients,
    settings: RunSettings,
    output: Path,
    *,
    quick: bool,
) -> tuple[Path, list[dict[str, object]]]:
    offsets = (-settings.p_margin, 0.75 * settings.p_margin, 2.0 * settings.p_margin)
    symmetries = ("none", "reflection", "none")
    labels = (
        "below onset: homogeneous",
        "standing branch: reflection-symmetric random seed",
        "generic random seed above onset",
    )
    rows: list[tuple[str, float, str, SimulationResult, Array, Array]] = []
    summaries: list[dict[str, object]] = []
    for index, (offset, symmetry, label) in enumerate(zip(offsets, symmetries, labels)):
        p = normal.critical.p0 + offset
        result = simulate(
            model,
            p=p,
            seed=41 + index,
            grid_points=settings.grid_points,
            total_time=settings.total_time * (0.55 if quick else 1.0),
            dt=settings.dt,
            save_every=settings.save_every,
            initial_scale=settings.initial_scale,
            initial_symmetry=symmetry,
        )
        A, B = project_critical_amplitudes(result, normal.critical)
        classification = classify_wave(
            A,
            B,
            min_amplitude=settings.pattern_amplitude_threshold,
        )
        rows.append((label, p, classification, result, A, B))
        summaries.append(
            {
                "label": label,
                "p": p,
                "initial_symmetry": symmetry,
                "classification": classification,
                "late_mean_abs_A": float(np.mean(np.abs(A[int(0.8 * A.size) :]))),
                "late_mean_abs_B": float(np.mean(np.abs(B[int(0.8 * B.size) :]))),
            }
        )
    def render(selected_output: Path, *, late_time_window: float | None) -> None:
        fig, axes = plt.subplots(
            3,
            2,
            figsize=subplot_figsize(2, 3, row_height=3.5),
            constrained_layout=True,
        )
        for row, (label, p, classification, result, A, B) in enumerate(rows):
            if late_time_window is None:
                visible = np.ones(result.t.size, dtype=bool)
            else:
                visible = result.t >= result.t[-1] - late_time_window
            visible_time = result.t[visible]
            extent = [
                result.x[0],
                result.x[-1],
                visible_time[0],
                visible_time[-1],
            ]
            # Each parameter regime and time window gets its own symmetric
            # scale so the near-onset patterns remain visible.
            activity_limit = _safe_limit(
                [result.u[visible, 0]], percentile=99.5
            )
            axes[row, 0].imshow(
                result.u[visible, 0],
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="RdBu_r",
                vmin=-activity_limit,
                vmax=activity_limit,
            )
            axes[row, 0].set(
                xlabel=f"space ({model.space_unit})",
                ylabel=f"time ({model.time_unit})",
            )
            _panel_text(
                axes[row, 0],
                rf"stimulus peak=0; $\delta p=p-p_0={p - normal.critical.p0:+.4g}$",
            )
            axes[row, 1].plot(
                np.abs(A[visible]), visible_time, label="|A|", color="C0"
            )
            axes[row, 1].plot(
                np.abs(B[visible]), visible_time, label="|B|", color="C1"
            )
            axes[row, 1].set(
                xlabel="projected amplitude",
                ylabel=f"time ({model.time_unit})",
                ylim=(visible_time[0], visible_time[-1]),
            )
        axes[0, 1].legend(frameon=False)
        fig.savefig(selected_output, dpi=180)
        plt.close(fig)

    render(output, late_time_window=None)
    response_period = 2.0 * np.pi / normal.critical.omega0
    duration = min(result.t[-1] - result.t[0] for _, _, _, result, _, _ in rows)
    late_time_window = min(20.0 * response_period, 0.25 * duration)
    late_output = output.with_name(f"{output.stem}_late_zoom{output.suffix}")
    render(late_output, late_time_window=late_time_window)
    return output, summaries


def run_case(
    name: str,
    model: ModelParameters,
    settings: RunSettings,
    *,
    output_root: Path,
    quick: bool = False,
) -> dict[str, object]:
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    critical = compute_bifurcation(
        model, p_bounds=settings.p_bounds, max_mode=settings.max_mode
    )
    normal = compute_normal_form(model, critical)
    plot_bifurcation(model, settings, critical, output / "bifurcation.png")
    plot_normal_form_branches(
        model, normal, settings, output / "normal_form_branches.png"
    )
    plot_ideal_wave_patterns(model, critical, output / "ideal_wave_patterns.png")

    below_p = critical.p0 - settings.p_margin
    forcing_results: dict[str, tuple[SimulationResult, ForcingCoefficients]] = {}
    for index, (label, (forcing, temporal_harmonic, spatial_harmonic)) in enumerate(
        _forcing_suite(model, normal, settings, below_p).items()
    ):
        result = simulate(
            model,
            p=below_p,
            forcing=forcing,
            seed=7,
            grid_points=settings.grid_points,
            total_time=settings.total_time * (0.55 if quick else 1.0),
            dt=settings.dt,
            save_every=settings.save_every,
            initial_scale=settings.initial_scale,
        )
        coefficients = compute_forcing_coefficients(
            model,
            normal,
            temporal_harmonic=temporal_harmonic,
            spatial_harmonic=spatial_harmonic,
        )
        forcing_results[label] = (result, coefficients)
        np.savez_compressed(
            output
            / f"forcing_{index}_{label.replace(' ', '_').replace('+', 'and')}.npz",
            t=result.t,
            x=result.x,
            u=result.u,
            forcing=result.forcing,
            p=result.p,
        )
    plot_forcing_comparison(
        model,
        normal,
        below_p,
        forcing_results,
        output / "forcing_comparison.png",
        min_amplitude=settings.pattern_amplitude_threshold,
    )
    late_time_window = 20.0 * 2.0 * np.pi / critical.omega0
    plot_forcing_comparison(
        model,
        normal,
        below_p,
        forcing_results,
        output / "forcing_comparison_late_zoom.png",
        min_amplitude=settings.pattern_amplitude_threshold,
        late_time_window=late_time_window,
    )
    _, frequency_doubling = plot_frequency_doubling_verification(
        model,
        critical,
        forcing_results,
        output / "frequency_doubling_verification.png",
    )
    _, off_resonance_summary = plot_off_resonance_and_subthreshold(
        model,
        normal,
        settings,
        below_p,
        output / "off_resonance_and_subthreshold.png",
        quick=quick,
    )
    pattern_verification: dict[str, object] | None = None
    if name == "v1_epvsst_vip":
        _, pattern_verification = plot_pattern_verification(
            model,
            normal,
            settings,
            output / "pattern_forming_ap_verification.png",
            quick=quick,
        )
    _, sweep_summary = plot_unforced_sweep(
        model, normal, settings, output / "unforced_parameter_sweep.png", quick=quick
    )

    summary: dict[str, object] = {
        "name": name,
        "model": {
            "K": model.K,
            "names": model.names,
            "A": model.A,
            "A_p": model.A_p,
            "Sigma_variance": model.Sigma,
            "tau": model.tau,
            "theta": model.theta,
            "gain": model.gain,
            "length": model.length,
            "time_unit": model.time_unit,
            "space_unit": model.space_unit,
            "metadata": model.metadata,
        },
        "run_settings": {
            "p_bounds": settings.p_bounds,
            "p_margin": settings.p_margin,
            "epsilon": float(np.sqrt(settings.p_margin)),
            "epsilon_squared": settings.p_margin,
            "forcing_scaling": "physical forcing = epsilon^2 * order-one coefficient",
            "drive_amplitude_fallback": settings.drive_amplitude,
            "drive_weights": settings.drive_weights,
            "forcing_target_growth": settings.forcing_target_growth,
            "grid_points": settings.grid_points,
            "total_time": settings.total_time,
            "dt": settings.dt,
            "initial_scale": settings.initial_scale,
            "pattern_amplitude_threshold": settings.pattern_amplitude_threshold,
            "temporal_frequency_multiplier": settings.temporal_frequency_multiplier,
            "spatial_forcing_epsilon_squared_coefficient": (
                settings.spatial_forcing_epsilon_squared_coefficient
            ),
        },
        "bifurcation": {
            "p0": critical.p0,
            "mode": critical.mode,
            "q": critical.q,
            "eigenvalue": critical.eigenvalue,
            "omega0": critical.omega0,
            "frequency_cycles_per_time": critical.frequency_cycles_per_time,
            "frequency_hz_if_ms": critical.frequency_hz
            if model.time_unit == "ms"
            else None,
            "phi": critical.phi,
            "psi": critical.psi,
        },
        "normal_form": {
            "report_parameter": "mu = p - p0",
            "alpha": normal.alpha,
            "b": normal.b,
            "c": normal.c,
            "preferred_pattern": normal.preferred_pattern,
            "f1": normal.f1,
            "f2": normal.f2,
            "f3": normal.f3,
            "M0": normal.M0,
            "M1": normal.M1,
            "q20": normal.q20,
            "q11": normal.q11,
            "h20_plus": normal.h20_plus,
            "h20_zero": normal.h20_zero,
            "h11_zero": normal.h11_zero,
            "h11_plus": normal.h11_plus,
            "Gs": normal.Gs,
            "Gc": normal.Gc,
        },
        "forcing": {
            label: {
                "eta_t": coefficients.eta_t,
                "eta_x": coefficients.eta_x,
                "p2omega": coefficients.p2omega,
                "s2k": coefficients.s2k,
                "critical_zero_growth_rate": zero_state_growth_rate(
                    normal, p=below_p, forcing=coefficients
                ),
                "forcing_peak": float(np.max(np.abs(result.forcing))),
                "forcing_peak_over_epsilon_squared": float(
                    np.max(np.abs(result.forcing)) / settings.p_margin
                ),
                "temporal_forcing_peak": float(
                    np.max(
                        np.abs(
                            result.forcing_spec.vector(
                                result.forcing_spec.temporal_amplitude, model.K
                            )
                        )
                    )
                ),
                "spatial_forcing_peak": float(
                    np.max(
                        np.abs(
                            result.forcing_spec.vector(
                                result.forcing_spec.spatial_amplitude, model.K
                            )
                        )
                    )
                ),
                "temporal_amplitude": result.forcing_spec.vector(
                    result.forcing_spec.temporal_amplitude, model.K
                ),
                "spatial_amplitude": result.forcing_spec.vector(
                    result.forcing_spec.spatial_amplitude, model.K
                ),
                "late_classification": classify_wave(
                    *project_critical_amplitudes(result, critical),
                    min_amplitude=settings.pattern_amplitude_threshold,
                ),
            }
            for label, (result, coefficients) in forcing_results.items()
        },
        "frequency_doubling": frequency_doubling,
        "off_resonance_and_subthreshold": off_resonance_summary,
        "pattern_forming_ap_verification": pattern_verification,
        "unforced_sweep": sweep_summary,
        "figures": {
            "bifurcation": str(output / "bifurcation.png"),
            "normal_form_branches": str(output / "normal_form_branches.png"),
            "ideal_wave_patterns": str(output / "ideal_wave_patterns.png"),
            "forcing_comparison": str(output / "forcing_comparison.png"),
            "forcing_comparison_late_zoom": str(
                output / "forcing_comparison_late_zoom.png"
            ),
            "frequency_doubling": str(
                output / "frequency_doubling_verification.png"
            ),
            "off_resonance_and_subthreshold": str(
                output / "off_resonance_and_subthreshold.png"
            ),
            "pattern_forming_ap_verification": (
                str(output / "pattern_forming_ap_verification.png")
                if pattern_verification is not None
                else None
            ),
            "unforced_parameter_sweep": str(output / "unforced_parameter_sweep.png"),
            "unforced_parameter_sweep_late_zoom": str(
                output / "unforced_parameter_sweep_late_zoom.png"
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n"
    )
    return summary


def run_all(
    *,
    selected: tuple[str, ...] = ("pi_ei", "v1_epvsst_vip"),
    output_root: str | Path = "simulations/output",
    quick: bool = False,
) -> dict[str, dict[str, object]]:
    available = presets()
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown presets: {sorted(unknown)}")
    root = Path(output_root)
    return {
        name: run_case(name, *available[name], output_root=root, quick=quick)
        for name in selected
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=("all", "pi_ei", "v1_epvsst_vip"), default="all"
    )
    parser.add_argument("--output", type=Path, default=Path("simulations/output"))
    parser.add_argument(
        "--quick", action="store_true", help="shorter simulations for a smoke run"
    )
    args = parser.parse_args()
    selected = ("pi_ei", "v1_epvsst_vip") if args.preset == "all" else (args.preset,)
    summaries = run_all(selected=selected, output_root=args.output, quick=args.quick)
    for name, summary in summaries.items():
        point = summary["bifurcation"]
        normal = summary["normal_form"]
        print(
            f"{name}: p0={point['p0']:.8f}, mode={point['mode']}, "
            f"omega0={point['omega0']:.6f}, pattern={normal['preferred_pattern']}"
        )


if __name__ == "__main__":
    main()
