"""V1-only 2D periodic and bounded-domain comparison pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from .fft import BifurcationPoint, compute_normal_form, linear_symbol
from .pipeline import _forcing_case, _panel_text, subplot_figsize, v1_epvsst_vip_preset
from .simulation_2d import (
    Forcing2D,
    SimulationResult2D,
    dominant_spatial_mode,
    dominant_temporal_frequency_hz,
    simulate_2d,
)

Array = np.ndarray


def _stimulus_annotation(result: SimulationResult2D) -> str:
    """Report component strengths and the below-onset parameter offset."""
    spec = result.forcing_spec
    components = []
    for symbol, value in (
        ("S", spec.spatial_amplitude),
        ("P", spec.temporal_amplitude),
        ("T", spec.traveling_amplitude),
    ):
        strength = float(np.max(np.abs(spec.vector(value, result.model.K))))
        if strength > 0.0:
            components.append(f"{symbol}={strength:.4g}")
    stimulus = ", ".join(components) if components else "peak=0"
    _, settings = v1_epvsst_vip_preset()
    return f"stimulus {stimulus}; $\\delta p=p-p_0={-settings.p_margin:+.4g}$"


@dataclass(frozen=True)
class BifurcationGeometry2D:
    """First 2D Hopf crossing and its representative wavevector."""

    boundary: str
    point: BifurcationPoint
    wavevector: tuple[int, int]
    radial_mode: float

    @property
    def periodic_mode_vector(self) -> tuple[float, float]:
        """Physical wavevector in cycles-across-domain coordinates."""
        factor = 1.0 if self.boundary == "periodic" else 0.5
        return factor * self.wavevector[0], factor * self.wavevector[1]


def compute_bifurcation_2d(
    model,
    *,
    boundary: str,
    p_bounds: tuple[float, float],
    max_index: int = 12,
    scan_points: int = 401,
    tolerance: float = 1e-11,
) -> BifurcationGeometry2D:
    """Find the first Hopf crossing over 2D Fourier or cosine modes."""
    if boundary not in ("periodic", "reflect"):
        raise ValueError("2D spectral bifurcation supports periodic or reflect")
    if max_index < 1 or scan_points < 3:
        raise ValueError("max_index and scan_points are too small")
    p_min, p_max = map(float, p_bounds)
    factor = (2.0 * np.pi if boundary == "periodic" else np.pi) / model.length
    # nx >= ny selects one representative from each square-domain symmetry
    # orbit while retaining every distinct radial wavenumber.
    candidates = [
        (nx, ny, factor * float(np.hypot(nx, ny)))
        for nx in range(max_index + 1)
        for ny in range(nx + 1)
    ]

    def leading(q: float, p: float) -> complex:
        values = np.linalg.eigvals(linear_symbol(model, q, p))
        return complex(values[np.argmax(values.real)])

    if max(leading(q, p_min).real for _, _, q in candidates) >= 0.0:
        raise ValueError("the lower p bound is not stable on the 2D mode grid")
    grid = np.linspace(p_min, p_max, scan_points)
    crossings: list[tuple[float, int, int, float]] = []
    for nx, ny, q in candidates:
        growth = np.asarray([leading(q, p).real for p in grid])
        indices = np.flatnonzero((growth[:-1] < 0.0) & (growth[1:] >= 0.0))
        if indices.size:
            index = int(indices[0])
            root = brentq(
                lambda value, selected_q=q: leading(selected_q, value).real,
                grid[index],
                grid[index + 1],
                xtol=tolerance,
            )
            crossings.append((float(root), nx, ny, q))
    if not crossings:
        raise ValueError("no 2D modal instability was found in p_bounds")

    p0, nx, ny, q = min(crossings)
    matrix = linear_symbol(model, q, p0)
    values, vectors = np.linalg.eig(matrix)
    max_real = float(np.max(values.real))
    indices = np.flatnonzero(np.isclose(values.real, max_real, atol=1e-8, rtol=0.0))
    index = int(indices[np.argmax(values[indices].imag)])
    eigenvalue = complex(values[index])
    phi = vectors[:, index]
    left_values, left_vectors = np.linalg.eig(matrix.conj().T)
    left_index = int(np.argmin(np.abs(left_values - np.conj(eigenvalue))))
    psi = left_vectors[:, left_index]
    pairing = np.vdot(psi, phi)
    if abs(pairing) < 1e-12:
        raise np.linalg.LinAlgError("critical 2D left/right eigenvectors are orthogonal")
    psi = psi / np.conj(pairing)
    radial_mode = float(q * model.length / (2.0 * np.pi))
    point = BifurcationPoint(
        p0=float(p0),
        mode=int(round(radial_mode)),
        q=float(q),
        eigenvalue=eigenvalue,
        phi=phi,
        psi=psi,
    )
    return BifurcationGeometry2D(boundary, point, (nx, ny), radial_mode)


def v1_unit_summary(
    model,
    critical: BifurcationPoint,
    *,
    stimulus_radial_mode: float | None = None,
) -> dict[str, float | str]:
    """Human-readable physical-unit conversions for the V1 preset."""
    length_mm = 0.1 * model.length
    radial_mode = critical.q * model.length / (2.0 * np.pi)
    wavelength_mm = 0.1 * 2.0 * np.pi / critical.q
    stimulus_radial_mode = (
        2.0 * radial_mode
        if stimulus_radial_mode is None
        else float(stimulus_radial_mode)
    )
    return {
        "time_unit": model.time_unit,
        "space_unit": model.space_unit,
        "one_space_unit_um": 100.0,
        "domain_side_model_units": float(model.length),
        "domain_side_mm": float(length_mm),
        "critical_radial_mode": float(radial_mode),
        "critical_wavelength_mm": float(wavelength_mm),
        "critical_angular_frequency_rad_per_ms": float(critical.omega0),
        "critical_frequency_hz": float(critical.frequency_hz),
        "stimulus_radial_mode": float(stimulus_radial_mode),
        "stimulus_wavelength_mm": float(length_mm / stimulus_radial_mode),
        "stimulus_frequency_hz": float(2.0 * critical.frequency_hz),
        "activity_unit": "dimensionless shifted rate/activity",
        "A_and_Ap_unit": "dimensionless signed integrated coupling gain",
        "Sigma_unit": "(0.1 mm)^2",
        "forcing_unit": "same dimensionless recurrent-input unit as u",
        "p_unit": "dimensionless",
    }


def _centered_e(result: SimulationResult2D, index: int = -1) -> Array:
    value = result.u[index, 0].astype(float)
    return value - value.mean()


def _selected_mode_physical_amplitude(result: SimulationResult2D) -> Array:
    """Convert the stored transform coefficient to a real-wave amplitude."""
    coefficient = np.abs(result.mode_signal[:, 0])
    if result.boundary == "periodic":
        return 2.0 * coefficient
    n = result.u.shape[-1]
    mx, my = result.diagnostic_wavevector
    ix, iy = int(round(2.0 * abs(mx))), int(round(2.0 * abs(my)))
    scale_x = np.sqrt((2.0 if ix > 0 else 1.0) / n)
    scale_y = np.sqrt((2.0 if iy > 0 else 1.0) / n)
    return coefficient * scale_x * scale_y


def plot_final_state_comparison(
    results: dict[str, dict[str, SimulationResult2D]], output: Path
) -> Path:
    boundaries = tuple(results)
    cases = tuple(next(iter(results.values())))
    values = [
        _centered_e(results[boundary][case])
        for boundary in boundaries
        for case in cases
    ]
    limit = max(float(np.percentile(np.abs(value), 99.5)) for value in values)
    limit = max(limit, 1e-8)
    fig, axes = plt.subplots(
        len(boundaries),
        len(cases),
        figsize=subplot_figsize(len(cases), len(boundaries), row_height=3.8),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row, boundary in enumerate(boundaries):
        for column, case in enumerate(cases):
            result = results[boundary][case]
            extent = [
                0.0,
                0.1 * result.model.length,
                0.0,
                0.1 * result.model.length,
            ]
            image = axes[row, column].imshow(
                _centered_e(result),
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            _panel_text(axes[row, column], _stimulus_annotation(result))
            axes[row, column].set_xlabel("x (mm)")
            if column == 0:
                boundary_name = (
                    "periodic" if boundary == "periodic" else "bounded, reflecting"
                )
                axes[row, column].set_ylabel(
                    f"{boundary_name}\ny (mm)"
                )
    assert image is not None
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        label="E activity minus its spatial mean",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_pattern_contrast(
    results: dict[str, dict[str, SimulationResult2D]], output: Path
) -> Path:
    boundaries = tuple(results)
    fig, axes = plt.subplots(
        1, len(boundaries), figsize=(6.2 * len(boundaries), 4.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, boundary in zip(axes, boundaries):
        for case, result in results[boundary].items():
            ax.plot(
                result.spatial_rms[:, 0],
                result.diagnostic_t / 1000.0,
                label=case,
                lw=1.8,
            )
        ax.set(
            xlabel="E spatial RMS (dimensionless activity)",
            ylabel="time (s)",
            title=(
                "periodic boundary"
                if boundary == "periodic"
                else "bounded reflecting boundary"
            ),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Growth or decay of two-dimensional spatial contrast")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_selected_mode_amplitude(
    results: dict[str, dict[str, SimulationResult2D]], output: Path
) -> Path:
    boundaries = tuple(results)
    fig, axes = plt.subplots(
        1, len(boundaries), figsize=(6.2 * len(boundaries), 4.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, boundary in zip(axes, boundaries):
        for case, result in results[boundary].items():
            ax.plot(
                _selected_mode_physical_amplitude(result),
                result.diagnostic_t / 1000.0,
                label=case,
                lw=1.7,
            )
        ax.set(
            xlabel="selected critical-wave amplitude",
            ylabel="time (s)",
            title=(
                "periodic boundary"
                if boundary == "periodic"
                else "bounded reflecting boundary"
            ),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Half-frequency response at the selected 2D critical wavevector")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_combined_evolution(
    results: dict[str, dict[str, SimulationResult2D]], output: Path
) -> Path:
    boundaries = tuple(results)
    fractions = (0.0, 0.5, 0.8, 1.0)
    selected = {
        boundary: results[boundary]["spatial + temporal"]
        for boundary in boundaries
    }
    values = []
    for result in selected.values():
        for fraction in fractions:
            index = min(round(fraction * (result.t.size - 1)), result.t.size - 1)
            values.append(_centered_e(result, index))
    limit = max(float(np.percentile(np.abs(value), 99.5)) for value in values)
    limit = max(limit, 1e-8)
    fig, axes = plt.subplots(
        len(boundaries),
        len(fractions),
        figsize=subplot_figsize(len(fractions), len(boundaries), row_height=3.8),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row, boundary in enumerate(boundaries):
        result = selected[boundary]
        extent = [0.0, 0.1 * result.model.length, 0.0, 0.1 * result.model.length]
        for column, fraction in enumerate(fractions):
            index = min(round(fraction * (result.t.size - 1)), result.t.size - 1)
            image = axes[row, column].imshow(
                _centered_e(result, index),
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            _panel_text(
                axes[row, column],
                _stimulus_annotation(result),
            )
            axes[row, column].set_xlabel("x (mm)")
            if column == 0:
                label = "periodic" if boundary == "periodic" else "bounded, reflecting"
                axes[row, column].set_ylabel(f"{label}\ny (mm)")
    assert image is not None
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        label="E activity minus its spatial mean",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def summarize_results(
    results: dict[str, dict[str, SimulationResult2D]],
    geometries: dict[str, BifurcationGeometry2D],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for boundary, cases in results.items():
        geometry = geometries[boundary]
        critical = geometry.point
        summary[boundary] = {}
        for case, result in cases.items():
            spatial = dominant_spatial_mode(result)
            selected_amplitude = _selected_mode_physical_amplitude(result)
            summary[boundary][case] = {
                **spatial,
                "late_E_spatial_rms": float(
                    np.mean(result.spatial_rms[int(0.8 * result.spatial_rms.shape[0]) :, 0])
                ),
                "late_selected_critical_wave_amplitude": float(
                    np.mean(selected_amplitude[int(0.8 * selected_amplitude.size) :])
                ),
                "probe_frequency_hz": dominant_temporal_frequency_hz(result),
                "selected_mode_frequency_hz": dominant_temporal_frequency_hz(
                    result, use_mode_signal=True
                ),
                "expected_pattern_frequency_hz": float(critical.frequency_hz),
                "expected_pattern_radial_mode": float(geometry.radial_mode),
                "critical_p0": float(critical.p0),
                "run_p": float(result.p),
            }
    return summary


def run_v1_2d_suite(
    *,
    output_root: str | Path = "simulations/output/v1_epvsst_vip_2d",
    boundaries: tuple[str, ...] = ("periodic", "reflect"),
    quick: bool = False,
    grid_points: int | None = None,
    total_time: float | None = None,
) -> tuple[
    dict[str, dict[str, SimulationResult2D]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, Path],
]:
    """Run the four forcing cases for periodic and bounded V1 squares."""
    model, settings = v1_epvsst_vip_preset()
    n = grid_points if grid_points is not None else (32 if quick else 64)
    duration = total_time if total_time is not None else (2000.0 if quick else 12000.0)
    save_every = 40 if quick else 100
    results: dict[str, dict[str, SimulationResult2D]] = {}
    geometries: dict[str, BifurcationGeometry2D] = {}
    for boundary in boundaries:
        if boundary not in ("periodic", "reflect", "zero"):
            raise ValueError("boundaries must contain periodic, reflect, or zero")
        geometry = compute_bifurcation_2d(
            model,
            boundary=boundary,
            p_bounds=settings.p_bounds,
            max_index=settings.max_mode,
        )
        geometries[boundary] = geometry
        print(
            f"{boundary}: p0={geometry.point.p0:.8f}, "
            f"wavevector={geometry.periodic_mode_vector}, "
            f"radial={geometry.radial_mode:.4f}, "
            f"f0={geometry.point.frequency_hz:.3f} Hz"
        )
        critical = geometry.point
        normal = compute_normal_form(model, critical)
        p = critical.p0 - settings.p_margin
        forcing_cases_1d = {
            "no forcing": None,
            "spatial only": _forcing_case(model, normal, settings, p, "spatial")[0],
            "temporal only": _forcing_case(model, normal, settings, p, "temporal")[0],
            "spatial + temporal": _forcing_case(model, normal, settings, p, "both")[0],
        }
        results[boundary] = {}
        for case, forcing_1d in forcing_cases_1d.items():
            print(f"  simulating {case} ({duration / 1000.0:.1f} s, {n}x{n})")
            forcing_2d = Forcing2D() if forcing_1d is None else Forcing2D.from_1d(forcing_1d)
            physical_mode = geometry.periodic_mode_vector
            if boundary == "periodic":
                forcing_2d.spatial_mode = (
                    2 * physical_mode[0],
                    2 * physical_mode[1],
                )
                forcing_2d.spatial_pattern = "plane"
            else:
                # A reflecting eigenfunction is cos(nx*pi*x/L)cos(ny*pi*y/L).
                # Multiplication by cos(2*nx*pi*x/L) maps part of it back to
                # the same (nx, ny) mode, exactly as the 1D 2k resonance does.
                # Doubling both cosine indices instead gives a much weaker
                # fourth-order product projection and is not the direct 1D
                # analogue.
                forcing_2d.spatial_mode = (
                    2 * geometry.wavevector[0],
                    0.0,
                )
                forcing_2d.spatial_pattern = "cosine_product"
            results[boundary][case] = simulate_2d(
                model,
                p=p,
                forcing=forcing_2d,
                seed=7,
                grid_points=n,
                total_time=duration,
                dt=0.5,
                save_every=save_every,
                diagnostic_every=1,
                initial_scale=settings.initial_scale,
                boundary=boundary,
                integrator="rk2",
                diagnostic_wavevector=physical_mode,
            )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures = {
        "final_states": plot_final_state_comparison(
            results, output_root / "final_state_comparison.png"
        ),
        "pattern_contrast": plot_pattern_contrast(
            results, output_root / "pattern_contrast.png"
        ),
        "critical_mode_amplitude": plot_selected_mode_amplitude(
            results, output_root / "critical_mode_amplitude.png"
        ),
        "combined_evolution": plot_combined_evolution(
            results, output_root / "combined_evolution.png"
        ),
    }
    summary = summarize_results(results, geometries)
    metadata = {
        "model": "v1_epvsst_vip",
        "forcing_population_order": list(model.names),
        "forcing_direction_weights": (
            list(settings.drive_weights)
            if settings.drive_weights is not None
            else [1.0] + [0.0] * (model.K - 1)
        ),
        "forcing_interpretation": model.metadata.get("forcing_direction"),
        "boundaries": list(boundaries),
        "bounded_interpretation": "reflecting/even extension at all four edges",
        "grid_points_per_axis": int(n),
        "total_time_ms": float(duration),
        "dt_ms": 0.5,
        "integrator": "second-order Runge--Kutta (Heun)",
        "reflecting_spatial_forcing": (
            "boundary-compatible cos(2*n_x*pi*x/L), which maps the critical "
            "cos(n_x*pi*x/L)cos(n_y*pi*y/L) eigenfunction back to itself"
        ),
        "boundary_bifurcations": {
            boundary: {
                "p0": float(geometry.point.p0),
                "run_p": float(geometry.point.p0 - settings.p_margin),
                "wavevector_indices": list(geometry.wavevector),
                "wavevector_in_cycles_across_domain": list(
                    geometry.periodic_mode_vector
                ),
                "spatial_stimulus_indices": (
                    [
                        2 * geometry.wavevector[0],
                        2 * geometry.wavevector[1],
                    ]
                    if boundary == "periodic"
                    else [2 * geometry.wavevector[0], 0]
                ),
                "radial_mode": float(geometry.radial_mode),
                "omega0_rad_per_ms": float(geometry.point.omega0),
                "frequency_hz": float(geometry.point.frequency_hz),
                "units": v1_unit_summary(
                    model,
                    geometry.point,
                    stimulus_radial_mode=(
                        2.0 * geometry.radial_mode
                        if boundary == "periodic"
                        else float(geometry.wavevector[0])
                    ),
                ),
            }
            for boundary, geometry in geometries.items()
        },
        "results": summary,
    }
    (output_root / "summary.json").write_text(json.dumps(metadata, indent=2))
    for boundary, cases in results.items():
        for case, result in cases.items():
            safe_case = case.replace(" ", "_").replace("+", "and")
            np.savez_compressed(
                output_root / f"{boundary}_{safe_case}_diagnostics.npz",
                saved_t_ms=result.t,
                final_u=result.u[-1],
                diagnostic_t_ms=result.diagnostic_t,
                probe=result.probe,
                spatial_rms=result.spatial_rms,
                mode_signal=result.mode_signal,
            )
    return results, summary, figures


if __name__ == "__main__":
    run_v1_2d_suite()
