"""Reproduce the numerical tables and figures used in the project report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import neural_field as nf

OUTPUT = Path("output/simulations")


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def spectrum_figure(model: nf.PopulationModel, critical: nf.CriticalMode) -> Path:
    scales = [0.99 * critical.scale, critical.scale, 1.01 * critical.scale]
    modes = np.arange(13)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
    for scale in scales:
        values = []
        frequencies = []
        for mode in modes:
            q = 2 * np.pi * mode / model.length
            eig = np.linalg.eigvals(nf.linear_symbol(model, q, scale))
            lead = eig[np.argmax(eig.real)]
            values.append(lead.real)
            frequencies.append(abs(lead.imag) * 1000 / (2 * np.pi))
        label = f"scale={scale:.3f}"
        axes[0].plot(modes, values, marker="o", ms=3, label=label)
        axes[1].plot(modes, frequencies, marker="o", ms=3, label=label)
    axes[0].axhline(0, color="0.25", lw=1)
    axes[0].axvline(critical.mode, color="C3", ls="--", lw=1)
    axes[1].axvline(critical.mode, color="C3", ls="--", lw=1)
    axes[0].set(xlabel="Fourier mode", ylabel="leading growth rate (ms$^{-1}$)")
    axes[1].set(xlabel="Fourier mode", ylabel="oscillation frequency (Hz)")
    axes[0].legend(frameon=False, fontsize=8)
    path = OUTPUT / "linear_spectrum.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def phase_diagram(normal: nf.NormalForm) -> Path:
    a, b, c = 0.01, normal.b.real, normal.c.real
    forcing = 0.08
    axis = np.linspace(0, 0.38, 26)
    aa, bb = np.meshgrid(axis, axis)
    panels = [
        (0.0, "unforced: traveling states attract"),
        (forcing, "symmetric 2:1 forcing: standing state attracts"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), constrained_layout=True)
    for ax, (q, title) in zip(axes, panels):
        da = aa * (a + b * aa**2 + c * bb**2) + q * bb
        db = bb * (a + b * bb**2 + c * aa**2) + q * aa
        speed = np.hypot(da, db)
        ax.streamplot(axis, axis, da, db, color=speed, density=1.0, cmap="viridis")
        starts = ([0.03, 0.04], [0.18, 0.05], [0.08, 0.22], [0.28, 0.25])
        for start in starts:
            _, history = nf.integrate_amplitudes(
                a=a,
                b=b,
                c=c,
                eta_t=q / 2,
                eta_x=q / 2,
                initial=np.array([start[0], 0.0, start[1], 0.0]),
                total=500,
                dt=0.05,
            )
            ax.plot(
                np.hypot(history[:, 0], history[:, 1]),
                np.hypot(history[:, 2], history[:, 3]),
                color="white",
                lw=1.2,
            )
        if q == 0.0:
            traveling = np.sqrt(-a / b)
            standing = np.sqrt(-a / (b + c))
            ax.scatter(
                [traveling, 0.0],
                [0.0, traveling],
                marker="o",
                s=48,
                color="white",
                edgecolor="black",
                zorder=5,
                label="stable traveling waves",
            )
            ax.scatter(
                [standing],
                [standing],
                marker="x",
                s=60,
                color="white",
                zorder=5,
                label="unstable standing wave",
            )
        else:
            standing = np.sqrt(-(a + q) / (b + c))
            ax.scatter(
                [standing],
                [standing],
                marker="o",
                s=50,
                color="white",
                edgecolor="black",
                zorder=5,
                label="stable pinned standing wave",
            )
        ax.set(
            xlim=(0, axis[-1]),
            ylim=(0, axis[-1]),
            xlabel=r"$|A|$",
            ylabel=r"$|B|$",
            title=title,
        )
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    path = OUTPUT / "amplitude_phase_diagrams.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def retinotopic_gallery() -> Path:
    return nf.save_retinotopic_gallery(OUTPUT / "retinotopic_hallucination_gallery.png")


def positive_intervals(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """Linearly interpolate the boundaries of contiguous y > 0 intervals."""
    intervals: list[list[float]] = []
    start: float | None = None
    for index, value in enumerate(y):
        if value > 0.0 and start is None:
            if index == 0:
                start = float(x[index])
            else:
                fraction = -y[index - 1] / (value - y[index - 1])
                start = float(x[index - 1] + fraction * (x[index] - x[index - 1]))
        if value <= 0.0 and start is not None:
            fraction = y[index - 1] / (y[index - 1] - value)
            stop = float(x[index - 1] + fraction * (x[index] - x[index - 1]))
            intervals.append([start, stop])
            start = None
    if start is not None:
        intervals.append([start, float(x[-1])])
    return intervals


def five_population_analysis() -> dict[str, object]:
    """Apply the normal form and forcing calculation after splitting SST."""
    model = nf.v1_functional_five_population()
    critical = nf.find_critical_mode(model, scale_bracket=(0.3, 2.0))
    normal = nf.normal_form_coefficients(model, critical)
    forcing = nf.forcing_suite(model, critical, amplitude=0.025)

    cases = [
        ("five_population_unforced", 1.012, forcing["unforced"]),
        ("five_population_combined", 0.990, forcing["combined_2to1"]),
    ]
    field_results: list[dict[str, float | str]] = []
    for index, (name, relative_scale, drive) in enumerate(cases):
        result = nf.simulate(
            model,
            dimension=1,
            grid_points=128,
            boundary="periodic",
            coupling_scale=critical.scale * relative_scale,
            forcing=drive,
            total_ms=1800.0,
            dt_ms=0.5,
            save_every=10,
            initial_noise=0.003,
            seed=31 + index,
        )
        nf.save_simulation_figure(result, OUTPUT / f"{name}.png")
        late = result.t > 0.8 * result.t[-1]
        amplitude_a, amplitude_b = nf.critical_amplitudes(result, critical)
        field_results.append(
            {
                "case": name,
                "late_E_standard_deviation": float(np.std(result.u[late, 0])),
                "late_mean_abs_A": float(np.mean(np.abs(amplitude_a[late]))),
                "late_mean_abs_B": float(np.mean(np.abs(amplitude_b[late]))),
            }
        )

    frequencies = np.arange(1.0, 36.01, 0.5)
    amplitudes = np.array([0.0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05])
    growth = np.empty((amplitudes.size, frequencies.size))
    selected_mode = np.empty_like(growth, dtype=int)
    scale = 0.99 * critical.scale
    rows: list[list[float | int]] = []
    for i, amplitude in enumerate(amplitudes):
        for j, frequency in enumerate(frequencies):
            spectrum = nf.floquet_spectrum(
                model,
                coupling_scale=scale,
                drive_frequency_hz=float(frequency),
                drive_amplitude=float(amplitude),
                max_mode=8,
                settle_cycles=16,
                samples_per_cycle=80,
            )
            growth[i, j] = spectrum.pattern_growth_per_ms
            selected_mode[i, j] = spectrum.pattern_mode
            rows.append(
                [
                    amplitude,
                    frequency,
                    spectrum.pattern_growth_per_ms,
                    spectrum.pattern_mode,
                ]
            )
    with (OUTPUT / "floquet_resonance_scan.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "drive_amplitude",
                "drive_frequency_hz",
                "max_finite_mode_growth_per_ms",
                "selected_mode",
            ]
        )
        writer.writerows(rows)

    amplitude_index = int(np.where(np.isclose(amplitudes, 0.025))[0][0])
    intervals = positive_intervals(frequencies, growth[amplitude_index])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    image = axes[0].pcolormesh(
        frequencies,
        amplitudes,
        1000.0 * growth,
        shading="auto",
        cmap="coolwarm",
    )
    axes[0].contour(
        frequencies,
        amplitudes,
        growth,
        levels=[0.0],
        colors="black",
        linewidths=1.0,
    )
    axes[0].axvline(critical.frequency_hz, color="white", ls="--", lw=1)
    axes[0].axvline(2.0 * critical.frequency_hz, color="white", ls=":", lw=1)
    axes[0].set(
        xlabel="uniform drive frequency (Hz)",
        ylabel="model input amplitude",
        title="finite-mode Floquet growth",
    )
    fig.colorbar(image, ax=axes[0], label="growth (s$^{-1}$)")
    axes[1].plot(
        frequencies,
        1000.0 * growth[amplitude_index],
        color="C0",
        lw=1.8,
    )
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].fill_between(
        frequencies,
        0.0,
        1000.0 * growth[amplitude_index],
        where=growth[amplitude_index] > 0.0,
        color="C3",
        alpha=0.25,
    )
    axes[1].set(
        xlabel="uniform drive frequency (Hz)",
        ylabel="max finite-mode growth (s$^{-1}$)",
        title="model pattern-forming bands at amplitude 0.025",
    )
    fig.savefig(OUTPUT / "floquet_resonance_map.png", dpi=190)
    plt.close(fig)

    nf.save_model_input_bundle(OUTPUT / "model_inputs", critical, amplitude=0.025)
    forced = nf.symmetric_forced_equilibria(
        a=0.01,
        b=normal.b.real,
        c=normal.c.real,
        eta_t=0.04,
        eta_x=0.04,
    )
    forced_all = nf.real_forced_equilibria(
        a=0.01,
        b=normal.b.real,
        c=normal.c.real,
        eta_t=0.04,
        eta_x=0.04,
    )
    return {
        "preset": model.provenance,
        "populations": list(model.names),
        "critical": {
            "coupling_scale": critical.scale,
            "mode": critical.mode,
            "wavenumber": critical.q,
            "wavelength_mm": 0.1 * model.length / critical.mode,
            "frequency_hz": critical.frequency_hz,
            "homogeneous_growth_rate": float(
                np.linalg.eigvals(
                    nf.linear_symbol(model, 0.0, critical.scale)
                ).real.max()
            ),
        },
        "normal_form": {
            "alpha": complex_pair(normal.alpha),
            "b": complex_pair(normal.b),
            "c": complex_pair(normal.c),
            "local_selection": normal.pattern(),
        },
        "forced_symmetric_states": [
            {
                "amplitude": abs(state["A"]),
                "eigenvalues": np.asarray(state["eigenvalues"]).tolist(),
                "stable": state["stable"],
            }
            for state in forced
        ],
        "forced_all_real_states": [
            {
                "A": state["A"],
                "B": state["B"],
                "branch": state["branch"],
                "eigenvalues": np.asarray(state["eigenvalues"]).tolist(),
                "stable": state["stable"],
            }
            for state in forced_all
        ],
        "floquet_scan": {
            "relative_coupling_scale": 0.99,
            "drive_amplitude": 0.025,
            "positive_growth_intervals_hz": intervals,
            "interpretation": (
                "finite-mode pattern growth in the model, not a human "
                "hallucination threshold or safety range"
            ),
        },
        "simulations": field_results,
    }


def run_fields(
    model: nf.PopulationModel, critical: nf.CriticalMode
) -> list[dict[str, object]]:
    suite = nf.forcing_suite(model, critical, amplitude=0.025)
    cases = [
        ("1d_periodic_unforced", 1, "periodic", 1.015, suite["unforced"], 1800.0),
        ("1d_periodic_temporal", 1, "periodic", 0.995, suite["temporal_2to1"], 1800.0),
        ("1d_periodic_spatial", 1, "periodic", 0.995, suite["spatial_2to1"], 1800.0),
        ("1d_periodic_combined", 1, "periodic", 0.995, suite["combined_2to1"], 1800.0),
        ("1d_reflect_combined", 1, "reflect", 0.995, suite["combined_2to1"], 1800.0),
        ("2d_periodic_unforced", 2, "periodic", 1.012, suite["unforced"], 1200.0),
        ("2d_reflect_unforced", 2, "reflect", 1.012, suite["unforced"], 1200.0),
    ]
    summaries: list[dict[str, object]] = []
    for index, (name, dimension, boundary, relative_scale, forcing, total) in enumerate(
        cases
    ):
        result = nf.simulate(
            model,
            dimension=dimension,
            grid_points=128 if dimension == 1 else 64,
            boundary=boundary,
            coupling_scale=critical.scale * relative_scale,
            forcing=forcing,
            total_ms=total,
            dt_ms=0.5,
            save_every=10 if dimension == 1 else 20,
            initial_noise=0.003,
            seed=2 + index,
        )
        nf.save_simulation_figure(result, OUTPUT / f"{name}.png")
        late = result.t > 0.8 * total
        item: dict[str, object] = {
            "case": name,
            "dimension": dimension,
            "boundary": boundary,
            "relative_scale": relative_scale,
            "late_E_standard_deviation": float(np.std(result.u[late, 0])),
            "final_E_min": float(result.u[-1, 0].min()),
            "final_E_max": float(result.u[-1, 0].max()),
        }
        if dimension == 1 and boundary == "periodic":
            amplitude_a, amplitude_b = nf.critical_amplitudes(result, critical)
            item["late_mean_abs_A"] = float(np.mean(np.abs(amplitude_a[late])))
            item["late_mean_abs_B"] = float(np.mean(np.abs(amplitude_b[late])))
        summaries.append(item)
    return summaries


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = nf.v1_three_population()
    critical = nf.find_critical_mode(model)
    normal = nf.normal_form_coefficients(model, critical)
    angles = np.arange(30, 181, 30)
    angular = [
        (int(angle), nf.angular_cross_coefficient(model, normal, float(angle)))
        for angle in angles
    ]

    with (OUTPUT / "angular_coefficients.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["angle_degrees", "cross_real", "cross_imag", "cross_minus_self_real"]
        )
        for angle, value in angular:
            writer.writerow([angle, value.real, value.imag, value.real - normal.b.real])

    spectrum_figure(model, critical)
    phase_diagram(normal)
    retinotopic_gallery()
    simulations = run_fields(model, critical)
    expanded_model = five_population_analysis()

    forced_states = nf.symmetric_forced_equilibria(
        a=0.01, b=normal.b.real, c=normal.c.real, eta_t=0.04, eta_x=0.04
    )
    forced_all_states = nf.real_forced_equilibria(
        a=0.01, b=normal.b.real, c=normal.c.real, eta_t=0.04, eta_x=0.04
    )
    summary = {
        "preset": model.provenance,
        "critical": {
            "coupling_scale": critical.scale,
            "mode": critical.mode,
            "wavenumber": critical.q,
            "eigenvalue": complex_pair(critical.eigenvalue),
            "frequency_hz": critical.frequency_hz,
            "homogeneous_growth_rate": float(
                np.linalg.eigvals(
                    nf.linear_symbol(model, 0.0, critical.scale)
                ).real.max()
            ),
        },
        "normal_form": {
            "alpha": complex_pair(normal.alpha),
            "b": complex_pair(normal.b),
            "c": complex_pair(normal.c),
            "local_selection": normal.pattern(),
        },
        "angular_cross_coefficients": {
            str(angle): complex_pair(value) for angle, value in angular
        },
        "forced_symmetric_states": [
            {
                "amplitude": abs(state["A"]),
                "eigenvalues": np.asarray(state["eigenvalues"]).tolist(),
                "stable": state["stable"],
            }
            for state in forced_states
        ],
        "forced_all_real_states": [
            {
                "A": state["A"],
                "B": state["B"],
                "branch": state["branch"],
                "eigenvalues": np.asarray(state["eigenvalues"]).tolist(),
                "stable": state["stable"],
            }
            for state in forced_all_states
        ],
        "simulations": simulations,
        "expanded_five_population_model": expanded_model,
        "data_resolution": {
            "transcriptomic_taxonomy_types": 133,
            "complete_layer_by_class_circuit_populations": 17,
            "applied_rate_model_populations": 5,
            "reason": (
                "fine-type physiological gains are incomplete; measured "
                "identities and connection probabilities are kept separate "
                "from calibrated Wilson-Cowan weights"
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["critical"], indent=2))
    print(json.dumps(summary["normal_form"], indent=2))


if __name__ == "__main__":
    main()
