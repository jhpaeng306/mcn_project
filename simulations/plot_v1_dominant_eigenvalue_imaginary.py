"""Plot Im(lambda_dom) versus wavenumber for the four-population V1 model.

The spectrum is evaluated at the finite-wavenumber Hopf onset.  At each
wavenumber, ``lambda_dom`` is the eigenvalue with the largest real part.  When
the dominant eigenvalues form a complex-conjugate pair, the member with the
positive imaginary part is plotted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fft import compute_bifurcation, linear_symbol
from .pipeline import v1_epvsst_vip_preset

MM_PER_MODEL_UNIT = 0.1


def dominant_eigenvalue(matrix: np.ndarray) -> complex:
    """Return the rightmost eigenvalue, choosing Im(lambda) >= 0 at a tie."""
    eigenvalues = np.linalg.eigvals(matrix)
    largest_real_part = float(eigenvalues.real.max())
    candidates = eigenvalues[
        np.isclose(eigenvalues.real, largest_real_part, atol=1e-10, rtol=0.0)
    ]
    return complex(candidates[np.argmax(candidates.imag)])


def plot(output_dir: Path, *, samples: int = 2001) -> tuple[Path, Path]:
    """Create PNG and PDF versions of the V1 dispersion-frequency plot."""
    if samples < 101:
        raise ValueError("samples must be at least 101")

    model, settings = v1_epvsst_vip_preset()
    critical = compute_bifurcation(
        model,
        p_bounds=settings.p_bounds,
        max_mode=settings.max_mode,
    )

    mode_max = settings.max_mode
    q_model = np.linspace(0.0, 2.0 * np.pi * mode_max / model.length, samples)
    dominant = np.asarray(
        [dominant_eigenvalue(linear_symbol(model, q, critical.p0)) for q in q_model]
    )
    q_per_mm = q_model / MM_PER_MODEL_UNIT
    omega = dominant.imag

    discrete_modes = np.arange(mode_max + 1)
    discrete_q_model = 2.0 * np.pi * discrete_modes / model.length
    discrete_dominant = np.asarray(
        [
            dominant_eigenvalue(linear_symbol(model, q, critical.p0))
            for q in discrete_q_model
        ]
    )
    discrete_q_per_mm = discrete_q_model / MM_PER_MODEL_UNIT

    critical_q_per_mm = critical.q / MM_PER_MODEL_UNIT
    critical_omega = critical.eigenvalue.imag
    critical_frequency_hz = 1000.0 * critical_omega / (2.0 * np.pi)
    critical_wavelength_mm = 2.0 * np.pi / critical_q_per_mm

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    line_color = "#0072B2"
    critical_color = "#D55E00"
    grid_color = "#B8B8B8"

    ax.plot(
        q_per_mm,
        omega,
        color=line_color,
        linewidth=2.2,
        label=r"$\operatorname{Im}\lambda_{\mathrm{dom}}(q)$",
    )
    ax.scatter(
        discrete_q_per_mm,
        discrete_dominant.imag,
        s=24,
        facecolor="white",
        edgecolor=line_color,
        linewidth=1.1,
        zorder=3,
        label="integer Fourier modes",
    )
    ax.axvline(critical_q_per_mm, color=critical_color, linewidth=1.2, linestyle="--")
    ax.scatter(
        [critical_q_per_mm],
        [critical_omega],
        marker="*",
        s=190,
        color=critical_color,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="critical mode",
    )
    ax.annotate(
        (
            rf"$k_0={critical.mode}$, $q_0={critical_q_per_mm:.3f}$ rad mm$^{{-1}}$"
            "\n"
            rf"$\operatorname{{Im}}\lambda_0={critical_omega:.5f}$ rad ms$^{{-1}}$"
            f"  ({critical_frequency_hz:.2f} Hz)"
            "\n"
            rf"$\lambda_{{\mathrm{{space}}}}={critical_wavelength_mm:.3f}$ mm"
        ),
        xy=(critical_q_per_mm, critical_omega),
        xytext=(12, -62),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": critical_color, "lw": 1.0},
    )

    ax.set(
        xlabel=r"spatial angular wavenumber $q$ (rad mm$^{-1}$)",
        ylabel=r"$\operatorname{Im}\lambda_{\mathrm{dom}}$ (rad ms$^{-1}$)",
        title=(
            "Four-population V1 model: dominant-eigenvalue frequency\n"
            rf"at Hopf onset $p=p_0={critical.p0:.6f}$"
        ),
        xlim=(q_per_mm[0], q_per_mm[-1]),
    )
    ax.set_ylim(bottom=min(-0.004, float(omega.min()) - 0.004))
    ax.grid(True, color=grid_color, linewidth=0.65, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    def q_to_mode(q_mm: np.ndarray | float) -> np.ndarray | float:
        return q_mm * (model.length * MM_PER_MODEL_UNIT) / (2.0 * np.pi)

    def mode_to_q(mode: np.ndarray | float) -> np.ndarray | float:
        return mode * 2.0 * np.pi / (model.length * MM_PER_MODEL_UNIT)

    top_axis = ax.secondary_xaxis("top", functions=(q_to_mode, mode_to_q))
    top_axis.set_xlabel(r"Fourier mode $k=qL/(2\pi)$")
    top_axis.set_xticks(np.arange(0, mode_max + 1, 2))

    def omega_to_hz(value: np.ndarray | float) -> np.ndarray | float:
        return 1000.0 * value / (2.0 * np.pi)

    def hz_to_omega(value: np.ndarray | float) -> np.ndarray | float:
        return 2.0 * np.pi * value / 1000.0

    right_axis = ax.secondary_yaxis("right", functions=(omega_to_hz, hz_to_omega))
    right_axis.set_ylabel("oscillation frequency (Hz)")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "v1_dominant_eigenvalue_imaginary_vs_wavenumber.png"
    pdf_path = output_dir / "v1_dominant_eigenvalue_imaginary_vs_wavenumber.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(
        f"p0={critical.p0:.12f}, k0={critical.mode}, "
        f"q0={critical_q_per_mm:.9f} rad/mm, "
        f"Im(lambda0)={critical_omega:.9f} rad/ms, "
        f"f0={critical_frequency_hz:.6f} Hz"
    )
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulations/output/v1_epvsst_vip"),
    )
    parser.add_argument("--samples", type=int, default=2001)
    args = parser.parse_args()
    plot(args.output_dir, samples=args.samples)


if __name__ == "__main__":
    main()
