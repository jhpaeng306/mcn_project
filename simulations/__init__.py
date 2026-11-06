"""General K-population neural-field simulations and Hopf analysis."""

from .amplitude import (
    AmplitudeTrajectory,
    classify_wave,
    integrate_amplitudes,
    project_critical_amplitudes,
    zero_state_eigenvalues,
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
from .simulation import (
    Forcing,
    ModelParameters,
    SimulationResult,
    activation,
    activation_derivatives,
    simulate,
)
from .simulation_2d import (
    Forcing2D,
    SimulationResult2D,
    dominant_spatial_mode,
    dominant_temporal_frequency_hz,
    simulate_2d,
)

__all__ = [
    "AmplitudeTrajectory",
    "BifurcationPoint",
    "Forcing",
    "ForcingCoefficients",
    "Forcing2D",
    "ModelParameters",
    "NormalFormCoefficients",
    "SimulationResult",
    "SimulationResult2D",
    "activation",
    "activation_derivatives",
    "classify_wave",
    "compute_bifurcation",
    "compute_forcing_coefficients",
    "compute_normal_form",
    "dominant_spatial_mode",
    "dominant_temporal_frequency_hz",
    "integrate_amplitudes",
    "linear_symbol",
    "modal_spectrum",
    "project_critical_amplitudes",
    "simulate",
    "simulate_2d",
    "zero_state_eigenvalues",
    "zero_state_growth_rate",
]
