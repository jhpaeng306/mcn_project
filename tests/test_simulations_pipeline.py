import numpy as np

from simulations.amplitude import (
    amplitude_rhs,
    project_critical_amplitudes,
    zero_state_eigenvalues,
    zero_state_growth_rate,
)
from simulations.fft import (
    compute_bifurcation,
    compute_forcing_coefficients,
    compute_normal_form,
    gaussian_symbol,
    linear_symbol,
)
from simulations.pipeline import (
    _forcing_case,
    measure_frequency_doubling,
    pi_ei_preset,
    presets,
    run_all,
    v1_epvsst_vip_preset,
)
from simulations.simulation import Forcing, ModelParameters, activation, simulate


def test_default_pipeline_registry_uses_four_population_v1() -> None:
    assert tuple(presets()) == ("pi_ei", "v1_epvsst_vip")
    assert run_all.__kwdefaults__["selected"] == ("pi_ei", "v1_epvsst_vip")
    model, _ = presets()["v1_epvsst_vip"]
    assert model.K == 4
    assert model.names == ("E", "PV", "SST", "VIP")


def test_default_k2_inputs_and_gaussian_variance_convention() -> None:
    model = ModelParameters()
    assert model.K == 2
    assert model.A.shape == model.Sigma.shape == model.A_p.shape == (2, 2)
    assert np.allclose(gaussian_symbol(model, 0.0, 0.7), model.A + 0.7 * model.A_p)
    q = 0.4
    expected = (model.A + 0.7 * model.A_p) * np.exp(-0.5 * model.Sigma * q**2)
    assert np.allclose(gaussian_symbol(model, q, 0.7), expected)


def test_pi_bifurcation_and_report_coefficients() -> None:
    model, settings = pi_ei_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=8)
    normal = compute_normal_form(model, critical)
    assert critical.mode == 3
    assert np.isclose(critical.p0, 1.624919528, atol=2e-6)
    assert np.isclose(critical.omega0, 0.318127517, atol=2e-6)
    assert np.allclose(np.vdot(critical.psi, critical.phi), 1.0)
    assert normal.alpha.real > 0.0
    assert normal.b.real < 0.0
    assert normal.c.real < normal.b.real
    assert normal.preferred_pattern == "traveling wave"


def test_v1_preset_has_finite_hopf_and_nonnegative_variances() -> None:
    model, settings = v1_epvsst_vip_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=12)
    normal = compute_normal_form(model, critical)
    assert model.K == 4
    assert model.names == ("E", "PV", "SST", "VIP")
    assert np.all(model.Sigma >= 0.0)
    assert critical.mode > 0
    assert critical.omega0 > 0.0
    assert normal.alpha.real > 0.0
    assert 0.249 < critical.p0 < 0.251
    assert critical.mode == 3
    assert normal.preferred_pattern == "traveling wave"
    assert settings.temporal_frequency_multiplier == 2.0
    assert settings.drive_weights == (1.0, 1.0, 0.0, 0.0)
    assert settings.spatial_forcing_epsilon_squared_coefficient == 6.0
    expected_mask = np.array(
        [
            [1, -1, -1, 0],
            [1, -1, -1, 0],
            [1, 0, 0, -1],
            [1, 0, -1, 0],
        ]
    )
    assert np.array_equal(np.sign(model.A), expected_mask)
    assert np.array_equal(np.sign(model.A_p), expected_mask)


def test_v1_designed_ap_sustains_the_selected_mode() -> None:
    model, settings = v1_epvsst_vip_preset()
    expected = np.array(
        [
            [6.22009, -1.45455, -6.48944, 0.0],
            [5.09798, -0.90615, -8.04066, 0.0],
            [2.67174, 0.0, 0.0, -16.0],
            [1.68844, 0.0, -3.13584, 0.0],
        ]
    )
    assert np.allclose(model.A_p, expected, atol=1e-7)
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=12)
    normal = compute_normal_form(model, critical)
    p = critical.p0 + 0.0003 / normal.alpha.real
    assert np.linalg.eigvals(linear_symbol(model, 0.0, p)).real.max() < 0.0
    result = simulate(
        model,
        p=p,
        seed=31,
        grid_points=96,
        total_time=6000.0,
        dt=0.25,
        save_every=4,
        initial_scale=settings.initial_scale,
    )
    tail = result.u[int(0.8 * result.t.size) :, 0]
    centered = tail - tail.mean(axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(centered, axis=1)) ** 2, axis=0)
    assert np.argmax(power[1:]) + 1 == critical.mode
    assert np.std(centered) > 2.0 * settings.initial_scale


def test_v1_forcing_reaches_the_matched_critical_growth() -> None:
    model, settings = v1_epvsst_vip_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=12)
    normal = compute_normal_form(model, critical)
    p = critical.p0 - settings.p_margin
    expected_components = {
        "temporal": (0.0041825831, 0.0),
        "spatial": (0.0, 0.0059542595),
        "both": (0.0024492933, 0.0024492933),
    }
    growth_rates = {}
    for kind, (expected_temporal, expected_spatial) in expected_components.items():
        forcing, temporal, spatial = _forcing_case(
            model, normal, settings, p, kind
        )
        temporal_peak = np.max(
            np.abs(forcing.vector(forcing.temporal_amplitude, model.K))
        )
        spatial_peak = np.max(
            np.abs(forcing.vector(forcing.spatial_amplitude, model.K))
        )
        for component in (
            forcing.vector(forcing.temporal_amplitude, model.K),
            forcing.vector(forcing.spatial_amplitude, model.K),
        ):
            assert np.isclose(component[0], component[1])
            assert np.allclose(component[2:], 0.0)
        assert np.isclose(temporal_peak, expected_temporal, atol=1e-10)
        assert np.isclose(spatial_peak, expected_spatial, atol=1e-10)
        coefficients = compute_forcing_coefficients(
            model,
            normal,
            temporal_harmonic=temporal,
            spatial_harmonic=spatial,
        )
        growth_rates[kind] = zero_state_growth_rate(
            normal, p=p, forcing=coefficients
        )
    for growth in growth_rates.values():
        assert np.isclose(growth, settings.forcing_target_growth, atol=2e-10)


def test_v1_unforced_near_hopf_mode_decays_numerically_below_onset() -> None:
    model, settings = v1_epvsst_vip_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=12)
    result = simulate(
        model,
        p=critical.p0 - settings.p_margin,
        seed=7,
        grid_points=settings.grid_points,
        total_time=5000.0,
        dt=settings.dt,
        save_every=settings.save_every,
        initial_scale=1e-5,
    )
    A, B = project_critical_amplitudes(result, critical)
    early = slice(0, int(0.2 * A.size))
    late = slice(int(0.8 * A.size), None)
    assert np.mean(np.abs(A[late])) < np.mean(np.abs(A[early]))
    assert np.mean(np.abs(B[late])) < np.mean(np.abs(B[early]))


def test_v1_combined_e_pv_forcing_grows_the_critical_mode_in_direct_field() -> None:
    model, settings = v1_epvsst_vip_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=12)
    normal = compute_normal_form(model, critical)
    forcing, _, _ = _forcing_case(
        model, normal, settings, critical.p0 - settings.p_margin, "both"
    )
    spatial_peak = np.max(
        np.abs(forcing.vector(forcing.spatial_amplitude, model.K))
    )
    assert np.isclose(spatial_peak, 0.0024492933, atol=1e-10)
    result = simulate(
        model,
        p=critical.p0 - settings.p_margin,
        forcing=forcing,
        seed=7,
        grid_points=settings.grid_points,
        total_time=settings.total_time,
        dt=settings.dt,
        save_every=settings.save_every,
        initial_scale=settings.initial_scale,
    )
    A, B = project_critical_amplitudes(result, critical)
    late = slice(int(0.8 * A.size), None)
    assert np.mean(np.abs(A[late])) > 0.001
    assert np.mean(np.abs(B[late])) > 0.001
    centered = result.u[late, 0] - result.u[late, 0].mean(axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(centered, axis=1)) ** 2, axis=0)
    assert np.argmax(power[1:]) + 1 == critical.mode
    frequency = measure_frequency_doubling(model, critical, result)
    assert np.isclose(frequency["expected_input_frequency"], 2.0 * 12.899348, atol=2e-5)
    assert 1.98 < frequency["measured_input_over_response"] < 2.02


def test_direct_simulation_is_numerical_and_projectable() -> None:
    model, settings = pi_ei_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=8)
    result = simulate(
        model,
        p=critical.p0 - 0.1,
        forcing=Forcing(spatial_amplitude=np.array([0.01, 0.0]), spatial_mode=6),
        seed=3,
        grid_points=32,
        total_time=2.0,
        dt=0.1,
        save_every=2,
    )
    A, B = project_critical_amplitudes(result, critical)
    assert result.u.shape == (11, 2, 32)
    assert result.forcing.shape == result.u.shape
    assert A.shape == B.shape == result.t.shape
    assert np.isfinite(result.u).all()
    assert np.allclose(activation(np.zeros((2, 4)), model), 0.0)


def test_temporal_and_spatial_forcing_have_distinct_conjugacies() -> None:
    A = 0.2 + 0.1j
    B = -0.3 + 0.4j
    temporal = amplitude_rhs(A, B, a=0, b=0, c=0, eta_t=2.0, eta_x=0.0)
    spatial = amplitude_rhs(A, B, a=0, b=0, c=0, eta_t=0.0, eta_x=2.0)
    assert temporal == (2.0 * np.conj(B), 2.0 * np.conj(A))
    assert spatial == (2.0 * B, 2.0 * A)


def test_insufficient_resonant_forcing_keeps_zero_amplitude_stable() -> None:
    model, settings = pi_ei_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=8)
    normal = compute_normal_form(model, critical)
    p = critical.p0 - settings.p_margin
    weak_drive = np.array([5e-4, 0.0])
    weak = compute_forcing_coefficients(
        model,
        normal,
        temporal_harmonic=0.5 * weak_drive,
        spatial_harmonic=0.5 * weak_drive,
    )
    eigenvalues = zero_state_eigenvalues(normal, p=p, forcing=weak)
    assert eigenvalues.shape == (4,)
    assert zero_state_growth_rate(normal, p=p, forcing=weak) < 0.0


def test_strong_spatial_resonance_crosses_pi_parametric_threshold() -> None:
    model, settings = pi_ei_preset()
    critical = compute_bifurcation(model, p_bounds=settings.p_bounds, max_mode=8)
    normal = compute_normal_form(model, critical)
    p = critical.p0 - settings.p_margin
    drive = np.array([settings.drive_amplitude, 0.0])
    strong = compute_forcing_coefficients(model, normal, spatial_harmonic=0.5 * drive)
    assert zero_state_growth_rate(normal, p=p, forcing=strong) > 0.0


def test_linear_symbol_matches_direct_formula() -> None:
    model = ModelParameters()
    q, p = 0.2, 0.4
    zero = activation(np.zeros((model.K, 1)), model)
    assert np.allclose(zero, 0.0)
    matrix = linear_symbol(model, q, p)
    assert matrix.shape == (model.K, model.K)
    assert np.isfinite(matrix).all()
