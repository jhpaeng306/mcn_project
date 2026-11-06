import numpy as np

import neural_field as nf


def test_v1_preset_has_finite_wave_hopf() -> None:
    model = nf.v1_three_population()
    critical = nf.find_critical_mode(model)
    assert critical.mode == 3
    assert abs(critical.eigenvalue.real) < 1e-9
    assert 13.5 < critical.frequency_hz < 15.0
    homogeneous = np.linalg.eigvals(nf.linear_symbol(model, 0.0, critical.scale))
    assert homogeneous.real.max() < 0.0
    assert np.allclose(np.vdot(critical.psi, critical.phi), 1.0)


def test_normal_form_selects_traveling_wave_locally() -> None:
    model = nf.v1_three_population()
    normal = nf.normal_form_coefficients(model, nf.find_critical_mode(model))
    assert normal.alpha.real > 0.0
    assert normal.b.real < 0.0
    assert normal.c.real < normal.b.real
    assert normal.pattern() == "traveling wave"


def test_forced_symmetric_jacobian_classification() -> None:
    states = nf.symmetric_forced_equilibria(
        a=0.1, b=-1.0, c=-0.5, eta_t=0.03, eta_x=0.02
    )
    nonzero = [state for state in states if abs(state["A"]) > 0]
    assert len(nonzero) == 2
    assert all(state["stable"] for state in nonzero)
    assert np.allclose(nonzero[0]["eigenvalues"], [-0.3, -0.2, -0.06, -0.04])


def test_all_real_forced_branches_and_exact_stability() -> None:
    states = nf.real_forced_equilibria(a=0.1, b=-1.0, c=-1.5)
    traveling = [
        state
        for state in states
        if state["branch"] == "exchange-broken"
        and np.isclose(state["A"] * state["B"], 0.0)
    ]
    assert len(traveling) == 4
    assert all(np.max(state["eigenvalues"]) <= 1e-12 for state in traveling)
    for state in states:
        A, B = state["A"], state["B"]
        assert np.isclose(A * (0.1 - A**2 - 1.5 * B**2), 0.0)
        assert np.isclose(B * (0.1 - B**2 - 1.5 * A**2), 0.0)


def test_small_simulations_cover_domains() -> None:
    model = nf.v1_three_population()
    critical = nf.find_critical_mode(model)
    for dimension, boundary in [
        (1, "periodic"),
        (1, "reflect"),
        (1, "zero"),
        (2, "periodic"),
    ]:
        result = nf.simulate(
            model,
            dimension=dimension,
            grid_points=24 if dimension == 1 else 16,
            boundary=boundary,
            coupling_scale=critical.scale,
            total_ms=3.0,
            dt_ms=0.5,
            save_every=2,
        )
        assert np.isfinite(result.u).all()
        assert result.u.shape[1] == model.populations


def test_retinotopic_patterns_are_static_and_finite_inside_disc() -> None:
    for kind in ("target", "pinwheel", "spiral", "honeycomb"):
        image = nf.retinotopic_pattern(kind, pixels=80)
        assert np.nanmin(image) >= -1.01
        assert np.nanmax(image) <= 1.01
        assert np.isnan(image[0, 0])


def test_functional_five_population_model_retains_pattern_instability() -> None:
    model = nf.v1_functional_five_population()
    critical = nf.find_critical_mode(model, scale_bracket=(0.3, 2.0))
    normal = nf.normal_form_coefficients(model, critical)
    assert model.names == ("E", "PV", "SST-GER", "SST-local", "VIP")
    assert critical.mode == 4
    assert 13.0 < critical.frequency_hz < 15.0
    assert normal.pattern() == "traveling wave"
    assert normal.c.real < normal.b.real < 0.0


def test_floquet_and_temporal_model_input_are_numerical_only() -> None:
    model = nf.v1_functional_five_population()
    critical = nf.find_critical_mode(model, scale_bracket=(0.3, 2.0))
    spectrum = nf.floquet_spectrum(
        model,
        coupling_scale=0.99 * critical.scale,
        drive_frequency_hz=2.0 * critical.frequency_hz,
        max_mode=5,
        settle_cycles=3,
        samples_per_cycle=20,
    )
    assert spectrum.growth_per_ms.shape == (6,)
    assert np.isfinite(spectrum.growth_per_ms).all()
    t, drive = nf.temporal_model_input(critical.frequency_hz, duration_ms=10.0)
    assert t.shape == drive.shape
    assert np.max(np.abs(drive)) <= 0.0250001


def test_k_population_and_mixed_gaussian_symbols() -> None:
    mu = np.array([[2.0, -1.0], [3.0, -2.0]])
    sigma = np.array([[0.5, 0.8], [0.6, 1.0]])
    q_squared = (2 / 4.0) ** 2 + 1.5**2
    expected = nf.gaussian_symbol(mu, sigma, q_squared)
    mixed = nf.mixed_gaussian_symbol(
        mu,
        sigma,
        orientation_mode=2,
        planar_wavenumber=1.5,
        orientation_scale=4.0,
    )
    assert np.allclose(mixed, expected)
    separate = nf.mixed_gaussian_symbol(
        mu,
        sigma,
        orientation_mode=2,
        planar_wavenumber=1.5,
        orientation_scale=4.0,
        planar_sigma=2.0 * sigma,
    )
    assert np.all(np.abs(separate) <= np.abs(mixed) + 1e-14)


def test_all_wave_support_equilibria_and_stability() -> None:
    states = nf.wave_support_equilibria(
        np.array([0.1 + 0.02j, 0.1 + 0.02j]),
        np.array([[-1.0 + 0.1j, -1.5], [-1.5, -1.0 + 0.1j]]),
    )
    stable = [state for state in states if state.stable_modulo_phase]
    assert {state.support for state in stable} == {(0,), (1,)}
    assert all(np.allclose(state.intensity.sum(), 0.1) for state in stable)
    standing = next(state for state in states if state.support == (0, 1))
    assert np.any(standing.active_eigenvalues.real > 0.0)
