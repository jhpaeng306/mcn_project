import numpy as np

from simulations.fft import compute_bifurcation
from simulations.pipeline import v1_epvsst_vip_preset
from simulations.pipeline_2d import compute_bifurcation_2d, v1_unit_summary
from simulations.simulation import Forcing, ModelParameters
from simulations.simulation_2d import Forcing2D, simulate_2d


def test_2d_forcing_lifts_the_1d_doubled_mode() -> None:
    forcing_1d = Forcing(
        temporal_amplitude=np.array([0.0, 0.2]),
        spatial_amplitude=np.array([0.0, 0.1]),
        temporal_omega=3.0,
        spatial_mode=6,
    )
    forcing_x = Forcing2D.from_1d(forcing_1d)
    forcing_y = Forcing2D.from_1d(forcing_1d, spatial_axis=1)
    assert forcing_x.spatial_mode == (6, 0)
    assert forcing_y.spatial_mode == (0, 6)
    assert forcing_x.temporal_omega == 3.0


def test_zero_field_remains_equilibrium_for_all_2d_boundaries() -> None:
    model = ModelParameters(
        K=1,
        A=np.array([[2.0]]),
        A_p=np.array([[0.0]]),
        Sigma=np.array([[0.4]]),
        tau=np.array([10.0]),
        theta=np.array([0.3]),
        gain=np.array([4.0]),
        length=4.0,
        names=("E",),
    )
    for boundary in ("periodic", "reflect", "zero"):
        result = simulate_2d(
            model,
            grid_points=12,
            total_time=1.0,
            dt=0.5,
            save_every=1,
            initial_scale=0.0,
            boundary=boundary,
        )
        assert result.u.shape == (3, 1, 12, 12)
        assert np.allclose(result.u, 0.0)
        assert np.allclose(result.spatial_rms, 0.0)


def test_v1_small_periodic_and_bounded_2d_runs_are_finite() -> None:
    model, _ = v1_epvsst_vip_preset()
    drive = np.array([1e-4, 1e-4, 0.0, 0.0])
    forcing = Forcing2D(
        temporal_amplitude=drive,
        spatial_amplitude=drive,
        temporal_omega=0.1,
        spatial_mode=(2, 0),
    )
    for boundary in ("periodic", "reflect"):
        result = simulate_2d(
            model,
            p=0.1,
            forcing=forcing,
            grid_points=16,
            total_time=2.0,
            dt=0.5,
            save_every=2,
            initial_scale=1e-5,
            boundary=boundary,
            diagnostic_wavevector=(1, 0),
        )
        assert result.u.shape == (3, model.K, 16, 16)
        assert result.probe.shape == (5, model.K)
        assert result.mode_signal.shape == (5, model.K)
        assert np.isfinite(result.u).all()
        assert np.isfinite(result.spatial_rms).all()


def test_v1_unit_summary_converts_ms_and_point_one_mm() -> None:
    model, settings = v1_epvsst_vip_preset()
    critical = compute_bifurcation(
        model, p_bounds=settings.p_bounds, max_mode=settings.max_mode
    )
    units = v1_unit_summary(model, critical)
    assert units["time_unit"] == "ms"
    assert units["one_space_unit_um"] == 100.0
    assert np.isclose(units["domain_side_mm"], 0.1 * model.length)
    assert np.isclose(
        units["critical_wavelength_mm"],
        0.1 * model.length / critical.mode,
    )
    assert np.isclose(
        units["stimulus_frequency_hz"], 2.0 * critical.frequency_hz
    )


def test_v1_2d_bifurcation_is_scanned_for_each_boundary_geometry() -> None:
    model, settings = v1_epvsst_vip_preset()
    periodic = compute_bifurcation_2d(
        model,
        boundary="periodic",
        p_bounds=settings.p_bounds,
        max_index=settings.max_mode,
    )
    reflecting = compute_bifurcation_2d(
        model,
        boundary="reflect",
        p_bounds=settings.p_bounds,
        max_index=settings.max_mode,
    )
    assert periodic.wavevector == (3, 1)
    assert reflecting.wavevector == (6, 3)
    assert periodic.periodic_mode_vector == (3.0, 1.0)
    assert reflecting.periodic_mode_vector == (3.0, 1.5)
    assert np.isclose(periodic.radial_mode, np.sqrt(10.0))
    assert np.isclose(reflecting.radial_mode, 0.5 * np.sqrt(45.0))
    assert periodic.point.omega0 > 0.0
    assert reflecting.point.omega0 > 0.0
