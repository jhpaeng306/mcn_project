"""Read-only audit of the final report's V1 examples; writes only audit JSON.

Run from the project root: .venv/bin/python output/audits/v1_final_report/check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from simulations.v1_extension import (
    make_model, analyze_model, eigen_derivative, forcing_cases, integrate_cases,
    measure_1d, dump_json,
)
from simulations.fft import gaussian_symbol, linear_symbol, compute_forcing_coefficients
from simulations.pipeline import v1_epvsst_vip_preset

SOURCE = ROOT / "simulations/output/v1_extension"
records = json.loads((SOURCE / "calibration.json").read_text())
audit = {}
for K in (3, 4):
    record = records[str(K)]
    model = make_model(K, record["G"])
    critical, normal, *_ = analyze_model(model)
    p, cases = forcing_cases(model, normal)
    q, omega = critical.q, critical.omega0
    phi, psi = critical.phi, critical.psi

    # Independently evaluate the report's perpendicular and traveling formulas.
    def Q(v, w, wave):
        return gaussian_symbol(model, wave, critical.p0) @ (normal.f2*v*w/2) / model.tau

    def C(v, w, z, wave):
        return gaussian_symbol(model, wave, critical.p0) @ (normal.f3*v*w*z/6) / model.tau

    def R(freq, wave, rhs):
        return np.linalg.solve(1j*freq*np.eye(K)-linear_symbol(model, wave, critical.p0), rhs)

    h0 = 2*R(0, 0, Q(phi, phi.conj(), 0))
    h20 = 2*R(2*omega, np.sqrt(2)*q, Q(phi, phi, np.sqrt(2)*q))
    h11 = 2*R(0, np.sqrt(2)*q, Q(phi, phi.conj(), np.sqrt(2)*q))
    cperp = np.vdot(psi, 2*Q(phi, h0, q)+2*Q(phi.conj(), h20, q)
                     +2*Q(phi, h11, q)+6*C(phi, phi, phi.conj(), q))
    drive = np.zeros(K)
    drive[:2] = 1
    eta_unit = 2*np.vdot(psi, Q(phi.conj(), R(2*omega, 2*q, drive/model.tau/2), q))
    eps2 = critical.p0-p
    eta_w = eta_unit*cases["temporal"]["temporal"]
    a = normal.alpha*(p-critical.p0)
    roots = np.roots([abs(normal.b)**2, 2*(a*normal.b.conjugate()).real,
                     abs(a)**2-abs(eta_w)**2])
    rho = float(max(roots.real))
    fc = compute_forcing_coefficients(model, normal,
        temporal_harmonic=drive*cases["temporal"]["temporal"]/2)
    item = {
        "p_notebook_onset": critical.p0,
        "p_report_run": p-.25,
        "critical_mode": critical.mode,
        "omega_rad_per_ms": omega,
        "frequency_hz": critical.frequency_hz,
        "homogeneous_growth_at_onset": float(np.linalg.eigvals(linear_symbol(model, 0, critical.p0)).real.max()),
        "frequency_slope_at_onset": eigen_derivative(model, q, critical.p0).imag,
        "alpha": normal.alpha, "b": normal.b, "c": normal.c,
        "c_perp": cperp,
        "traveling_scaled_peak_threshold_mu_minus_one": abs(normal.alpha)/abs(eta_unit),
        "traveling_actual_scaled_peak": cases["temporal"]["temporal"]/eps2,
        "traveling_predicted_physical_amplitude": np.sqrt(rho),
        "traveling_stability_real_parts": [
            (a+2*normal.b*rho).real, (a+normal.c*rho).real, (a+cperp*rho).real],
        "temporal_existence_margin_abs_eta_minus_abs_a": abs(fc.eta_t)-abs(a),
        "forcing": cases,
    }
    assert abs(critical.p0-.25) < 1e-9
    assert abs(omega-.08) < 1e-9
    expected_slope, expected_cperp, expected_threshold = {
        3: (.0331673, -15.34933, 1.05854),
        4: (.0457008, -16.42225, 1.02588),
    }[K]
    assert abs(item["frequency_slope_at_onset"]-expected_slope) < 5e-8
    assert abs(cperp.real-expected_cperp) < 5e-6
    assert abs(item["traveling_scaled_peak_threshold_mu_minus_one"]-expected_threshold) < 5e-6
    assert item["temporal_existence_margin_abs_eta_minus_abs_a"] > 0
    assert max(item["traveling_stability_real_parts"]) < 0
    for dim in (1, 2):
        with np.load(SOURCE / f"field_{dim}d_{K}pop.npz", allow_pickle=False) as saved:
            cached = dict(saved)
        cached["labels"] = cached["labels"].tolist()
        for key, expected in (("G", model.A_p), ("tau", model.tau),
                              ("Sigma", model.Sigma), ("gain", model.gain),
                              ("theta", model.theta)):
            np.testing.assert_allclose(cached[key], expected, atol=1e-12, rtol=1e-12)
        assert abs(float(cached["p"])-p) < 1e-10
        if dim == 1:
            protocol = cases
            item["saved_1d_metrics"] = measure_1d(cached, critical)
        else:
            protocol = {
                "axis exact": {"traveling": cases["temporal"]["temporal"]},
                "axis + y seed": {"traveling": cases["temporal"]["temporal"]},
                "static + y seed": {"spatial": cases["spatial"]["spatial"]},
                "temporal + y seed": {"temporal": cases["temporal"]["temporal"]},
                "unforced + y seed": {},
            }
            item["saved_2d_transverse_rms_initial"] = cached["transverse_rms"][0]
            item["saved_2d_transverse_rms_final"] = cached["transverse_rms"][-1]
            mode = cached["modes"][:, 0, :, 1]
            tail = cached["t"] >= .8*cached["t"][-1]
            plus = mode @ psi.conj()
            minus = mode.conj() @ psi.conj()
            item["saved_traveling_tail_A_plus"] = float(np.mean(abs(plus[tail])))
            item["saved_traveling_tail_A_minus"] = float(np.mean(abs(minus[tail])))
        replay = integrate_cases(model, critical, protocol, p=p, dim=dim,
            n=int(cached["n"]), duration=1000, dt=float(cached["dt"]),
            save_ms=10 if dim == 1 else 20, exact_axis=("axis exact",))
        count = len(replay["t"])
        error = float(np.max(abs(replay["modes"]-cached["modes"][:count])))
        item[f"replay_{dim}d_first_1000ms_max_modal_error"] = error
        np.testing.assert_allclose(replay["modes"], cached["modes"][:count], atol=1e-12, rtol=1e-10)
        item[f"saved_{dim}d_settings"] = {key: cached[key] for key in
            ("n", "dt", "duration", "p", "labels")}
    audit[str(K)] = item
    print(f"K={K}: report values and cached replay checks passed", flush=True)

legacy, settings = v1_epvsst_vip_preset()
audit["legacy_pipeline_difference"] = {
    "max_absolute_G_difference_from_report_K4": float(np.max(abs(legacy.A_p-np.array(records["4"]["G"])))),
    "p_margin": settings.p_margin,
    "growth_target": settings.forcing_target_growth,
}
dump_json(Path(__file__).with_name("checks.json"), audit)
