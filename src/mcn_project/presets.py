"""Parameter sets. Edit numbers here; every notebook and test reads them."""

import numpy as np

from .model import NeuralField

# Two-population E-I Wilson-Cowan field from the PI's wc2dn code
# (archive/legacy_code/wc2dn.py). Kernel exp(-(x/sigma)^2) has variance sigma^2/2.
# Sigmoid 1/(1+exp(-4u)) gives gain 4. kappa = a_EE; one length unit = one grid cell.
WC = dict(aie=10.0, aei=10.0, aii=8.0, sige=4.0, sigi=8.0, te=0.5, ti=0.3,
          taue=3.0, taui=6.6, length=60.0)

# Four-population V1 model (paper Sec. 7). Time in ms, one length unit = 0.1 mm.
# Billeh et al. (2020) distance rule exp(-r^2/s^2) gives Sigma = (s / 100 um)^2 / 2.
V1_S_UM = np.array([[114.0, 95.0, 85.0, 85.0],
                    [92.0, 120.0, 120.0, 120.0],
                    [103.0, 120.0, 120.0, 120.0],
                    [103.0, 120.0, 120.0, 120.0]])
V1_G = np.array([[6.22009, -1.45455, -6.48944, 0.0],
                 [5.09798, -0.90615, -8.04066, 0.0],
                 [2.67174, 0.0, 0.0, -16.0],
                 [1.68844, 0.0, -3.13584, 0.0]])
V1_TAU = np.array([10.0, 12.0, 35.0, 20.0])
V1_GAIN = np.array([3.47682761, 7.34413013, 3.74650527, 6.0])
V1_THETA = np.array([0.28483799, 0.12068148, 0.40137356, 0.7])
V1_LENGTH = 43.0
V1_DRIVE = np.array([1.0, 1.0, 0.0, 0.0])  # visual input to E and PV
# E/PV/SST block with E->E raised 6.22009 -> 6.72 so that the first onset has k0 = 3.
V1_G3 = np.array([[6.72, -1.45455, -6.48944],
                  [5.09798, -0.90615, -8.04066],
                  [2.67174, 0.0, 0.0]])


def wilson_cowan(length: float = WC["length"]) -> NeuralField:
    p = WC
    return NeuralField(
        J0=np.array([[0.0, -p["aie"]], [p["aei"], -p["aii"]]]),
        J1=np.array([[1.0, 0.0], [0.0, 0.0]]),
        Sigma=np.array([[p["sige"], p["sigi"]], [p["sige"], p["sigi"]]]) ** 2 / 2,
        tau=np.array([p["taue"], p["taui"]]),
        gain=np.array([4.0, 4.0]),
        theta=np.array([p["te"], p["ti"]]),
        length=length,
        names=("E", "I"),
        drive=np.array([1.0, 0.0]),
    )


def v1(K: int = 4, length: float = V1_LENGTH) -> NeuralField:
    """kappa scales the gain matrix: J(kappa) = kappa G."""
    G = {4: V1_G, 3: V1_G3}[K]
    return NeuralField(
        J0=np.zeros((K, K)),
        J1=G.copy(),
        Sigma=(V1_S_UM[:K, :K] / 100) ** 2 / 2,
        tau=V1_TAU[:K].copy(),
        gain=V1_GAIN[:K].copy(),
        theta=V1_THETA[:K].copy(),
        length=length,
        names=("E", "PV", "SST", "VIP")[:K],
        drive=V1_DRIVE[:K].copy(),
    )
