import numpy as np
from scipy.constants import c, pi

def marcuse_w_from_v(v_hz, *, a_core_m, NA):
    lam = c / v_hz
    V = (2 * pi / lam) * a_core_m * NA
    return a_core_m * (0.65 + 1.619 / (V**1.5) + 2.879 / (V**6))

def geom_two_areas(v0_hz, v_grid_hz, *,
                      a_core_m, NA, 
                      n2_m2_per_W=2.6e-20,
                      pump_lambda_m=980e-9,
                      pump_is_cladding=False,
                      a_clad_m=None):
    """
    Outputs:
      gamma0     : Kerr gamma at v0 using Aeff_s(v0) (for NLSE)
      a_eff_s    : array Aeff_s(v_grid) = pi*w(v)^2 (for signal photon-flux in sums)
      overlap_s  : array Gamma_s(v_grid) = 1-exp(-2*(a_core/w(v))^2)
      a_eff_p    : scalar pump denominator area used in pump transition rates
      overlap_p  : scalar pump overlap factor

    Pump handling:
      - core pumped (a_clad_m is None): a_eff_p = Aeff_p_core, overlap_p = Gamma_p_core
      - cladding pumped (a_clad_m given): a_eff_p = A_clad, overlap_p = 1
    """
    # --- signal mode area vs frequency grid ---
    w_s = marcuse_w_from_v(v_grid_hz, a_core_m=a_core_m, NA=NA)
    a_eff_s = pi * w_s**2
    overlap_s = 1.0 - np.exp(-2.0 * (a_core_m / w_s)**2)

    # --- gamma at center (use mode area at v0) ---
    w0 = marcuse_w_from_v(v0_hz, a_core_m=a_core_m, NA=NA)
    A0 = pi * w0**2
    lam0 = c / v0_hz
    gamma0 = (2 * pi / lam0) * n2_m2_per_W / A0  # 1/(W*m)

    # ---- pump area/overlap ----
    if pump_is_cladding:
        if a_clad_m is None:
            raise ValueError("pump_is_cladding=True requires a_clad_m (inner cladding radius in meters).")
        a_eff_p = pi * a_clad_m**2
        overlap_p = 1.0
    else:
        v_p = c / pump_lambda_m
        w_p = marcuse_w_from_v(v_p, a_core_m=a_core_m, NA=NA)
        a_eff_p = pi * w_p**2
        overlap_p = 1.0 - np.exp(-2.0 * (a_core_m / w_p)**2)

    return gamma0, a_eff_s, overlap_s, a_eff_p, overlap_p