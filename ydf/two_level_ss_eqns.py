"""
this scratch file is to re-create just the rate equations (not PyNLO) but this
time including pump excited state absorption (ESA).
"""

# %% ----- imports
import numpy as np
from scipy.constants import h

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3

nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

# lifetime
tau = 860e-6 # from Lindberg et al. (see below for exact reference); also agrees well 'Temperature Effects on the Emission Properties of Yb-doped optical fibers' by Newell et al. (2006)

# -----------------------------------------------------------------------------
# solving for the 2 level system with levels coupled the same way as:
#
#   Lindberg et al. Scientific Reports 6, 34742 (2016).
#
#   sigma = overlap * sigma * P / (h * nu * A)
# -----------------------------------------------------------------------------


def _factor_sigma(sigma, nu, P, overlap, A):
    return overlap * sigma * P / (h * nu * A)

def n2_func(
    n,
    A_doped,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p_a,
    sigma_p_e,
    sigma_a,
    sigma_e,
    tau,
):
    sigma_a_p_s = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, A_doped)
    sigma_e_p_s = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, A_doped)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sum_a_p_s = np.sum(sigma_a_p_s)
        sum_e_p_s = np.sum(sigma_e_p_s)
    
    return _n2_func(
        n,
        A_doped,
        overlap_p,
        nu_p,
        P_p,
        sigma_p_a,
        sigma_p_e,
        sum_a_p_s,
        sum_e_p_s,
        tau,
        )

def _n2_func(
    n,
    A_doped,
    overlap_p,
    nu_p,
    P_p,
    sigma_p_a,
    sigma_p_e,
    sum_a_p_s,
    sum_e_p_s,
    tau,
):
    return (sum_a_p_s + _factor_sigma(sigma_p_a, nu_p, P_p, overlap_p, A_doped)) * n / (sum_a_p_s + sum_e_p_s + _factor_sigma(sigma_p_a + sigma_p_e, nu_p, P_p, overlap_p, A_doped) + 1 / tau)

def dPp_dz(
    n,
    A_doped,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p_a,
    sigma_p_e,
    sigma_a,
    sigma_e,
    tau,
):
    args = [
        n,
        A_doped,
        overlap_p,
        overlap_s,
        nu_p,
        P_p,
        nu_s,
        P_s,
        sigma_p_a,
        sigma_p_e,
        sigma_a,
        sigma_e,
        tau,
    ]
    n2 = n2_func(*args)

    return overlap_p * (sigma_p_e * n2 + sigma_p_a * n2 - sigma_p_a * n) * P_p

def dPs_dz(
    n,
    A_doped,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p_a,
    sigma_p_e,
    sigma_a,
    sigma_e,
    tau,
):
    args = [
        n,
        A_doped,
        overlap_p,
        overlap_s,
        nu_p,
        P_p,
        nu_s,
        P_s,
        sigma_p_a,
        sigma_p_e,
        sigma_a,
        sigma_e,
        tau,
    ]
    n2 = n2_func(*args)

    return overlap_s * (sigma_e * n2 + sigma_a * n2 - sigma_a * n) * P_s


# same as dPs_dz but without the multiplying factor of P_s
def gain(
    n,
    A_doped,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    sigma_p_a,
    sigma_p_e,
    sigma_a,
    sigma_e,
    tau,
):
    args = [
        n,
        A_doped,
        overlap_p,
        overlap_s,
        nu_p,
        P_p,
        nu_s,
        P_s,
        sigma_p_a,
        sigma_p_e,
        sigma_a,
        sigma_e,
        tau,
    ]
    n2 = n2_func(*args)

    return overlap_s * (sigma_e * n2 + sigma_a * n2 - sigma_a * n)