# %% ----- imports
import sys

sys.path.append("../")
from scipy.constants import c
from helpers import geom_factors
from ydf.re_nlse_joint_2level import YDF
#from edf.re_nlse_joint_5level import EDF
import pynlo
import pynlo.utility.clipboard
import numpy as np
import matplotlib.pyplot as plt
#from edf import edfa
import collections
from ydf.utility import crossSection, PM980_XP_betas


ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])


def propagate(fiber, pulse, length, n_records=None):
    """
    propagates a given pulse through fiber of given length

    Args:
        fiber (instance of SilicaFiber): Fiber
        pulse (instance of Pulse): Pulse
        length (float): fiber elngth

    Returns:
        output: model, sim
    """
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.7 / 10)
loss_mat = 10 ** (-1 / 10)

f_r = 200e6
n = 256
v_min = c / 1150e-9
v_max = c / 910e-9
v0 = c / 1030e-9
e_p = 35e-3 / 2 / f_r
temperature = 300

t_fwhm = 100e-15
min_time_window = 10e-12
pulse = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
    alias=2,
)

# %% -------------- load absorption coefficients from Topper et al. ------------------
spl_sigma_a = crossSection(temperature).sigma_a
spl_sigma_e = crossSection(temperature).sigma_e

# %% -------------- load dispersion coefficients ------------------------------
polyfit_n = PM980_XP_betas(v0).polyfit

# passive fiber (PM980) for passive NLSE segment gamma0
pm980_a_core_m = (5.5e-6)/2 # Coherent's PM980-XP
pm980_NA       = 0.12

# ---- gamma0 for passive PM980 propagation (only scalar needed) ----
gamma0_pm980_m,_,_ = geom_factors(
    pulse.v0, pulse.v_grid,
    a_core_m=pm980_a_core_m,
    NA=pm980_NA,
)

# %% ---------- optional passive fiber ----------------------------------------
pm980 = pynlo.materials.SilicaFiber()
beta_n_pm980 = PM980_XP_betas(v0).polyfit
pm980.set_beta_from_beta_n(v0, beta_n_pm980)
pm980.gamma = gamma0_pm980_m

length_pm980 = 1.119
# ignore numpy error if length = 0.0, it occurs when n_records is not None and
# propagation length is 0, the output pulse is still correct
model_pm980, sim_pm980 = propagate(pm980, pulse, length_pm980)
pulse_pm980 = sim_pm980.pulse_out

# %% ------------ active fiber ------------------------------------------------
# active YDF fiber geometry (signal mode)
ydf_a_core_m = (6e-6)/2 # Yb1200-6/125DC
ydf_A_doped = np.pi * (ydf_a_core_m ** 2)  # [m^2] doped area
ydf_NA       = 0.12

# pump config (for EDF rate equations)
pump_is_cladding = True     # True/False
pump_a_clad_m    = 125e-6     # [m] only if pump_is_cladding=True

gamma0_ydf_m, overlap_s, overlap_p = geom_factors(
    pulse.v0, pulse.v_grid,
    a_core_m=ydf_a_core_m,
    NA=ydf_NA,
    pump_is_cladding=pump_is_cladding,
    a_clad_m=pump_a_clad_m,
)

n_ion_n = 9.6e25 # obtained from Thorlabs representative for (nLIGHT's) Yb1200-6/125DC

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

length = 1.5

ydf = YDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion_n,
    a_eff=a_eff_n,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit_n)
beta_n = edf._beta(pulse.v_grid)
edf.gamma = gamma_n

# %% ----------- edfa ---------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
    p_fwd=pulse_pm1550,
    p_bck=None,
    edf=edf,
    length=length,
    Pp_fwd=2 * loss_ins * loss_spl,
    Pp_bck=2 * loss_ins * loss_spl,
    n_records=100,
)
sim = sim_fwd

# %% ----------- plot results -------------------------------------------------
sol_Pp = sim.Pp
sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1)
z = sim.z
n1 = sim.n1_n
n2 = sim.n2_n
n3 = sim.n3_n
n4 = sim.n4_n
n5 = sim.n5_n

fig = plt.figure(
    num=f"power evolution for {length} normal edf and {length_pm1550} pm1550 pre-chirp",
    figsize=np.array([11.16, 5.21]),
)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(z, sol_Pp, label="pump", linewidth=2)
ax1.plot(z, sol_Ps * loss_ins * loss_spl, label="signal", linewidth=2)
ax1.grid()
ax1.legend(loc="upper left")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

ax2.plot(z, n1, label="n1", linewidth=2)
ax2.plot(z, n2, label="n2", linewidth=2)
ax2.plot(z, n3, label="n3", linewidth=2)
ax2.plot(z, n4, label="n4", linewidth=2)
ax2.plot(z, n5, label="n5", linewidth=2)
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")
fig.tight_layout()

sim.plot(
    "wvl",
    num=f"spectral evolution for {length} normal edf and {length_pm1550} pm1550 pre-chirp",
)
