# %% ----- imports
import sys

sys.path.append("../")  # change to your pynlo path
from scipy.constants import c
from helpers import geom_factors
from edf.re_nlse_joint_5level import EDF
import pynlo
import pynlo.utility.clipboard
import numpy as np
import matplotlib.pyplot as plt
from edf import edfa
import collections
from edf.utility import crossSection, ER80_4_125_betas


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


# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% -------------- load dispersion coefficients ------------------------------
polyfit_n = ER80_4_125_betas().polyfit

# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.7 / 10)
loss_mat = 10 ** (-1 / 10)

f_r = 200e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 35e-3 / 2 / f_r

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

# passive fiber (PM1550) for passive NLSE segment gamma0
pm1550_a_core_m = (8.5e-6)/2 # Coherent's PM1550-XP
pm1550_NA       = 0.125

# ---- gamma0 for passive PM1550 propagation (only scalar needed) ----
gamma0_pm1550_m,_,_ = geom_factors(
    pulse.v0, pulse.v_grid,
    a_core_m=pm1550_a_core_m,
    NA=pm1550_NA,
)
print(gamma0_pm1550_m)
# %% ---------- optional passive fiber ----------------------------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma0_pm1550_m

length_pm1550 = 1.119
# ignore numpy error if length = 0.0, it occurs when n_records is not None and
# propagation length is 0, the output pulse is still correct
model_pm1550, sim_pm1550 = propagate(pm1550, pulse, length_pm1550)
pulse_pm1550 = sim_pm1550.pulse_out

# %% ------------ active fiber ------------------------------------------------
# active EDF fiber geometry (signal mode)
edf_a_core_m = (4e-6)/2 # Er80-4/125
edf_A_doped = np.pi * (edf_a_core_m ** 2)  # [m^2] doped area
edf_NA       = 0.2

# pump config (for EDF rate equations)
pump_is_cladding = False     # True/False
pump_a_clad_m    = ...     # [m] only if pump_is_cladding=True

gamma0_edf_m, overlap_s, overlap_p = geom_factors(
    pulse.v0, pulse.v_grid,
    a_core_m=edf_a_core_m,
    NA=edf_NA,
    pump_is_cladding=pump_is_cladding,
    a_clad_m=pump_a_clad_m,
)
print(gamma0_edf_m,overlap_p,overlap_s)
n_ion_n = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

length = 1.5

edf = EDF(
    f_r=f_r,
    overlap_p=overlap_p,
    overlap_s=overlap_s,
    n_ion=n_ion_n,
    A_doped=edf_A_doped,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit_n)
beta_n = edf._beta(pulse.v_grid)
edf.gamma = gamma0_edf_m

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

# keep all figures on screen
plt.show()