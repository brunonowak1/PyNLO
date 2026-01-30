# %% ----- imports
from ydf import utility
import pathlib
import pandas as pd
from scipy.constants import c
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

path = pathlib.Path(utility.__file__).parent

# %% ----- absorption and emission cross-sections in K -----
class crossSection:
    def __init__(self, temperature = 300):
        self.temperature = float(temperature)

        # --- load absorption spreadsheet ---
        df_a = pd.read_excel(path / "Topper_provided_cross_sections/Topper_et_al._YDF_absorption_cross_sections.xlsx")
        arr_a = df_a.to_numpy()

        # Row 0: units, Row 1: temperatures across columns 1..end, Rows 2..: data
        T_grid = arr_a[1, 1:].astype(float)                 # shape (nT,)
        lam_nm_a = arr_a[2:, 0].astype(float)                 # shape (nlam,)
        sig_a_Tgrid = arr_a[2:, 1:].astype(float)           # shape (nlam, nT)

        # --- load emission spreadsheet ---
        df_e = pd.read_excel(path / "Topper_provided_cross_sections/Topper_et_al._YDF_emission_cross_sections.xlsx")
        arr_e = df_e.to_numpy()

        lam_nm_e = arr_e[2:, 0].astype(float) 
        sig_e_Tgrid = arr_e[2:, 1:].astype(float)           # shape (nlam, nT)            

        # --- interpolate in temperature (per wavelength row, all at once) ---
        # interp1d can interpolate along an axis; here axis=1 is temperature.
        sig_a = interp1d(T_grid, sig_a_Tgrid, axis=1, bounds_error=False, 
            fill_value=(sig_a_Tgrid[:, 0], sig_a_Tgrid[:, -1]))(self.temperature)
        sig_e = interp1d(T_grid, sig_e_Tgrid, axis=1, bounds_error=False,
            fill_value=(sig_e_Tgrid[:, 0], sig_e_Tgrid[:, -1]))(self.temperature)

        # wavelength in meters
        lam_m_a = lam_nm_a * 1e-9
        lam_m_e = lam_nm_e * 1e-9

        # --- build sigma(v) callables (same as before) ---
        self.sigma_a = interp1d(
            c / lam_m_a[::-1], sig_a[::-1], bounds_error=False, fill_value=0.0
        )
        self.sigma_e = interp1d(
            c / lam_m_e[::-1], sig_e[::-1], bounds_error=False, fill_value=0.0
        )


# %% ---- fiber dispersion
class PM980_XP_betas:
    def __init__(self, v0):
        frame = pd.read_excel(
            path / "Allured_PM980_XP_digitized_phase/Allured_PM980_XP_phase.xlsm"
        )
        data = frame.to_numpy().astype(float)

        # Expect columns: [wavelength_nm, phase_rad_per_m]
        wl = data[:, 0] * 1e-9          # m
        beta = data[:, 1]               # rad/m  (this is β(ω))

        omega = 2 * np.pi * c / wl
        omega0 = 2 * np.pi * v0

        # Ensure omega is monotonic for differentiation
        idx = np.argsort(omega)
        omega = omega[idx]
        beta = beta[idx]
        wl = wl[idx]

        # ---- local window around center wavelength lambda0 = c / v0
        # Example: +/- 80 nm window (tune this)
        m = (wl > (c / v0 - 100e-9)) & (wl < (c / v0 + 100e-9))

        # ---- fit beta(omega) about omega0, then differentiate analytically
        # Need deg = 5 to get beta2..beta5 at omega0
        polyfit = np.polyfit(omega[m] - omega0, beta[m], deg=5)  # highest order first
        self.polyfit = polyfit[::-1][2:6] * np.array([2, 6, 24, 120], dtype=float)  # [beta2..beta5]