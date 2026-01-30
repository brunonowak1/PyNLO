# YDF - Ytterbium-Doped Fiber Amplifier Package

A comprehensive Python package for modeling ytterbium-doped fiber amplifiers (YDFAs) integrated with PyNLO's nonlinear optics simulation framework. This package implements a 2-level rate equation model for ytterbium ions coupled with the nonlinear Schrödinger equation for realistic YDFA simulation.

## Overview

The YDF package provides:
- **2-level erbium ion model** 
- **Bidirectional amplification** with forward and backward pumping
- **Coupled rate equations** and nonlinear pulse propagation
- **Adaptive step-size control** for numerical stability
- **Complete integration** with PyNLO's pulse and fiber simulation framework

## Package Structure

```
edf/
├── README.md                      # This file
├── __init__.py                    # Package initialization
├── ydfa.py                        # High-level amplification functions
├── two_level_ss_eqns.py           # 2-level rate equation mathematics
├── re_nlse_joint_2level.py        # Main YDF class and models
└── utility/
    ├── __init__.py                                 # Utility functions and data
    └── NLight_provided_cross_sections/             # Ytterbium cross-section data from nLIGHT
        ├── nLIGHT YDF absorption emission cross sections.xlsx
    └── Topper_provided_cross_sections/             # Temperature-dependent ytterbium cross-section data from Topper et al. (this data used in the simulations rather than nLIGHT's)
        ├── notes.txt                               # Reference data details
        ├── Topper_et_al._YDF_absorption_cross_sections.xlsx
        ├── Topper_et_al._YDF_emission_cross_sections.xlsx
    └── Allured_PM980_XP_digitized_phase/           # Phase data for obtaining GVD and its derivatives
        ├── notes.txt                               # Reference data details
        ├── Allured_PM980_XP_digitized_phase.xlsm 
└── Fiber Amplifier Gain Dynamics/ # MATLAB code from Shu-Wei Huang's lab for validation. Propagation is based on methods of Lindelberg et al. (see below for reference details)  
    ├── Example_1_GainManagedAmplification.m        # MATLAB example with same parameters as PyNLO/examples/ydfa_examples/simple_ydfa.py for cross comparison.  
    ├──...supporting files for Example_1_GainManagedAmplification.m
```

## Mathematical Model based on Lindberg et al. Sci. Rep. 6, 34742 (2016).

### 2-Level Erbium Ion System

The package implements a two-level energy structure:

```
Level 2 (^2F_(5/2))  ←─── Excited state
    │ τ = 0.86 = ms
Level 1 (^2F_(7/2))  ←─── Ground state
```

### Key Parameters

- **Lifetimes:**
  - `τ = 0.86 ms`: Excited state lifetime

### Steady-State Population Solutions

The population density n₂ is solved analytically from the steady-state rate equations. The solutions are derived by Linderberg et al. and implemented in `two_level_ss_eqns.py`.

## Core Classes

### YDF Class

The main class inheriting from `pynlo.materials.SilicaFiber`:

```python
from ydf.re_nlse_joint_5level import YDF
from helpers import geom_factors    # helpers is described in the Helpers Folder section below

ydf = YDF(
    f_r=100e6,                 # Repetition frequency (Hz)
    overlap_p=overlap_p,       # Pump mode overlap from geom_factors, typically at 976/980 nm
    overlap_s=overlap_s,       # Signal mode overlap from geom_factors as a function of frequency
    n_ion=9.6e25,              # Ion concentration (ions/m³)
    A_doped=A_doped,           # Doped area (m²)
    sigma_p_a=sigma_p_a,       # Pump absorption cross section, typically at 976/980 nm
    sigma_p_e=sigma_p_e,       # Pump emission cross section, typically at 976/980 nm
    sigma_a=sigma_a,           # Signal absorption cross-sections as a function of frequency
    sigma_e=sigma_e            # Signal emission cross-sections as a function of frequency
    nu_p=nu_p                  # Pump center frequency, typically at 976/980 nm
)
```

### Mode Class

Custom mode class with YDF-specific functionality:

- **Dynamic gain calculation** based on population inversions
- **Pump power evolution** via RK45 integration
- **Population density tracking** n_2
- **Bidirectional propagation support**

### Model Classes

- **Model_YDF**: Extends `pynlo.model.Model` with adaptive stepping
- **NLSE**: Standard NLSE propagation with YDF gain

## Usage Examples

### Basic EDFA Simulation

```python
from scipy.constansts import c
import numpy as np
import pynlo
from helpers import geom_factors
from ydf.re_nlse_joint_2level import YDF
from ydf import ydfa
from ydf.utility import crossSection, PM980_XP_betas

# Load cross-sections and dispersion data
spl_sigma_a = crossSection(temperature=300).sigma_a
spl_sigma_e = crossSection(temperature=300).sigma_e
polyfit_n = PM980_XP_betas(v0=c/1030e-9).polyfit

# Create input pulse
pulse = pynlo.light.Pulse.Sech(
    n=256,
    v_min=c/1200e-9,
    v_max=c/860e-9, 
    v0=c/1030e-9,
    e_p=35e-3/2/f_r,
    t_fwhm=100e-15,
    min_time_window=10e-12
)

# Configure EDF
sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p_a = spl_sigma_a(c/980e-9)
sigma_p_e = spl_sigma_e(c/980e-9)

gamma, overlap_s, overlap_p = geom_factors(
    pulse.v0, pulse.v_grid,
    a_core_m=6e-6/2,
    NA=0.12,
    pump_lambda_m = 980e-9,
    pump_is_cladding=True,
    a_clad_m=125e-6/2,
)

ydf = YDF(
    f_r=200e6,
    overlap_p=overlap_p,
    overlap_s=overlap_s,
    n_ion=9.6e25,
    A_doped=np.pi*(6e-6/2)**2
    sigma_p_a=sigma_p_a
    sigma_p_e=sigma_p_e,
    sigma_a=sigma_a,
    sigma_e=sigma_e
    nu_p=c/980e-9
)
ydf.set_beta_from_beta_n(pulse.v0, polyfit_n)
ydf.gamma=gamma

# Generate model and simulate
model = ydf.generate_model(pulse, Pp_fwd=2.0)
sim = model.simulate(length=1.5, n_records=100)

# Access results
pump_power = sim.Pp
signal_power = sim.p_v
populations = [sim.n1_n, sim.n2_n, sim.n3_n, sim.n4_n, sim.n5_n]
```

### Bidirectional Amplification

For complete bidirectional simulation with iterative convergence:

```python
from ydf import ydfa

model_fwd, sim_fwd, model_bck, sim_bck = ydfa.amplify(
    p_fwd=input_pulse,
    p_bck=None,              # Or provide backward seed
    ydf=ydf,
    length=1.5,
    Pp_fwd=2.0,             # Forward pump power (W)
    Pp_bck=2.0,             # Backward pump power (W)
    n_records=100,
    tolerance=1e-3          # Convergence tolerance
)
```

### Rate Equations Only

For validation or when nonlinear effects are negligible:

```python
from ydf.two_level_ss_eqns import (
    dPp_dz, dPs_dz, gain, n2_func
)

# Calculate population densities
n2 = n2_func(n_ion, A_doped, overlap_p, overlap_s, 
             nu_p, P_p, nu_s, P_s, sigma_p_a, sigma_p_e, sigma_a, sigma_e, tau)

# Calculate gain coefficient
g = gain(n_ion, A_doped, overlap_p, overlap_s,
         nu_p, P_p, nu_s, P_s, sigma_p_a, sigma_p_e, sigma_a, sigma_e, tau)
```

## Advanced Features

### Shock Wave Effects

```python
# Automatic shock time calculation
model = ydf.generate_model(pulse, t_shock="auto", raman_on=True)

# Manual shock time specification  
model = ydf.generate_model(pulse, t_shock=1e-12, raman_on=True)
```

### Custom Cross-Sections

```python
# Use custom absorption/emission spectra
ydf = YDF(
    # ... other parameters ...
    sigma_a=custom_absorption_spectrum,
    sigma_e=custom_emission_spectrum
)
```

### Cladding-Pumped Fiber

```python
# Enable cladding pumping (sets overlap_p so that A_eff=A_cladding for pump in transition rates)
pump_is_cladding = True
pump_a_clad_m = 125e-6/2

__ , __, overlap_p = geom_factors(
    # .... other parameters ...
    pump_is_cladding=pump_is_cladding,
    a_clad_m=pump_a_clad_m,
)
```

## Data Sources

The package includes experimentally measured data from

### nLIGHT:

- **Cross sections**: Absorption and emission spectra for Yb ions

### Topper et al.:

- **Cross sections**: Absorption and emission spectra for Yb ions as a function of temperature

### Allured et al.:

- **Digitized phase**: Frequency-dependent phase for PM980-XP for GVD, TOD, FOD... calculations

## Integration with PyNLO

The YDF package seamlessly integrates with PyNLO's ecosystem:

- **Pulse objects**: Compatible with all PyNLO pulse types
- **Fiber materials**: Inherits from `SilicaFiber` class
- **Propagation models**: Uses PyNLO's adaptive step-size algorithms
- **Visualization**: Compatible with PyNLO's plotting functions

## Dependencies

- **numpy**: Array operations and mathematical functions
- **scipy**: Scientific computing (constants, interpolation, integration)
- **pynlo**: Core nonlinear optics framework
- **pandas**: Data file reading
- **matplotlib**: Plotting (for examples)

### Helpers Folder

The helpers folder contains two functions used for obtaining frequency-dependent overlap factors (and therefore effective areas for the transition rates in the rate equations) given a fiber's core radius and NA: marcuse_w_from_v and geom_factors. 

marcuse_w_from_v returns the mode field radius using the Marcuse formula given the fiber core size and NA.

geom_factors returns an overlap factor array for the signal (if signal is polychromatic) and scalar overlap factor for the pump using the mode-field radii from the marcuse_w_from_v function as input. Importantly, the nonlinear coefficient gamma at the center frequency, which is an input to the NLSE solver, is also obtained using geom_factors since gamma depends on the effective area (and therefore overlap factor) at the center frequency.

The formulas for and reasoning behind marcuse_w_from_v and geom_factor are described in Chapter 2 of Nonlinear Fiber Optics by Govind Agrawal.

## References

1. Lindberg et al. Sci. Rep. 6, 34742 (2016)
2. Topper et al. Opt. Mater. Express 14, 2956-2971 (2024)
3. Allured and Ashcom. Appl. Opt. 60, 6371-6384 (2021)
2. nLIGHT Corporation fiber specifications and cross-section data
4. Agrawal, G. P. (2019). Nonlinear Fiber Optics (6th ed.). Academic Press.
3. PyNLO documentation: [PyNLO repository](https://github.com/peter-chang62/PyNLO)

## Examples and Validation

See the `examples/ydfa_examples/simple_edfa.py` for basic amplification simulation as well as `ydf/Fiber Amplifier Gain Dynamics/Example_1_GainManagedAmplification` for MATLAB-based basic amplification simulation with the same rate equations, fiber specifications, and pulse parameters as the `simple_edfa.py example`. 
-

## License

This package is distributed under the same license as PyNLO. See the main repository for license details.