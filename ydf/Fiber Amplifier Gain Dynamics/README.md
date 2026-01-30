
# Fiber Amplifier Gain Dynamics 
Jonathan Musgrave


Optical amplifiers based on lengths of rare-earth doped optical fibers are a ubiquitous tool to amplify optical signals across a wide range of wavelength bands. Although predominantly leveraged for low noise amplification in the C and L bands for optical telecommunications that have showed great promise for compact and high energy scaling for ultrafast optical pulses ranging from femtoseconds up to nanoseconds. When considering such high peak power and broad bandwidth systems it becomes increasingly important to consider the dispersive nature of the gain media itself. In this work we present a broadly applicable model for simulating ultrafast optical amplification. Our work considers not only the standard Kerr and Raman nonlinearities but also the shape of the gain curve by coupling the Generalized Nonlinear Schrodinger Equation to a set of rate-equations which govern the gain dynamics during propagation. This work is especially targeted for studying the quickly developing and designing gain-managed nonlinear fiber amplifier for pulses around 1um range. 


# Fiber_Prop_Lz_v1.m
Calculates the propagation of a given fiber defined by the parameteres in the struc fiber. signal and pumps define the optical fields, sys defines the system simulation parameters such as the axes and plotting. step_size is the non adaptable step_size of the propagation running

sys controls all parameters regarding the numerical. Such as the temporal step size, window size, center wavelength etc... Importantly sys contains the system frep in sys.frep

signal struct controls the input and parts of the signal that define its propgation such as the input pulse shape (gaussian be default), initial energy, etc..

pump struct controls the pumps and can be defined as either forward or backward directions, the power as well as the wavelength,

fiber struct controls all the relevant fiber parameters, the doping level, the spectroscopic data, the nonlinearity, mode size etc... Probably the most simulation data is stored here

# PropagateCNLSE_dz.m
Propagates the signal in one step of dz using the coupled generalized Nonlinear schrodinger equation using the split step fourier method. Note that this is called in the Fiber_Prop_Lz_v1.m file. 

# Rate_Equation_NoASEdz.m
Propagates the signal and pump one step of dz using the rate equation for a two level system relevant for most rare-earth dopants.

# units.m
Contains all units for ease of use

# Example 1
Simple GMN amplification following Frank Wise's original paper in 2019 in a 10um DCF YDFA

# Example 2
Counter propagating pump example for a signal being amplified. for a single clad fiber

# Utils folder
has all the dependency for the code to running

# Plotting Function folder
Contains plotting functions