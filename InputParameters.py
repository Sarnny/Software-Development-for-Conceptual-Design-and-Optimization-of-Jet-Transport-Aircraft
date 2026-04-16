"""
Aircraft Preliminary Design - Constants & Requirements
=======================================================
All values are SI units unless noted otherwise.
This module contains ONLY constants and pure helper functions.
No side effects, no print statements, no computations at import time.
"""

import numpy as np

# --- Design Requirements ---------------------------------------------------
n_pax = 150               # design passengers
n_pax_m = 180              # max passengers (high-density layout)
Range = 5500               # km  design range
TOFL = 2400                # m   max takeoff field length
LDL = 1800                 # m   max landing distance
Va_limit = 73              # m/s max approach speed
b_limit = 80               # m   max wing span
M_N_c = 0.82               # FIXED design cruise Mach number 
h_c = 10000                 # FIXED cruise altitude
# --- Passenger / Payload ---------------------------------------------------
M_pax = 80                 # kg  per passenger
M_bag = 30                 # kg  per bag

# --- Cabin Layout ----------------------------------------------------------
seat_pitch = 0.85          # m
n_seat_abreast = 6
n_toilet = 2
n_galley = 2
n_seatrows = 30
n_aisles = 1
n_crossaisles = 2
container_h = 1.63         # m  cargo container height
aisle_h = 2.0              # m  Aisles height

# --- Powerplant ------------------------------------------------------------
R_bypass = 5.5             # bypass ratio
N_eng = 2                  # number of engines

# --- Cabin Pressure --------------------------------------------------------
P_cabin = 0.58             # bar  max working differential pressure

# --- Mass-estimation coefficients (Howe) -----------------------------------
def calc_C1(Range, n_pax):
    """Lifting-surface mass coefficient - Eqn 6.23a."""
    return 0.00072 - 0.0005 * (270 + 0.05 * Range) * n_pax * 1e-6

C1 = calc_C1(Range, n_pax)
C2 = 0.79                 # fuselage (Table 6.6)
C3 = 1.56                 # powerplant (Table 6.8)
C4 = 0.14                 # systems (Table 6.9)
F_op = 12                 # operational items per pax (kg) - Eqn 6.21a

# --- Performance Parameters ------------------------------------------------
Mg_S_0 = 6500              # N/m^2  typical wing loading (Table 5.4)
T_Mg_0 = 0.34              # typical thrust loading (Table 3.3A)

Delt_LET = 0.4             # LE flap lift increment - takeoff (para 6.2.4.4)
Delt_TET = 0.7             # TE Fowler-flap lift increment - takeoff (Table 6.1)
Delt_LEL = 0.65            # LE flap lift increment - landing
Delt_TEL = 1.35            # TE flap lift increment - landing
F_F = 1.2                  # flap drag factor (Fowler) - Eqn 6.15b
Rw = 5.5                   # wetted-area ratio (Table 6.3)
A_f = 0.93                 # airfoil factor (supercritical)
T_f = 1.2                  # type factor - jet airliners (Table 6.4)
cl_zero = 0.4              # zero-lift cl reference (fraction of the wing over which the flow is laminar)
k_e = 0.1                  # takeoff empirical constant
f_lambda = 0.0062          # taper-ratio function, typical value 
N_e_wing = 0               # engines mounted over the wing
c_prime = 0.7             # SFC reference factor (N/N/h)
Ftau = 1                   # thrust-factor multiplier
SP = 16                    # structural parameter for wing span

n_load = 2.5               # limit load factor
g = 9.81                   # m/s^2
Nbar = n_load * 1.65       # equivalent load for mass calculation
# --- ISA Atmosphere Functions -----------------------------------------------
rho_sl = 1.225             # kg/m^3  sea-level density

def pressure(h):
    """Static pressure (kPa) at altitude h (m) - troposphere ISA."""
    h = np.clip(h, 0, 44000)  # clamp to valid range
    return 101.29 * (1 - 2.25e-5 * h) ** 5.256

def density(h):
    """Air density (kg/m^3) at altitude h (m) - troposphere ISA."""
    h = np.clip(h, 0, 44000)  # clamp to valid range
    return rho_sl * (1 - 2.256e-5 * h) ** 4.256

def sigma(h):
    """Density ratio = rho(h)/rho_sl - troposphere ISA."""
    h = np.clip(h, 0, 44000)  # clamp to valid range
    return (1 - 2.256e-5 * h) ** 4.256

def speed_of_sound(h):
    """Speed of sound (m/s) at altitude h (m) - troposphere ISA.
    Using T = 288.15 - 0.0065*h  and  a = sqrt(gamma*R*T)."""
    T = 288.15 - 0.0065 * h
    return np.sqrt(1.4 * 287.05 * max(T, 1.0))


# --- Initial Guess (design-variable vector) --------------------------------
#  [AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, M_N_c, h_c, Mto]
# Adjusted to be closer to user targets (T0=95k, Sw=125, M=0.78, Mto=65k)
x0 = np.array( [8.0, 120.0, 30.0, 0.2, 0.4, 200000.0, 0.2, 60000.0])

