"""
Mass Calculation Module
=======================
Pure functions that compute component masses from design variables.
No side effects, no print statements at module level.
"""

import numpy as np
from InputParameters import (
    n_pax, n_pax_m, M_pax, M_bag, F_op,
    n_seat_abreast, n_galley, seat_pitch, n_toilet, n_crossaisles, n_aisles,
    container_h, aisle_h, P_cabin, C1, C2, C3, C4, Nbar, g, M_N_c, h_c
)
# from PerformanceAnalysisCombined import calculate_fuel_required
# Note: Vdive is passed as a parameter from PerformanceAnalysisCombined.evaluate_all()


# === Fuselage Geometry (fixed — depends only on cabin layout) ===============

def fuselage_geometry():
    """Return (length, width, height) of the fuselage in metres."""
    cabin_length = ((n_pax_m / n_seat_abreast) + n_galley) * seat_pitch + n_toilet + 0.8 * n_crossaisles
    fus_length = cabin_length / 0.7
    cabin_width = 0.5 * n_seat_abreast + 0.55 * n_aisles
    fus_width = cabin_width + 0.3
    fus_height = (container_h + 0.4) + (aisle_h + 0.35)
    return fus_length, fus_width, fus_height

# Pre-compute once (these depend only on cabin layout constants)
_L, _B, _H = fuselage_geometry()


# === Fixed Masses (independent of design variables) =========================

def M_fuselage():
    """Fuselage mass (kg) — Eqn for pressurised transport."""
    L, B, H = _L, _B, _H
    term1 = 9.75 + 5.84 * B
    term2 = (2 * L) / (B + H) - 1.5
    return C2 * P_cabin * term1 * term2 * (B + H) ** 2

def M_payload():
    """Payload mass (kg)."""
    return n_pax * (M_pax + M_bag)

def M_operational():
    """Operational items mass (kg)."""
    n_crew = (n_pax / 50) + 2
    return 85 * n_crew + F_op * n_pax

def M_fixed():
    """Total fixed mass = fuselage + payload + operational (kg).
    This does NOT depend on any design variable."""
    return M_fuselage() + M_payload() + M_operational()

# Cache the fixed mass (constant for this aircraft configuration)
Mfix = M_fixed()

# === Variable Masses (depend on design variables) ===========================

def M_lifting_surface(AR, Sw, QW_4, t_c, taper_ratio, Mto, Vdive):
    """Lifting surface mass (kg) — Eqn 6.23.
    
    Parameters:
        Vdive: Design dive speed (m/s) — calculated from performance analysis
               (M_N_dive * a_c where M_N_dive = M_N_c + 0.05)
    
    Uses numpy only (no mpmath)."""
    cos_QW4 = np.cos(np.radians(QW_4))
    sec_QW4 = 1.0 / cos_QW4
    taper_term = (1 + 2 * taper_ratio) / (3 + 3 * taper_ratio)
    inner = (AR ** 0.5
             * Sw ** 1.5
             * sec_QW4
             * taper_term
             * (Mto / Sw)
             * Nbar ** 0.3
             * (Vdive / t_c) ** 0.5)
    return C1 * inner ** 0.9

def M_powerplant(T0):
    """Powerplant mass (kg) — engine + installation.
    Corrected: Assume 6.5 is the Engine Thrust-to-Weight Ratio.
    Mass = Thrust / (T/W * g).
    """

    M_eng = T0 / (6.5 * g)
    return M_eng * C3

def M_systems(Mto):
    """Systems mass (kg)."""
    return Mto * C4

def M_fuel(FMf, Mto):
    """Fuel mass (kg) from fuel-mass fraction."""
    return FMf * Mto

 

def M_total(AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, Mto, Vdive):
    """Total takeoff mass (kg) = fixed + variable masses.
    All variable masses are recomputed from the design variables."""
    Mls = M_lifting_surface(AR, Sw, QW_4, t_c, taper_ratio, Mto, Vdive)
    Mpp = M_powerplant(T0)
    Msys = M_systems(Mto)
    Mfuel = M_fuel(FMf, Mto)

    return Mfix + Mls + Mpp + Msys + Mfuel


def print_mass_breakdown(AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, Mto, Vdive):
    """Print detailed mass breakdown."""
    Mls = M_lifting_surface(AR, Sw, QW_4, t_c, taper_ratio, Mto, Vdive)
    Mpp = M_powerplant(T0)
    Msys = M_systems(Mto)
    Mfuel = M_fuel(FMf, Mto)

    print("-" * 60)
    print("  MASS BREAKDOWN DETAILED")
    print("-" * 60)
    print(f"  Fuselage:     {M_fuselage():10.1f} kg")
    print(f"  Payload:      {M_payload():10.1f} kg")
    print(f"  Operational:  {M_operational():10.1f} kg")
    print(f"  -----------------------")
    print(f"  Total Fixed:  {Mfix:10.1f} kg")
    print(f"  -----------------------")
    print(f"  Lifting Surface:   {Mls:10.1f} kg")
    print(f"  Powerplant:   {Mpp:10.1f} kg")
    print(f"  Systems:      {Msys:10.1f} kg")
    print(f"  Fuel:         {Mfuel:10.1f} kg")
    print("-" * 60)
    calc_total = Mfix + Mls + Mpp + Msys + Mfuel
    print(f"  CALCULATED TOTAL: {calc_total:10.1f} kg")
    print("-" * 60)


