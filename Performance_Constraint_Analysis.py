"""
Performance Analysis Module
===========================
Evaluate aircraft performance and constraints based on design variables.
NO GLOBAL STATE for calculations. All logic encapsulated in `evaluate_all`.
"""

import math
import numpy as np
import MassCalculation as MC
from InputParameters import (
    Range, TOFL, LDL, Va_limit, b_limit, SP, n_load, g, rho_sl, R_bypass, Mg_S_0, T_Mg_0,
    pressure, density, sigma, speed_of_sound,
    Delt_LET, Delt_TET, Delt_LEL, Delt_TEL, F_F, Rw, A_f, T_f, cl_zero,
    k_e, f_lambda, N_e_wing, c_prime, Ftau,
)

# --- Helper Functions (Pure) -----------------------------------------------

def f_tau(Ftau, R_bypass, s, M_N, sigma_val):
    """
    Calculate the thrust factor (tau) using equation (3.7a).
    """
    if 0 <= M_N <= 0.4:
        K_1t = 1.0
        K_2t = 0
        K_3t = -0.6
        K_4t = -0.04
    elif 0.4 < M_N <= 0.9:
        K_1t = 0.88
        K_2t = -0.016
        K_3t = -0.3
        K_4t = 0.0
    else:
        K_1t = 0.88; K_2t = -0.016; K_3t = -0.3; K_4t = 0.0
    
    tau = Ftau * (K_1t + K_2t * R_bypass + (K_3t + K_4t * R_bypass) * M_N) * sigma_val ** s
    return tau

def f_tau_bar(R_w, t_c): 
    """Correction factor for wing thickness, using equation (6.13b)."""
    return ((R_w - 2) / R_w) + (1.9 / R_w) * (1 + 0.526 * (t_c / 0.25) ** 3)

def f_C_DZ(cl_zero, M_N, A_f, t_c, cos_QW_4, R_w_, T_f, Sw, tau_bar):
    """
    Calculate zero-lift drag coefficient (C_DZ) using equation (6.13a).
    """
    term1 = 1 - (2 * cl_zero) / R_w_
    inner_term = (M_N * (cos_QW_4)**0.5) / (A_f - t_c)
    #if inner_term < 0: inner_term = 0
    term2 = 1 - 0.2 * M_N + 0.12 * (inner_term ** 20)
    
    return 0.005 * term1 * tau_bar * term2 * (R_w_ * T_f * Sw ** -0.1)

def f_delta_C_DT(F_F, AR):
    """
    Calculate increment in drag due to high lift devices (delta_C_DT) using equation (6.15b).
    """
    return (0.03 * F_F - 0.004) / (AR ** 0.33)

def f_k_v(f_lambda, N_e_wing, AR, cos_QW_4, t_c, M_N):
    """
    Calculate (k_v)_0 using the bracketed value from equation (6.14a).
    """
    term1 = ((0.142 + f_lambda * AR) / (cos_QW_4 ** 2)) * (10 * t_c) ** 0.33
    term2 = 0.1 * (3 * N_e_wing + 1) / ((4 + AR) ** 0.8)
    term3 = (1 + (0.12 * M_N**6))
    return (1 / (AR * np.pi)) * term3 * (1 + term1 + term2)

def calculate_fuel_required(Mto, V_co, h_c, Climb_gradient_2, ToL, LFL, Range, V_TAS, tsfc_val, L_c, D_c):
    """
    Calculate total fuel mass required for the mission using Breguet Range calculation.
    """
    # 1. Takeoff (1% of Mto)
    M_fuel_to = 0.01 * Mto

    # 2. Climb (3% of Mto - implied by 0.97 start cruise mass)
    M_fuel_co = (1 - 0.97) * Mto

    # 3. Cruise
    delta_h_climb = h_c - 10.7
    S_climb = ((V_co * delta_h_climb) / (Climb_gradient_2 * V_co)) / 1000
    
    S_c = (Range - (ToL/1000) - (LFL/1000) - S_climb ) * 1000 #meter
    # S_c not used as the difference is small
    
    #S = 3600 * (V_TAS / c) * (L_c / D_c) m 
    exponent = Range * 1000 / (3600 * (V_TAS / tsfc_val) * (L_c / D_c))
    # Clamp exponent to prevent overflow
    exponent = np.clip(exponent, -300, 300)

    M1_M2 = np.exp(exponent)
    M2_M1 = 1.0 / M1_M2
    
    M_fuel_c = (1 - M2_M1) * 0.97 * Mto 
    
    # 4. Landing (1% of Mto)
    M_fuel_ld = 0.01 * Mto

    return M_fuel_to + M_fuel_co + M_fuel_c + M_fuel_ld


# --- Main Evaluation Function (The core of the fix) ------------------------

def evaluate_all(x):
    """
    Compute all aircraft characteristics, masses, performance, and constraints
    for a given vector of design variables.

    x = [AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, M_N_c, h_c, Mto]
    """
    # Unpack Design Variables
    AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, M_N_c, h_c, Mto = x
    
    # Derived geometry
    cos_QW_4 = np.cos(np.radians(QW_4))
    
    # Atmosphere at cruise
    rho_c = density(h_c)
    sigma_c = sigma(h_c)
    a_c = speed_of_sound(h_c)
    
    # Flight speeds
    V_TAS = 340 * M_N_c * sigma_c ** 0.117
    q_c = 0.5 * rho_c * V_TAS**2  # Dynamic pressure at cruise

    # Masses at conditions
    Mcruise = 0.97 * Mto
    
    # -----------------------------------------------------------------------
    # 1. Aerodynamics & Drag
    # -----------------------------------------------------------------------
    
    tau_bar = f_tau_bar(Rw, t_c)
    
    # Lift coefficients
    C_LUS = 0.8 * (1.5 + Delt_LET + Delt_TET) * cos_QW_4 
    C_Lmax_to = (1.5 + Delt_LET + Delt_TET) * cos_QW_4 
    C_Lmax_L = (1.5 + Delt_LEL + Delt_TEL) * cos_QW_4 

    # --- Takeoff Performance ---
    h_to = 0
    sigma_to = sigma(h_to) 
    rho_to = density(h_to)
    
    #V_to = 1.2 * np.sqrt((2 * Mto * g) / (rho_to * Sw * C_Lmax_to))   #differemt from V_co by density
    #M_N_to = V_to / 340
    #tau_to = f_tau(Ftau, R_bypass, 0.7, M_N_to, sigma_to)
    #T_to = T0 * tau_to
    #T_W = T_to / (Mto * g) # Thrust to Weight

    Mg_S = (Mto * g) / Sw  # Wing loading (N/m^2)
    T_W = T0 / (Mto * g) # Thrust to Weight
    
    term1 = (k_e / C_LUS) * T_W ** -1.35 * Mg_S
    term2 = 6 * ((Mto * g) / (Sw * C_LUS))**0.5
    term3 = 120 * (1 - T_W )
    ToL = term1 + term2 + term3

    # --- Climb Gradient (2nd Segment, OEI) ---
    Vus = np.sqrt((2 * Mto * g) / (rho_to * Sw * C_LUS))
    M_N_co = 0.00324 * Vus
    Vs_to = np.sqrt((2 * Mto * g) / (density(10.7) * Sw * C_Lmax_to))       #Stall speed at takeoff
    V_co = 1.2 * Vs_to 
    
    C_DZ_co = f_C_DZ(cl_zero, M_N_co, A_f, t_c, cos_QW_4, Rw, T_f, Sw, tau_bar)
    delta_C_DT = f_delta_C_DT(F_F, AR)
    CD_co = C_DZ_co + delta_C_DT # effective zero-lift drag coefficient during climb (with flaps) 
    
    kv_co = f_k_v(f_lambda, N_e_wing, AR, cos_QW_4, t_c, M_N_co)
    CL_climb = 0.84 * C_LUS
    CD_climb = CD_co + kv_co * CL_climb**2
    
    # Thrust at climb out
    # Assuming R is Rw (Bypass Ratio) = 5.5 and s = 0.7 
    tau_co = f_tau(Ftau, R_bypass, 0.7, M_N_co, sigma_to)
    T_co = T0 * tau_co
    
    D_co = 0.5 * density(10.7) * V_co**2 * Sw * CD_climb
    gamma2 = (T_co - D_co) / (Mto * g)

    # --- Cruise Performance ---
    # Thrust at cruise
    tau_c = f_tau(Ftau, R_bypass, 0.7, M_N_c, sigma_c)
    T_c = T0 * tau_c
    
    # Drag at cruise
    C_DZ_c = f_C_DZ(cl_zero, M_N_c, A_f, t_c, cos_QW_4, Rw, T_f, Sw, tau_bar)
    kv_c = f_k_v(f_lambda, N_e_wing, AR, cos_QW_4, t_c, M_N_c)
    
    CL_c = (Mcruise * g) / (q_c * Sw)
    L_c = CL_c * q_c * Sw   
    CD_c = C_DZ_c + kv_c * CL_c**2
    D_c = q_c * Sw * CD_c
    
    # Rate of Climb at Cruise (Initial)
    RoC = (T_c - D_c) * V_TAS / (Mcruise * g)
    
    # Buffet Limit
    CL_buffet = 0.65 * cos_QW_4
    
    # --- Landing Performance ---
    Vs_L = np.sqrt((2 * Mto * g) / (rho_to * Sw * C_Lmax_L))
    Va = 1.3 * Vs_L
    LFL = 488 + 4.5 * Va + 0.196 * Va**2    #Factor 1.67 included 
    
    # ------ Wing span limited by structure---------
    # b_al calculation
    b_struct = (Sw * (((SP * cos_QW_4)**1.6) / (n_load**0.8)) * (t_c**0.4)) **0.5


    # ---  Fuel Required  ---

    #Print ground distance covered at cruise
    delta_h_climb = h_c - 10.7
    S_climb = ((V_co * delta_h_climb) / (gamma2 * V_co)) / 1000
    S_c = (Range - (ToL/1000) - (LFL/1000) - S_climb ) 
    # S_c not used as the difference from range is small


    tsfc_val = c_prime * (1 - 0.15 * (R_bypass ** 0.65)) * (1 + 0.28 * (1 + 0.063 * (R_bypass ** 2)) * M_N_c) * sigma_c**0.08
    #M1_M2 = np.exp(Range *1000 / (3600 * (V_TAS / tsfc_val) * (L_c / D_c)))
    M_fuel_req = calculate_fuel_required(Mto, V_co, h_c, gamma2, ToL, LFL, Range, V_TAS, tsfc_val, L_c, D_c)

    
    # -------Volume fuel mass (Available)------
    vol_term = (1 - 0.89 * taper_ratio + 0.49 * taper_ratio**2) / AR
    M_fuel_vol = 420 * b_struct * Sw * t_c * vol_term

    # Gust Sensitivity
    # Design Dive Speed Vdive using Mdive
    M_N_dive =  M_N_c + 0.05
    Vdive = M_N_dive * a_c
    
    gust_denom = (0.32 + (0.16 * AR / cos_QW_4)) * (1 - (M_N_c * cos_QW_4)**2)**0.5
    # Protect against divide by zero in gust denominator (optimization error)
    gust_load = (2.7 * AR * Vdive) / gust_denom

    #Drag divergence Mach number (M_N_DD)
    A_F = 0.95     # airfoil factor (supercritical)

    M_DD_airfoil = A_F - 0.1 * CL_c - t_c
    M_DD_wing = M_DD_airfoil / (cos_QW_4**0.5)

    M_N_crit = M_DD_wing - 0.11

    # --- Mass Iteration Closure ---
    # Call Mass Calculation
    
    # Compute Total Mass
    Mtotal_calc = MC.M_total(AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, Mto, Vdive)

    # Return Dictionary for Optimization
    return {
        "Mtotal": Mtotal_calc,
        "M_fuel_FFM": FMf * Mto,
        "M_fuel_req": M_fuel_req,
        "M_fuel_vol": M_fuel_vol,
        "ToL": ToL,
        "gamma2": gamma2,
        "RoC": RoC,
        "CL_buffet": CL_buffet,
        "CL_c": CL_c,
        "CD_c": CD_c,
        "T_c": T_c,
        "D_c": D_c,
        "L_c": L_c,
        "LFL": LFL,
        "Va": Va,
        "gust": gust_load,
        "b_struct": b_struct,
        "M_N_crit": M_N_crit,
        "M_DD_wing": M_DD_wing,
        "Vdive": Vdive,
        "V_TAS": V_TAS,
        "tsfc": tsfc_val,
        "S_c": S_c,
        "Mg_S": Mg_S,
        "T_W": T_W,
        #"M1_M2": M1_M2,
    }


def print_constraints(r):
    """Debug print for the optimization dictionary."""
    print("-" * 60)
    print(f"  Mtotal (Calc):   {r['Mtotal']:.1f} kg")
    print(f"  Fuel (Frac):     {r['M_fuel_FFM']:.1f} kg")
    print(f"  Fuel (Req):      {r['M_fuel_req']:.1f} kg")
    print(f"  Available Vol:   {r['M_fuel_vol']:.1f} kg")
    print(f"  TOFL:            {r['ToL']:.1f} m")
    print(f"  LFL:             {r['LFL']:.1f} m")
    print(f"  Climb Gradient:  {r['gamma2']:.4f}")
    print(f"  Rate of Climb:   {r['RoC']:.2f} m/s")
    print(f"  Approach Speed:  {r['Va']:.2f} m/s")
    print(f"  Cruise CL:       {r['CL_c']:.4f} (Buffet: {r['CL_buffet']:.4f})")
    print(f"  CD at Cruise:      {r['CD_c']:.6f}")
    print(f"  Lift at Cruise:  {r['L_c']:.1f} N")
    print(f"  Drag at Cruise:  {r['D_c']:.1f} N")
    print(f"  Critical Mach Number: {r['M_N_crit']:.4f}")
    print(f"  Drag Divergence Mach Number: {r['M_DD_wing']:.4f}")
    print(f"  Dive Speed:      {r['Vdive']:.2f} m/s")
    print(f"  V_TAS (Cruise):  {r['V_TAS']:.2f} m/s")
    print(f"  TSFC (Cruise):   {r['tsfc']:.4f} N/N/h")
    print(f" Ground distance covered at cruise:   {r['S_c']:.1f} m")
    print(f"  Wing Loading:    {r['Mg_S']:.1f} N/m²")
    print(f"  T/W Ratio:       {r['T_W']:.4f}")
    #print(f" M1_M2 (Breguet):     {r['M1_M2']:.4f}")
    print("-" * 60)

#Constrainst_check = print_constraints(evaluate_all(x0))  # Check constraints at initial guess
#print({"Initial constraint evaluation at x0": Constrainst_check})


