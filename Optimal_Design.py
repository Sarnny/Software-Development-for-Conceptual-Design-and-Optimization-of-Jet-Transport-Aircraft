"""
Optimization Module
====================
Minimises takeoff mass (Mto) using SLSQP with 11 design constraints.
This is the SINGLE entry point — run this file to execute the optimisation.

Design variables:
    x = [AR, Sw, QW_4, t_c, taper_ratio, T0, FMf, M_N_c, h_c, Mto]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import MassCalculation as MC

from InputParameters import (
    x0, TOFL, LDL, Va_limit, b_limit, M_N_c, h_c
)
from Performance_Constraint_Analysis import evaluate_all, print_constraints

# Insert M_N_c and h_c into x0
# Original x0 was [AR, Sw, QW_4, t_c, taper, T0, FMf, Mto]
# New x0 needs to be [AR, Sw, QW_4, t_c, taper, T0, FMf, M_N_c, h_c, Mto]
x0 = np.concatenate((x0[:-1], [M_N_c, h_c], [x0[-1]]))


# ========================================================================
#  Bounds on design variables
# ========================================================================
#        AR       Sw     QW_4    t/c   taper    T0       FMf    M_N_c   h_c       Mto
bounds = [
    (5,   50),     # AR
    (50,  1000),   # Sw  (m^2)
    (0,  50),      # QW_4 (deg)s
    (0.1, 0.5),    # t/c
    (0.3, 1.0),    # taper_ratio
    (10000, 1000000), # T0 (N) - static sea level thrust
    (0.1, 0.5),    # FMf
    (0.3, 1.0),      # M_N_c
    (3000, 100000),# h_c (m)
    (10000, 1000000),  # Mto (kg)
]

# ------------------------------------------------------------------------
# Normalization Helpers
# ------------------------------------------------------------------------

def normalize(x_real):
    """Convert physical variables to normalized [0, 1] range."""
    x_norm = []
    for i, val in enumerate(x_real):
        lb, ub = bounds[i]
        # Protect against potential zero division (though bounds should be distinct)
        if abs(ub - lb) < 1e-9:
            norm_val = 0.5 
        else:
            norm_val = (val - lb) / (ub - lb)
        x_norm.append(norm_val)
    return np.array(x_norm)

def unnormalize(x_norm):
    """Convert normalized [0, 1] variables back to physical units."""
    x_real = []
    for i, val in enumerate(x_norm):
        lb, ub = bounds[i]
        real_val = lb + val * (ub - lb)
        x_real.append(real_val)
    return np.array(x_real)


# ========================================================================
#  Objective function
# ========================================================================

def objective(x_norm):
    """
    Objective function for the optimizer (working in normalized space).
    Target: Minimize Mtotal while respecting soft penalties for stability.
    """
    # 1. Unnormalize to get physical variables
    x = unnormalize(x_norm)
    
    # 2. Extract physical variables for readability
    AR    = x[0]
    Sw    = x[1]
    QW_4  = x[2]
    t_c   = x[3]
    T0    = x[5]
    FMf   = x[6]
    M_N_c = x[7]
    h_c   = x[8]
    Mto   = x[9]

    # 3. Evaluate Physics
    # We catch errors to prevent the optimizer from crashing if it tries jagged regions
    try:
        r = evaluate_all(x)
        Mtotal = r["Mtotal"]
    except Exception as e:
        return 1e9 # Return huge value if physics fails

    # 1. Base Objective: Mass
    # Scale Mtotal so it is ~1.0
    obj = r["Mtotal"] / 50000.0 
    
    # # 2. Penalties 
    # pen = 0.0
    # w = 10.0 # Moderate weight for preferences

    # # --- USER PREFERENCES (Soft Targets) ---
    
    # # Target Thrust ~ 95,000 N
    # # We use a quadratic penalty to encourage being close to 95k
    # pen += w * ((T0 - 95000.0) / 95000.0)**2

    # # Target Low Wing Area (Prefer Sw around 120-130 m^2)
    # # If Sw > 125, penalize. If Sw < 125, no penalty (it's "low" enough).
    # if Sw > 150.0:
    #     pen += w * ((Sw - 150.0) / 150.0)**2

    # # Target High Mach (Prefer M >= 0.78)
    # if M_N_c < 0.55:
    #     # Heavily penalize low speed
    #     pen += w * 5.0 * ((0.55 - M_N_c) / 0.55)**2

    # # Target High Altitude (Prefer h >= 10000 m)
    # if h_c < 10000.0:
    #     pen += w * ((10000.0 - h_c) / 10000.0)**2

    # # --- SAFETY GUIDANCE ---
    
    # # Don't let T/W get dangerously low (Safety)
    # g = 9.81
    # TW = T0 / (Mto * g)
    # if TW < 0.25:
    #     pen += w * ((0.25 - TW)/0.25)**2

    # # Don't let fuel fraction get unrealistically low (Safety)
    # if FMf < 0.15:
    #     pen += w * ((0.15 - FMf)/0.15)**2

    # # MASS CLOSURE HELP
    # # We add this to objective because C13 is strict equality and hard for SLSQP
    # # This helps "guide" the optimizer to the valid region
    # mass_err = (r["Mtotal"] - Mto) / Mto
    # pen += w * 5.0 * mass_err**2

    return obj #+ pen


# ========================================================================
#  Constraints  (Normalized Wrappers)
# ========================================================================
#  Optimization variable is now x_norm. We must unnormalize before calc.

def _eval(x_norm):
    x = unnormalize(x_norm)
    return evaluate_all(x)

def con_fuel_balance(x_norm):
    """C1: Fuel from fraction == fuel required (equality)."""
    r = _eval(x_norm)
    return (r["M_fuel_FFM"] - r["M_fuel_req"]) / 20000.0

def con_fuel_volume(x_norm):
    """C2: Available fuel volume >= fuel required."""
    r = _eval(x_norm)
    return (r["M_fuel_vol"] - r["M_fuel_req"]) / 20000.0 

def con_tofl(x_norm):
    """C3: Takeoff field length <= TOFL (inequality)."""
    r = _eval(x_norm)
    return (TOFL - r["ToL"]) / 100.0  

def con_climb_gradient(x_norm):
    """C4: Second-segment climb gradient >= 0.024 (inequality)."""
    r = _eval(x_norm)
    return (r["gamma2"] - 0.024) 

def con_roc(x_norm):
    """C5: Rate of climb >= 1.5 m/s (inequality)."""
    r = _eval(x_norm)
    return (r["RoC"] - 1.5) / 5.0

def con_buffet(x_norm):
    """C6: Cruise CL <= buffet limit (inequality)."""
    r = _eval(x_norm)
    return (r["CL_buffet"] - r["CL_c"]) / 1e-2

def con_thrust_drag(x_norm):
    """C7: Cruise thrust >= cruise drag."""
    r = _eval(x_norm)
    return (r["T_c"] - r["D_c"]) / 1000.0

def con_landing_dist(x_norm):
    """C8: Landing field length <= LDL."""
    r = _eval(x_norm)
    return (LDL - r["LFL"]) / 100.0

def con_approach_speed(x_norm):
    """C9: Approach speed <= Va_limit."""
    r = _eval(x_norm)
    return (Va_limit - r["Va"]) / 50.0

def con_gust(x_norm):
    """C10: Wing loading >= gust sensitivity limit."""
    r = _eval(x_norm)
    return (r["Mg_S"]- r["gust"]) / 1000.0

def con_wingspan(x_norm):
    """C11: Wing span <= maximum allowed."""
    r = _eval(x_norm)
    return (b_limit - r["b_struct"]) / 10.0

def con_machnumber(x_norm):
    """C12: Cruise Mach number <= drag divergence Mach number."""
    r = _eval(x_norm)
    x = unnormalize(x_norm) 
    M_N_c = x[7]
    return (r["M_DD_wing"]  - M_N_c) #subtract by 0.05 if needed 

def con_mass_closure(x_norm):
    """C13: Mto (design variable) == Mtotal computed (equality)."""
    r = _eval(x_norm)
    x = unnormalize(x_norm)
    Mto = x[9]
    return (r["Mtotal"] - Mto) / 100000.0

constraints = [
    {"type": "eq",   "fun": con_fuel_balance},
    {"type": "ineq", "fun": con_fuel_volume},
    {"type": "ineq", "fun": con_tofl},
    {"type": "ineq", "fun": con_climb_gradient},
    {"type": "ineq", "fun": con_roc},
    {"type": "ineq", "fun": con_buffet},
    {"type": "ineq", "fun": con_thrust_drag},
    {"type": "ineq", "fun": con_landing_dist},
    {"type": "ineq", "fun": con_approach_speed},
    {"type": "ineq", "fun": con_gust},
    {"type": "ineq", "fun": con_wingspan},
    {"type": "ineq", "fun": con_machnumber},
    {"type": "eq",   "fun": con_mass_closure},
]

# ========================================================================
#  Run with Plotting
# ========================================================================

def plot_convergence(history):
    """
    Plot the convergence history of the design variables.
    """
    # Convert history (normalized) to separate physical lists
    history_phys = []
    for x_norm in history:
        history_phys.append(unnormalize(x_norm))
    history_phys = np.array(history_phys)
    
    iterations = range(len(history_phys))
    labels = ["Aspect Ratio", "Wing Area (m^2)", "Sweep (deg)", "t/c", "Taper Ratio", 
              "Thrust (N)", "Fuel Mass Frac", "Cruise Mach", "Cruise Alt (m)", "MTOW (kg)"]
    
    # Create a 2x5 grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("SLSQP Optimization Convergence Trends", fontsize=16)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.plot(iterations, history_phys[:, i], 'b.-', linewidth=1.5, markersize=8)
        ax.set_title(labels[i], fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("Iteration")
        
        # improved formatting
        ax.tick_params(axis='both', which='major', labelsize=9)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # Make room for title
    plt.show()


# ========================================================================
#  Run
# ========================================================================

def run_optimisation(x0=x0, verbose=True):
    """Execute the SLSQP optimisation and return the scipy result."""
    
    # 1. Normalize Initial Guess
    x0_norm = normalize(np.array(x0))

     # History tracking
    history = [x0_norm]
    
    def callback(xk):
        # Callback returns xk (current normalized design variables)
        # We append a copy to our history list
        history.append(xk.copy())
    
    # 2. Define Normalization Bounds for Optimizer
    # Since we are normalizing, the bounds for every variable are [0.0, 1.0]
    # We add a tiny buffer (eps) to prevent issues at exactly 0.0 or 1.0 if sensitive
    bounds_norm = [(0.0, 1.0) for _ in x0]

    if verbose:
        print("=" * 60)
        print("  Aircraft Preliminary Design Optimisation")
        print("=" * 60)
        print(f"\n  Initial guess (Physical): {np.array2string(np.asarray(x0), precision=2)}")
        # Calculate true initial mass for display
        r0_disp = evaluate_all(x0)
        
        # Print initial performance/constraints matching the end output
        print_constraints(r0_disp)
        
        print(f"  Initial Mtotal: {r0_disp['Mtotal']:.4f} kg")

        # --- Print Mass Breakdown BEFORE Optimization ---
        AR0, Sw0, QW_40, t_c0, taper0, T00, FMf0, M_N_c0, h_c0, Mto0 = x0
        Vdive0 = r0_disp["Vdive"]
        MC.print_mass_breakdown(AR0, Sw0, QW_40, t_c0, taper0, T00, FMf0, Mto0, Vdive0)

        print()

    sol = minimize(
        objective,
        x0_norm,              # Pass NORMALIZED guess
        method="SLSQP",
        bounds=bounds_norm,   # Pass NORMALIZED bounds
        constraints=constraints,
        callback=callback,    # <--- Add callback here
        options={
            "maxiter": 1000,      
            "ftol": 1e-4,          
            "disp": verbose,
        },
    )

    # 3. CONVERT RESULT BACK TO PHYSICAL UNITS
    x_opt_physical = unnormalize(sol.x)
    sol.x = x_opt_physical # overwrite in scale for reporting
    
    # Re-calculate function value with physical units for reporting accuracy if needed
    # But sol.fun is the normalized objective value.
    
    if verbose:
        print("\n" + "=" * 60)
        print("  Optimisation Result")
        print("=" * 60)
        labels = ["AR", "Sw", "QW_4", "t/c", "taper", "T0", "FMf", "M_N_c", "h_c", "Mto"]
        for lbl, val in zip(labels, sol.x):
            print(f"  {lbl:>8s} = {val:12.4f}")
        
        # Recalculate true mass
        r_opt = evaluate_all(sol.x)
        print(f"\n  Optimal Mtotal = {r_opt['Mtotal']:.1f} kg")
        print(f"  Success: {sol.success}")
        print(f"  Message: {sol.message}\n")

        print_constraints(r_opt)

        Vdive = r_opt["Vdive"]
        MC.print_mass_breakdown(sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4], sol.x[5], sol.x[6], sol.x[9], Vdive)


    # Final constraint check (only printed once if verbose=True)
    check_constraints_post_optimization(sol, verbose=verbose)

    # Plot convergence history
    print("\nGenerating convergence plots...")
    plot_convergence(history)

    return sol


def check_constraints_post_optimization(sol, tolerance=1.0, verbose=True):
    """
    Evaluate all constraints after optimization and print in compact format.
    
    Parameters:
    -----------
    sol : scipy OptimizeResult
        The optimization result from run_optimisation()
    tolerance : float
        Tolerance for constraints (default 1.0 for physical units like kg)
    """
    r = evaluate_all(sol.x)
    
    print("\n" + "=" * 80)
    print("  POST-OPTIMIZATION CONSTRAINT CHECK")
    print("=" * 80)
    
    constraints_data = [
        ("C1: Fuel Balance", "eq",  r["M_fuel_FFM"] - r["M_fuel_req"]),
        ("C2: Fuel Volume", "ineq", r["M_fuel_vol"] - r["M_fuel_req"]),
        ("C3: TOFL", "ineq", TOFL - r["ToL"]),
        ("C4: Climb Gradient", "ineq", r["gamma2"] - 0.024),
        ("C5: Rate of Climb", "ineq", r["RoC"] - 1.5),
        ("C6: Buffet Margin", "ineq", r["CL_buffet"] - r["CL_c"]),
        ("C7: Thrust vs Drag", "ineq", r["T_c"] - r["D_c"]),
        ("C8: Landing Distance", "ineq", LDL - r["LFL"]),
        ("C9: Approach Speed", "ineq", Va_limit - r["Va"]),
        ("C10: Gust Sensitivity", "ineq", r["Mg_S"] - r["gust"]),
        ("C11: Wing Span", "ineq", b_limit - r["b_struct"]),
        ("C12: Drag Divergence Mach Number", "ineq", r["M_DD_wing"] - sol.x[7]),
        ("C13: Mass Closure", "eq", sol.x[9] - r["Mtotal"]),
    ]
    
    failures = []
    for name, c_type, diff in constraints_data:
        # FIXED: Correct logic for equality vs inequality
        if c_type == "eq":
            # Equality: difference should be close to zero
            is_satisfied = abs(diff) <= tolerance
        else:  # ineq
            # Inequality: difference should be >= -tolerance (allows small numerical errors)
            is_satisfied = diff >= -tolerance
        
        status = "✓" if is_satisfied else "✗"
        print(f"  {status} {name:20s} | Diff: {diff:12.6f}")
        if not is_satisfied:
            failures.append(name)
    
    print("=" * 80)
    print(f"  Summary: {len(constraints_data) - len(failures)}/{len(constraints_data)} constraints satisfied", end="")
    if failures:
        print(f" | Failed: {', '.join(failures)}")
    else:
        print(" | ✓ All passed!")
    print("=" * 80)


if __name__ == "__main__":
    run_optimisation()