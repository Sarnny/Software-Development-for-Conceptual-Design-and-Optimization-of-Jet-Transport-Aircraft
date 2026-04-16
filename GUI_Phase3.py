import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.optimize import minimize
import threading
import sys
import io

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from InputParameters import x0, TOFL, LDL, Va_limit, b_limit, M_N_c, h_c
from Performance_Constraint_Analysis import evaluate_all
import MassCalculation as MC

# -----------------------------------------------------------------------------
# CORE OPTIMIZATION SETUP
# -----------------------------------------------------------------------------

BOUNDS = [
    (5.0, 50.0),           # 0: AR
    (50.0, 1000.0),        # 1: Sw (m^2)
    (0.0, 50.0),           # 2: QW_4 (deg)
    (0.1, 0.5),            # 3: t/c
    (0.3, 1.0),            # 4: taper_ratio
    (10000.0, 1000000.0),  # 5: T0 (N)
    (0.1, 0.5),            # 6: FMf
    (0.3, 1.0),            # 7: M_N_c
    (3000.0, 100000.0),    # 8: h_c
    (10000.0, 1000000.0),  # 9: Mto
]

def normalize(x_real):
    x_norm = []
    for i, val in enumerate(x_real):
        lb, ub = BOUNDS[i]
        if abs(ub - lb) < 1e-9:
            x_norm.append(0.5)
        else:
            x_norm.append((val - lb) / (ub - lb))
    return np.array(x_norm)

def unnormalize(x_norm):
    x_real = []
    for i, val in enumerate(x_norm):
        lb, ub = BOUNDS[i]
        x_real.append(lb + val * (ub - lb))
    return np.array(x_real)

def objective(x_norm):
    x = unnormalize(x_norm)
    try:
        r = evaluate_all(x)
        return r['Mtotal'] / 50000.0
    except Exception:
        return 1e9

def _eval(x_norm):
    x = unnormalize(x_norm)
    return evaluate_all(x)

def con_fuel_balance(x_norm): return (_eval(x_norm)["M_fuel_FFM"] - _eval(x_norm)["M_fuel_req"]) / 20000.0
def con_fuel_volume(x_norm): return (_eval(x_norm)["M_fuel_vol"] - _eval(x_norm)["M_fuel_req"]) / 20000.0 
def con_tofl(x_norm): return (TOFL - _eval(x_norm)["ToL"]) / 100.0  
def con_climb_gradient(x_norm): return (_eval(x_norm)["gamma2"] - 0.024) 
def con_roc(x_norm): return (_eval(x_norm)["RoC"] - 1.5) / 5.0
def con_buffet(x_norm): return (_eval(x_norm)["CL_buffet"] - _eval(x_norm)["CL_c"]) / 1e-2
def con_thrust_drag(x_norm): return (_eval(x_norm)["T_c"] - _eval(x_norm)["D_c"]) / 1000.0
def con_landing_dist(x_norm): return (LDL - _eval(x_norm)["LFL"]) / 100.0
def con_approach_speed(x_norm): return (Va_limit - _eval(x_norm)["Va"]) / 50.0
def con_gust(x_norm): return (_eval(x_norm)["Mg_S"] - _eval(x_norm)["gust"]) / 1000.0
def con_wingspan(x_norm): return (b_limit - _eval(x_norm)["b_struct"]) / 10.0
def con_machnumber(x_norm): return (_eval(x_norm)["M_DD_wing"] - unnormalize(x_norm)[7])
def con_mass_closure(x_norm): return (_eval(x_norm)["Mtotal"] - unnormalize(x_norm)[9]) / 100000.0

constraints = [
    {"type": "eq", "fun": con_fuel_balance}, {"type": "ineq", "fun": con_fuel_volume},
    {"type": "ineq", "fun": con_tofl}, {"type": "ineq", "fun": con_climb_gradient},
    {"type": "ineq", "fun": con_roc}, {"type": "ineq", "fun": con_buffet},
    {"type": "ineq", "fun": con_thrust_drag}, {"type": "ineq", "fun": con_landing_dist},
    {"type": "ineq", "fun": con_approach_speed}, {"type": "ineq", "fun": con_gust},
    {"type": "ineq", "fun": con_wingspan}, {"type": "ineq", "fun": con_machnumber},
    {"type": "eq", "fun": con_mass_closure},
]

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# -----------------------------------------------------------------------------
# PHASE 3 GUI INTERFACE
# -----------------------------------------------------------------------------

class OptimizationGUI_Phase3(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Aircraft Design Optimization - Phase 3 (Carpet Plot Sweep)')
        self.geometry('1400x900')
        
        self.columnconfigure(0, weight=0, minsize=350)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.var_names = [
            'Aspect ratio (AR)', 'Sw', 'Quarter-chord', 't/c', 'Taper ratio', 
            'SL Thrust (N)', 'FMf', 'M_N_c (Unused)', 'h_c (Unused)', 'Mtoini'
        ]
        
        # Load custom defaults if available natively from the script context
        default_guess = np.array([8.0, 120.0, 30.0, 0.2, 0.4, 200000.0, 0.2, 0.82, 10000.0, 60000.0])
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.size == 10: default_guess = x0_arr.copy()
        elif x0_arr.size == 9: default_guess = np.array([x0_arr[0], x0_arr[1], x0_arr[2], x0_arr[3], x0_arr[4], x0_arr[5], x0_arr[6], float(M_N_c), x0_arr[7], x0_arr[8]])
        elif x0_arr.size == 8: default_guess = np.array([x0_arr[0], x0_arr[1], x0_arr[2], x0_arr[3], x0_arr[4], x0_arr[5], x0_arr[6], float(M_N_c), float(h_c), x0_arr[7]])

        self.entries = []
        self._build_layout(default_guess)
        
    def _build_layout(self, default_guess):
        # LEFT PANEL
        left_panel = tk.Frame(self)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        tk.Label(left_panel, text='Phase 3: Mach & Altitude Sweep', font=('Arial', 14, 'bold')).pack(pady=(10, 10))
        
        opt_frame = tk.LabelFrame(left_panel, text='Optimizer Controls', font=('Arial', 12, 'bold'))
        opt_frame.pack(fill='x', pady=5)
        
        self.optimizer_var = tk.StringVar(value='SLSQP')
        self.opt_combo = ttk.Combobox(opt_frame, textvariable=self.optimizer_var, values=['SLSQP', 'trust-constr'], state='readonly', font=('Arial', 12))
        self.opt_combo.pack(pady=10, padx=10, fill='x')
        
        self.run_btn = ttk.Button(opt_frame, text='Generate Carpet Plot', command=self.start_optimization)
        self.run_btn.pack(pady=5, padx=10, fill='x', ipady=5)
        
        self.clear_btn = ttk.Button(opt_frame, text='Clear Results', command=self.clear_results)
        self.clear_btn.pack(pady=5, padx=10, fill='x', ipady=5)

        self.download_btn = ttk.Button(opt_frame, text='Download Plot', command=self.download_plot)
        self.download_btn.pack(pady=5, padx=10, fill='x', ipady=5)
        
        guess_frame = tk.LabelFrame(left_panel, text='Initial Guess Seed', font=('Arial', 12, 'bold'))
        guess_frame.pack(fill='x', pady=5)
        
        for i, name in enumerate(self.var_names):
            f = tk.Frame(guess_frame)
            f.pack(fill='x', pady=2, padx=5)
            tk.Label(f, text=name, font=('Arial', 10), width=18, anchor='w').pack(side='left')
            e = ttk.Entry(f, width=15, font=('Arial', 10))
            e.insert(0, str(default_guess[i]))
            # Lock M_N_c and h_c because they will be swept dynamically
            if i == 7 or i == 8:
                e.config(state='readonly')
            e.pack(side='right')
            self.entries.append(e)
            
        status_frame = tk.LabelFrame(left_panel, text='Status', font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', pady=5)
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(status_frame, textvariable=self.status_var, font=('Arial', 11), fg='blue', wraplength=300).pack(pady=5)
        self.progress_var = tk.StringVar(value='Points Simulated: 0 / 30')
        tk.Label(status_frame, textvariable=self.progress_var, font=('Arial', 11, 'bold')).pack(pady=5)
        
        # RIGHT PANEL (PLOT CANVAS)
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
    def clear_results(self):
        self.ax.clear()
        self.ax.set_title("Takeoff Mass Variation with Cruise Mach Number and Altitude", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_ylabel("Takeoff Mass (kg)", fontsize=11, fontweight='bold')
        self.ax.set_xticks([])
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        self.canvas.draw()
        
        self.status_var.set('Ready to optimize.\nSelect settings and click Run.')
        self.progress_var.set('Points Simulated: 0 / 30')

    def download_plot(self):
        try:
            file_path = filedialog.asksaveasfilename(
                title='Save Phase 3 Plot',
                defaultextension='.png',
                initialfile='phase3_carpet_plot.png',
                filetypes=[('PNG Image', '*.png'), ('JPEG Image', '*.jpg;*.jpeg'), ('PDF File', '*.pdf'), ('All Files', '*.*')],
            )
            if not file_path:
                return
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo('Saved', f'Plot saved successfully to:\n{file_path}')
        except Exception as e:
            messagebox.showerror('Save Error', f'Could not save plot: {e}')

    def start_optimization(self):
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set('Optimizing 30 configurations...\n(This will output to Terminal too)')
        self.progress_var.set('Points Simulated: 0 / 30')
        import time
        self.start_t = time.time()
        try:
            x_initial = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror('Input Error', 'All active fields must be real numbers.')
            self.run_btn.config(state=tk.NORMAL)
            return

        threading.Thread(target=self.run_solver, args=(np.array(x_initial),), daemon=True).start()

    def run_solver(self, x_init_real):
        import time
        try:
            # Definition of Phase 3 sweeps
            mach_list = [0.5, 0.6, 0.7, 0.8, 0.9]
            alt_list = [8000, 9000, 10000, 11000, 12000, 13000]
            
            mtow_matrix = np.full((len(mach_list), len(alt_list)), np.nan)
            feas_matrix = np.full((len(mach_list), len(alt_list)), False)
            
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = Tee(sys.stdout, new_stdout)
            
            print('============================================================')
            print('  Starting 2D Sweep: h_c (8000 to 13000) & M_N_c (0.5 to 0.9)')
            print('============================================================')
            
            total_pts = len(mach_list) * len(alt_list)
            completed_pts = 0
            feasible_count = 0
            opt_method = self.optimizer_var.get()
            lb_vec = np.array([b[0] for b in BOUNDS], dtype=float)
            ub_vec = np.array([b[1] for b in BOUNDS], dtype=float)
            
            for i, cur_mach in enumerate(mach_list):
                for j, cur_alt in enumerate(alt_list):
                    print(f"\n---> Optimizing for h_c = {cur_alt:.0f}, M_N_c = {cur_mach:.2f}...")
                    
                    fixed_mach_norm = (cur_mach - BOUNDS[7][0]) / (BOUNDS[7][1] - BOUNDS[7][0])
                    fixed_alt_norm = (cur_alt - BOUNDS[8][0]) / (BOUNDS[8][1] - BOUNDS[8][0])
                    
                    bounds10 = [(0.0, 1.0)] * 10
                    bounds10[7] = (fixed_mach_norm, fixed_mach_norm)  # Lock Mach
                    bounds10[8] = (fixed_alt_norm, fixed_alt_norm)    # Lock Altitude
                    
                    # Refresh initial guess so gradients are completely localized
                    x0_norm = normalize(x_init_real)
                    x0_norm[7] = fixed_mach_norm
                    x0_norm[8] = fixed_alt_norm
                    
                    opts = {"maxiter": 1000, "disp": False}
                    if opt_method == "SLSQP":
                        opts["ftol"] = 1e-4
                        opts["maxiter"] = 200
                    elif opt_method == "trust-constr":
                        opts["xtol"] = 1e-4

                    sol = minimize(objective, x0_norm, method=opt_method, bounds=bounds10, constraints=constraints, options=opts)

                    # Project to physical bounds to avoid tiny numerical spillover in reported values.
                    x10_opt_norm = np.clip(sol.x, 0.0, 1.0)
                    x10_opt_real = np.clip(unnormalize(x10_opt_norm), lb_vec, ub_vec)
                    x10_opt_real[7] = cur_mach
                    x10_opt_real[8] = cur_alt
                    r_opt = evaluate_all(x10_opt_real)
                    
                    c_diffs = [
                        ('C1: Fuel Balance', r_opt['M_fuel_FFM'] - r_opt['M_fuel_req'], 'eq'), ('C2: Fuel Volume', r_opt['M_fuel_vol'] - r_opt['M_fuel_req'], 'ineq'),
                        ('C3: TOFL', TOFL - r_opt['ToL'], 'ineq'), ('C4: Climb Gradient', r_opt['gamma2'] - 0.024, 'ineq'), ('C5: Rate of Climb', r_opt['RoC'] - 1.5, 'ineq'),
                        ('C6: Buffet Margin', r_opt['CL_buffet'] - r_opt['CL_c'], 'ineq'), ('C7: Thrust vs Drag', r_opt['T_c'] - r_opt['D_c'], 'ineq'),
                        ('C8: Landing Distance', LDL - r_opt['LFL'], 'ineq'), ('C9: Approach Speed', Va_limit - r_opt['Va'], 'ineq'), ('C10: Gust Sensitivity', r_opt['Mg_S'] - r_opt['gust'], 'ineq'),
                        ('C11: Wing Span', b_limit - r_opt['b_struct'], 'ineq'), ('C12: Drag Divergence Mach Number', r_opt['M_DD_wing'] - x10_opt_real[7], 'ineq'),
                        ('C13: Mass Closure', r_opt['Mtotal'] - x10_opt_real[9], 'eq')
                    ]
                    
                    print('\n================================================================================')
                    print('  POST-OPTIMIZATION CONSTRAINT CHECK')
                    print('================================================================================')
                    satisfied = 0
                    for cname, cval, ctype in c_diffs:
                        is_pass = False
                        if ctype == 'eq' and abs(cval) <= 1.0: is_pass = True
                        elif ctype == 'ineq' and cval >= -1.0: is_pass = True
                        
                        if is_pass: 
                            satisfied += 1
                            symbol = '✓'
                        else: 
                            symbol = 'x'
                        print(f'  {symbol} {cname:<24s} | Diff: {cval:12.6f}')
                        
                    mtow_matrix[i, j] = x10_opt_real[9]
                    
                    print('================================================================================')
                    total_c = len(c_diffs)
                    if satisfied == total_c:
                        print(f'  Summary: {satisfied}/{total_c} constraints satisfied | ✓ All passed!')
                        feas_matrix[i, j] = True
                        feasible_count += 1
                    else:
                        print(f'  Summary: {satisfied}/{total_c} constraints satisfied | x Partial failure')

                    print('\n     Design Variables:')
                    var_keys = ['AR', 'Sw', 'QW_4', 't/c', 'taper ratio', 'T0', 'FMf', 'M_N_c', 'h_c', 'Mtoini']
                    for lbl, val in zip(var_keys, x10_opt_real):
                        print(f'             {lbl:>11s} = {val:12.4f}')
                        
                    completed_pts += 1
                    self.after(0, self._update_progress, completed_pts, total_pts)
            
            print('\n================================================================================')
            print('  SUMMARY: MTOW (kg) VARIATION ACROSS MACH AND ALTITUDE')
            print('================================================================================')
            header_str = f"{'Altitude (m)':<14}|" + "".join([f"{m:>12.1f}" for m in mach_list])
            print(f"{'':<14} Cruise Mach Number")
            print(header_str)
            print("-" * len(header_str))
            
            for j, alt in enumerate(alt_list):
                row_str = f"{alt:<14}|"
                for i, mach in enumerate(mach_list):
                    val = mtow_matrix[i, j]
                    val_str = f"{val:12.1f}" if not np.isnan(val) else f"{'NaN':>12}"
                    row_str += val_str
                print(row_str)
            print('================================================================================\n')

            ext_time = time.time() - self.start_t
            
            sys.stdout = old_stdout
            with open('Optimization_Phase3_Results.txt', 'w', encoding='utf-8') as f:
                f.write(new_stdout.getvalue())
                
            self.after(0, self._plot_carpet_and_status, mach_list, alt_list, mtow_matrix, feas_matrix, feasible_count, total_pts, ext_time)

        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, self._handle_error, str(e))
            
    def _update_progress(self, pts, total):
        self.progress_var.set(f'Points Simulated: {pts} / {total}')
        
    def _handle_error(self, err_msg):
        warn_msg = (
            'Optimization could not find a valid solution at the current grid point.\n\n'
            f'Details: {err_msg}\n\n'
            'Try a different initial guess or optimizer and run again.'
        )
        messagebox.showwarning('Optimization Warning', warn_msg)
        self.status_var.set('WARNING: Solver failed at grid point')
        self.run_btn.config(state=tk.NORMAL)

    def _plot_carpet_and_status(self, mach_list, alt_list, mtow_matrix, feas_matrix, feasible_count, total_pts, ext_time):
        self.ax.clear()
        
        # Exclude M=0.5 from the plot only
        if 0.5 in mach_list:
            idx_05 = mach_list.index(0.5)
            mach_list = mach_list[:idx_05] + mach_list[idx_05+1:]
            mtow_matrix = np.delete(mtow_matrix, idx_05, axis=0)
            feas_matrix = np.delete(feas_matrix, idx_05, axis=0)
            colors_M = ['#4b4d8c', '#158378', '#f2d847', '#ffba08']  # Adjusted palette without the first element
        else:
            colors_M = ['#302251', '#4b4d8c', '#158378', '#f2d847', '#ffba08']
        
        # Design layout values for custom Carpet Plot shifts linking M and Alt.
        w_M = 3.5 
        w_H = 1.0 
        
        X_grid = np.zeros((len(mach_list), len(alt_list)))
        for i in range(len(mach_list)):
            for j in range(len(alt_list)):
                X_grid[i, j] = i * w_M + j * w_H
                
        colors_H = ['#3f205c', '#542674', '#703080', '#954a7c', '#c16262', '#dca95f']
        
        # Draw Altitude curves (the vertical grids)
        for j in range(len(alt_list)):
            c = colors_H[j % len(colors_H)]
            self.ax.plot(X_grid[:, j], mtow_matrix[:, j], color=c, lw=1.2, label=f'h = {alt_list[j]} m')
            last_idx = len(mach_list) - 1
            if not np.isnan(mtow_matrix[last_idx, j]):
                self.ax.text(X_grid[last_idx, j] + 0.15, mtow_matrix[last_idx, j] + 100, f'{alt_list[j]}', ha='left', va='bottom', fontsize=8, rotation=60, color='#454545')
                
        # Draw Mach curves (the horizontal/primary sweeps)
        for i in range(len(mach_list)):
            c = colors_M[i % len(colors_M)]
            self.ax.plot(X_grid[i, :], mtow_matrix[i, :], color=c, lw=2.5, label=f'M = {mach_list[i]:.1f}')
            first_idx = 0
            if not np.isnan(mtow_matrix[i, first_idx]):
                self.ax.text(X_grid[i, first_idx] - 0.3, mtow_matrix[i, first_idx] + 200, f'{mach_list[i]:.1f}', ha='right', va='center', fontsize=10, fontweight='bold', color='#1e1e1e')
        
        # Overlay points: Feasible (Invisible), Infeasible (red X)
        for i in range(len(mach_list)):
            for j in range(len(alt_list)):
                if not feas_matrix[i, j]:
                    self.ax.plot(X_grid[i, j], mtow_matrix[i, j], 'rx', markersize=6, mew=2, label='Infeasible' if (i==0 and j==0) else "")

        # Determine global minimum explicitly for marking
        min_idx = np.unravel_index(np.nanargmin(mtow_matrix), mtow_matrix.shape)
        min_x = X_grid[min_idx[0], min_idx[1]]
        min_y = mtow_matrix[min_idx[0], min_idx[1]]
        self.ax.plot(min_x, min_y, 'ko', markersize=8, label='Global optimum point')
        
        # Determine design point specifically at Mach 0.8 and Alt 10000
        try:
            m_idx = mach_list.index(0.8)
            h_idx = alt_list.index(10000)
            design_x = X_grid[m_idx, h_idx]
            design_y = mtow_matrix[m_idx, h_idx]
            self.ax.plot(design_x, design_y, 'k*', markersize=10, label='Design point')
        except ValueError:
            pass
            
        self.ax.set_title("Takeoff Mass Variation with Cruise Mach Number and Altitude", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_ylabel("Takeoff Mass (kg)", fontsize=11, fontweight='bold')
        self.ax.set_xticks([])
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        # Custom structured legend
        from matplotlib.lines import Line2D
        handles, labels = self.ax.get_legend_handles_labels()
        
        # Left Legend (Mach and Altitude curves)
        left_legend_elements = []
        left_legend_elements.append(Line2D([0], [0], color='none', label='Mach Curves'))
        for i in range(len(mach_list)):
            left_legend_elements.append(Line2D([0], [0], color=colors_M[i % len(colors_M)], lw=2.5, label=f'M = {mach_list[i]:.1f}'))
            
        left_legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
        left_legend_elements.append(Line2D([0], [0], color='none', label='Altitude Curves'))
        for j in range(len(alt_list)):
            left_legend_elements.append(Line2D([0], [0], color=colors_H[j % len(colors_H)], lw=1.2, label=f'h = {alt_list[j]} m'))
            
        leg1 = self.ax.legend(handles=left_legend_elements, loc='upper left', frameon=False, fontsize=9)
        self.ax.add_artist(leg1)
        
        # Right Legend (Design point, Global optimum, Infeasible)
        right_handles = []
        right_labels = []
        for h, l in zip(handles, labels):
            if l and not l.startswith('M =') and not l.startswith('h =') and l not in right_labels:
                right_handles.append(h)
                right_labels.append(l)
                
        if right_handles:
            self.ax.legend(handles=right_handles, labels=right_labels, loc='upper right', frameon=False, fontsize=9)
        
        # Status text summary
        self.status_var.set(f'Sweep Complete!\nFeasible Points: {feasible_count} / {total_pts}\nExecution time: {ext_time:.1f}s\n\nData saved to Optimization_Phase3_Results.txt')
        self.run_btn.config(state=tk.NORMAL)
        self.canvas.draw()

if __name__ == '__main__':
    app = OptimizationGUI_Phase3()
    app.mainloop()