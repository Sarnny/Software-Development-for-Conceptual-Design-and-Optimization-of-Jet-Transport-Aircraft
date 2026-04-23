import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.optimize import minimize
import threading
import sys
import io
import time
import os
from PIL import Image, ImageTk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from InputParameters import x0, TOFL, LDL, Va_limit, b_limit, M_N_c, h_c
from Performance_Constraint_Analysis import evaluate_all, print_constraints
import MassCalculation as MC

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

def con_fuel_balance(x_norm): r = _eval(x_norm); return (r['M_fuel_FFM'] - r['M_fuel_req']) / 20000.0
def con_fuel_volume(x_norm): r = _eval(x_norm); return (r['M_fuel_vol'] - r['M_fuel_req']) / 20000.0
def con_tofl(x_norm): r = _eval(x_norm); return (TOFL - r['ToL']) / 100.0
def con_climb_gradient(x_norm): r = _eval(x_norm); return (r['gamma2'] - 0.024)
def con_roc(x_norm): r = _eval(x_norm); return (r['RoC'] - 1.5) / 5.0
def con_buffet(x_norm): r = _eval(x_norm); return (r['CL_buffet'] - r['CL_c']) / 1e-2
def con_thrust_drag(x_norm): r = _eval(x_norm); return (r['T_c'] - r['D_c']) / 1000.0
def con_landing_dist(x_norm): r = _eval(x_norm); return (LDL - r['LFL']) / 100.0
def con_approach_speed(x_norm): r = _eval(x_norm); return (Va_limit - r['Va']) / 50.0
def con_gust(x_norm): r = _eval(x_norm); return (r['Mg_S'] - r['gust']) / 1000.0
def con_wingspan(x_norm): r = _eval(x_norm); return (b_limit - r['b_struct']) / 10.0
def con_machnumber(x_norm): r = _eval(x_norm); x = unnormalize(x_norm); return (r['M_DD_wing'] - x[7])
def con_mass_closure(x_norm): r = _eval(x_norm); x = unnormalize(x_norm); return (r['Mtotal'] - x[9]) / 100000.0

constraints = [
    {'type': 'eq',   'fun': con_fuel_balance},
    {'type': 'ineq', 'fun': con_fuel_volume},
    {'type': 'ineq', 'fun': con_tofl},
    {'type': 'ineq', 'fun': con_climb_gradient},
    {'type': 'ineq', 'fun': con_roc},
    {'type': 'ineq', 'fun': con_buffet},
    {'type': 'ineq', 'fun': con_thrust_drag},
    {'type': 'ineq', 'fun': con_landing_dist},
    {'type': 'ineq', 'fun': con_approach_speed},
    {'type': 'ineq', 'fun': con_gust},
    {'type': 'ineq', 'fun': con_wingspan},
    {'type': 'ineq', 'fun': con_machnumber},
    {'type': 'eq',   'fun': con_mass_closure},
]

class OptimizationGUI_Module2(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Aircraft Design Optimization - Module 2 (Mach Sweep)')
        self.geometry('1400x900')
        
        self.columnconfigure(0, weight=0, minsize=260)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)

        self.var_names = [
            'Aspect ratio (AR)', 'Sw', 'Quarter-chord sweep', 'thickness to chord', 'taper ratio', 
            'Sea level thrust', 'FMf', 'M_N_c', 'h_c', 'Mtoini'
        ]
        self.var_keys = ['AR', 'Sw', 'QW_4', 't/c', 'taper ratio', 'T0', 'FMf', 'M_N_c', 'h_c', 'Mtoini']

        default_guess = np.array([8.0, 120.0, 30.0, 0.2, 0.4, 200000.0, 0.2, 0.82, 10000.0, 60000.0])
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.size == 10:
            default_guess = x0_arr.copy()
        elif x0_arr.size == 9:
            default_guess = np.array([x0_arr[0], x0_arr[1], x0_arr[2], x0_arr[3], x0_arr[4], x0_arr[5], x0_arr[6], float(M_N_c), x0_arr[7], x0_arr[8]])
        elif x0_arr.size == 8:
            default_guess = np.array([x0_arr[0], x0_arr[1], x0_arr[2], x0_arr[3], x0_arr[4], x0_arr[5], x0_arr[6], float(M_N_c), float(h_c), x0_arr[7]])

        self.entries = []
        self.opt_res_vars = [tk.StringVar(value='num') for _ in range(10)]
        self.mach_mto_vars = {f'{m:.2f}': tk.StringVar(value='---') for m in __import__('numpy').linspace(0.5, 0.9, 9)}
        self.status_vars = {
            'success': tk.StringVar(value='-'),
            'message': tk.StringVar(value='-'),
            'iterations': tk.StringVar(value='-'),
            'time': tk.StringVar(value='-'),
            'constraints': tk.StringVar(value='-')
        }

        self._build_layout(default_guess)

    def _build_layout(self, default_guess):
        top_left = tk.Frame(self)
        top_left.grid(row=0, column=0, sticky='n', padx=5, pady=0)
        
        tk.Label(top_left, text='Optimizer:', font=('Arial', 14, 'bold')).pack(pady=(10, 2))
        self.optimizer_var = tk.StringVar(value='SLSQP')
        self.opt_combo = ttk.Combobox(top_left, textvariable=self.optimizer_var, values=['SLSQP', 'trust-constr'], state='readonly', font=('Arial', 16), width=12)
        self.opt_combo.pack(pady=(0, 15), padx=10)
        
        self.run_btn = ttk.Button(top_left, text='Run Module 2 (Mach Sweep)', command=self.start_optimization)
        self.run_btn.pack(pady=5, ipadx=10, ipady=3)

        self.clear_btn = ttk.Button(top_left, text='Clear Results', command=self.clear_results)
        self.clear_btn.pack(pady=5, ipadx=10, ipady=3)

        self.download_btn = ttk.Button(top_left, text='Download Plot', command=self.download_plot)
        self.download_btn.pack(pady=5, ipadx=10, ipady=3)

        self.next_module_btn = ttk.Button(top_left, text='Next Module ->', command=self.switch_to_module3)
        self.next_module_btn.pack(pady=5, ipadx=10, ipady=3)

        # Removed Next Module button for Module 2 window.

        top_right = tk.LabelFrame(self, text='Initial guess (input)', font=('Arial', 16, 'bold'))
        top_right.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        for i, name in enumerate(self.var_names):
            row = i % 5
            col = (i // 5) * 2
            tk.Label(top_right, text=name, font=('Arial', 12)).grid(row=row, column=col, sticky='w', padx=15, pady=6)
            e = ttk.Entry(top_right, width=15, font=('Arial', 12))
            e.insert(0, str(default_guess[i]))
            e.bind('<KeyRelease>', self.clear_results)
            e.grid(row=row, column=col+1, padx=15, pady=6, sticky='w')
            if i == 7: # M_N_c (Mach)
                e.config(state='readonly')
            self.entries.append(e)

        try:
            logo_path = 'logo.png'
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                lbl_logo = tk.Label(top_right, image=self.logo_img)
                lbl_logo.grid(row=0, column=4, rowspan=5, padx=20, sticky='e')
                top_right.columnconfigure(4, weight=1)
        except Exception as e:
            print(f'Logo failed to load: {e}')



        mid_right = tk.Frame(self, bd=1, relief='solid')
        mid_right.grid(row=1, column=1, rowspan=3, sticky='nsew', padx=5, pady=5)
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self._init_empty_plots()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bot_left = tk.LabelFrame(self, text='Sweep Results (Mto)', font=('Arial', 16, 'bold'))
        bot_left.grid(row=2, column=0, sticky='new', padx=5, pady=0)
        
        tk.Label(bot_left, text='Mach Number', font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky='w', padx=10, pady=2)
        tk.Label(bot_left, text='Mto (kg)', font=('Arial', 11, 'bold')).grid(row=0, column=1, sticky='w', padx=10, pady=2)
        
        for r, m in enumerate(__import__('numpy').linspace(0.5, 0.9, 9), start=1):
            lbl_key = f'{m:.2f}'
            tk.Label(bot_left, text=lbl_key, font=('Arial', 11)).grid(row=r, column=0, sticky='w', padx=10, pady=1)
            tk.Label(bot_left, textvariable=self.mach_mto_vars[lbl_key], font=('Arial', 11), width=10, anchor='w').grid(row=r, column=1, sticky='w', padx=5, pady=1)

        status_frame = tk.LabelFrame(self, text='Optimization status', font=('Arial', 16, 'bold'))
        status_frame.grid(row=1, column=0, sticky='new', padx=5, pady=0)
        
        tk.Label(status_frame, text='Success', font=('Arial', 11)).grid(row=0, column=0, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['success'], font=('Arial', 11), width=10, anchor='w').grid(row=0, column=1, sticky='w', padx=5, pady=2)
        tk.Label(status_frame, text='Iterations', font=('Arial', 11)).grid(row=0, column=2, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['iterations'], font=('Arial', 11), width=6, anchor='w').grid(row=0, column=3, sticky='w', padx=5, pady=2)

        tk.Label(status_frame, text='Message', font=('Arial', 11)).grid(row=1, column=0, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['message'], font=('Arial', 11), wraplength=280, justify='left', anchor='w').grid(row=1, column=1, columnspan=3, sticky='w', padx=5, pady=2)
        
        tk.Label(status_frame, text='Feasibility', font=('Arial', 11)).grid(row=2, column=0, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['constraints'], font=('Arial', 11), width=10, anchor='w').grid(row=2, column=1, sticky='w', padx=5, pady=2)
        tk.Label(status_frame, text='Execution time', font=('Arial', 11)).grid(row=2, column=2, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['time'], font=('Arial', 11), width=10, anchor='w').grid(row=2, column=3, sticky='w', padx=5, pady=2)

    def _init_empty_plots(self):
        self.fig.clf()
        self.axs = self.fig.subplots(3, 2)
        self.fig.subplots_adjust(hspace=0.6, wspace=0.55, top=0.94, bottom=0.08, left=0.08, right=0.92)
        for ax in self.axs.flat:
            ax.clear()
            ax.plot([0, 1], [0, 1], color='#e0e0e0', lw=1)
            ax.plot([0, 1], [1, 0], color='#e0e0e0', lw=1)
            ax.set_xticks([])
            ax.set_yticks([])

        self.axs[0,0].set_xlabel('Mach Number', fontsize=10)
        self.axs[0,1].set_xlabel('Mach Number', fontsize=10)
        self.axs[1,0].set_xlabel('Mach Number', fontsize=10)
        self.axs[1,1].set_xlabel('Mach Number', fontsize=10)
        self.axs[2,0].set_xlabel('Mach Number', fontsize=10)
        self.axs[2,1].set_xlabel('Mach Number', fontsize=10)

    def clear_results(self, event=None):
        if self.status_vars['success'].get() not in ['-', 'Running...']:
            for k in self.mach_mto_vars: self.mach_mto_vars[k].set('---')
            self.status_vars['success'].set('-')
            self.status_vars['message'].set('-')
            self.status_vars['iterations'].set('-')
            self.status_vars['time'].set('-')
            self.status_vars['constraints'].set('-')
            
            self._init_empty_plots()
            self.canvas.draw()

    def download_plot(self):
        try:
            file_path = filedialog.asksaveasfilename(
                title='Save Module 2 Plot',
                defaultextension='.png',
                initialfile='module2_mach_sweep.png',
                filetypes=[('PNG Image', '*.png'), ('JPEG Image', '*.jpg;*.jpeg'), ('PDF File', '*.pdf'), ('All Files', '*.*')],
            )
            if not file_path:
                return
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo('Saved', f'Plot saved successfully to:\n{file_path}')
        except Exception as e:
            messagebox.showerror('Save Error', f'Could not save plot: {e}')
            
    def switch_to_module3(self):
        import subprocess
        import os
        import sys
        from tkinter import messagebox
        try:
            module3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GUI_Phase3.py')
            subprocess.Popen([sys.executable, module3_path])
        except Exception as e:
            messagebox.showerror('Error', f'Could not launch Module 3 App: {e}')

    def start_optimization(self):
        self.run_btn.config(state=tk.DISABLED)
        for k in self.mach_mto_vars: self.mach_mto_vars[k].set('...')
        self.status_vars['success'].set('Running...')
        self.status_vars['message'].set('...')
        self.status_vars['iterations'].set('...')
        self.status_vars['time'].set('...')
        self.status_vars['constraints'].set('...')

        try:
            x_initial = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror('Input Error', 'All fields must be valid numbers.')
            self.run_btn.config(state=tk.NORMAL)
            return

        threading.Thread(target=self.run_solver, args=(np.array(x_initial),), daemon=True).start()

    def run_solver(self, x_init_real):
        import time, io, sys
        start_t = time.time()
        try:
            x0_norm = normalize(x_init_real)
            opt_method = self.optimizer_var.get()
            
            mach_values = np.linspace(0.5, 0.9, 9)
            results_10d = []
            masses_hist = []


            class Tee(object):
                def __init__(self, *files):
                    self.files = files
                def write(self, obj):
                    for f in self.files:
                        f.write(obj)
                        f.flush()
                def flush(self):
                    for f in self.files:
                        f.flush()
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = Tee(sys.stdout, new_stdout)
            
            print('\n============================================================')
            print('  Starting Mach Sweep from 0.5 to 0.9')
            print('============================================================')

            sol = None
            r_opt = None
            feasibility_count = 0
            
            for cur_mach in mach_values:
                print(f"\n---> Optimizing for M_N_c = {cur_mach:.2f}...")
                b_lb, b_ub = BOUNDS[7]
                fixed_mach_norm = (cur_mach - b_lb) / (b_ub - b_lb) if (b_ub - b_lb) > 1e-9 else 0.5
                
                bounds10 = [(0.0, 1.0)] * 10
                bounds10[7] = (fixed_mach_norm, fixed_mach_norm)

                x0_norm[7] = fixed_mach_norm

                opts = {"maxiter": 1000, "disp": False}
                if opt_method == "SLSQP": 
                    opts["ftol"] = 1e-4
                    opts["maxiter"] = 200
                elif opt_method == "trust-constr":
                    opts["xtol"] = 1e-4


                sol = minimize(
                    objective, x0_norm, method=opt_method, bounds=bounds10, constraints=constraints, options=opts
                )

                # Retrieve Full state
                x10_opt_norm = sol.x
                x10_opt_real = unnormalize(x10_opt_norm)
                results_10d.append(x10_opt_real)
                r_opt = evaluate_all(x10_opt_real)
                masses_hist.append(r_opt)
                
                # Cold start: reset x0_norm for the next iteration to prevent gradient corruption
                x0_norm = normalize(x_init_real)
                
                Vdive = r_opt['Vdive']
                c_diffs = [
                    ('C1: Fuel Balance' , r_opt['M_fuel_FFM'] - r_opt['M_fuel_req'], 'eq'),
                    ('C2: Fuel Volume', r_opt['M_fuel_vol'] - r_opt['M_fuel_req'], 'ineq'),
                    ('C3: TOFL', TOFL - r_opt['ToL'], 'ineq'),
                    ('C4: Climb Gradient', r_opt['gamma2'] - 0.024, 'ineq'),
                    ('C5: Rate of Climb', r_opt['RoC'] - 1.5, 'ineq'),
                    ('C6: Buffet Margin', r_opt['CL_buffet'] - r_opt['CL_c'], 'ineq'),
                    ('C7: Thrust vs Drag', r_opt['T_c'] - r_opt['D_c'], 'ineq'),
                    ('C8: Landing Distance', LDL - r_opt['LFL'], 'ineq'),
                    ('C9: Approach Speed', Va_limit - r_opt['Va'], 'ineq'),
                    ('C10: Gust Sensitivity', r_opt['Mg_S'] - r_opt['gust'], 'ineq'),
                    ('C11: Wing Span', b_limit - r_opt['b_struct'], 'ineq'),
                    ('C12: Drag Divergence Mach Number', r_opt['M_DD_wing'] - x10_opt_real[7], 'ineq'),
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
                    print(f'  {symbol} {cname:<22s} | Diff: {cval:12.6f}')

                print('================================================================================')
                total_c = len(c_diffs)
                if satisfied == total_c:
                    print(f'  Summary: {satisfied}/{total_c} constraints satisfied | ✓ All passed!')
                    feasibility_count += 1
                else:
                    print(f'  Summary: {satisfied}/{total_c} constraints satisfied | x Partial failure')
                print('================================================================================')
                print(f'     Done. Success: {sol.success}, MTOW: {x10_opt_real[9]:.1f} kg')
                print(f'     Constraints: {"Passed" if satisfied==total_c else "Failed"}')
                print('     Design Variables:')
                for lbl, val in zip(self.var_keys, x10_opt_real):
                    print(f'             {lbl:>8s} = {val:12.4f}') 

            ext_time = time.time() - start_t
            print(f"\nModule 2 Completed successfully: {sol.success} in {ext_time:.2f}s")
            
            sys.stdout = old_stdout
            out_file = 'Optimization_Module2_Results.txt'
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(new_stdout.getvalue())

            last_x = results_10d[-1]
            last_r = masses_hist[-1]
            Vdive = last_r['Vdive']
            Mls = MC.M_lifting_surface(last_x[0], last_x[1], last_x[2], last_x[3], last_x[4], last_x[9], Vdive)
            Mpower = MC.M_powerplant(last_x[5])
            Msys = MC.M_systems(last_x[9])
            Mfuel = MC.M_fuel(last_x[6], last_x[9])
            Mfuse = MC.M_fuselage()
            Mpayload = MC.M_payload()
            Mop = MC.M_operational()
            Mtotal_calc = last_r['Mtotal']

            self.after(0, self._update_gui_post_calc, sol.success, f"Sweep Done ({feasibility_count} / {len(mach_values)} Feasible)", sol.nit, ext_time, f"{feasibility_count} / {len(mach_values)}", last_x, mach_values, np.array(results_10d))

        except Exception as e:
            self.after(0, self._handle_error, str(e))

    def _update_gui_post_calc(self, success, message, nit, ext_time, constraints_str, x_opt, mach_values, results_10d):
        self.status_vars['success'].set(str(success))
        self.status_vars['message'].set(str(message))
        self.status_vars['iterations'].set(str(nit))
        self.status_vars['time'].set(f'{ext_time:.2f} s')
        self.status_vars['constraints'].set(constraints_str)


        for cur_mach, x10_real in zip(mach_values, results_10d):
            lbl_key = f'{cur_mach:.2f}'
            if lbl_key in self.mach_mto_vars:
                self.mach_mto_vars[lbl_key].set(f'{x10_real[9]:.2f}')

        self._plot_module2(mach_values, results_10d)
        self.run_btn.config(state=tk.NORMAL)

    def _handle_error(self, err_msg):
        messagebox.showerror('Optimization Error', err_msg)
        self.status_vars['success'].set('Error')
        self.status_vars['message'].set('Failed')
        self.status_vars['constraints'].set('...')
        self.run_btn.config(state=tk.NORMAL)

    def _plot_module2(self, mach_values, results_10d):
        # Configure subplots 3x3 for the 9 variables against Mach number
        self.fig.clf()
        self.axs = self.fig.subplots(3, 3)
        self.fig.subplots_adjust(hspace=0.6, wspace=0.45, top=0.94, bottom=0.08, left=0.08, right=0.95)
        
        plot_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9] # Exclude Mach (index 7)
        labels = ['AR', 'Sw', 'Sweep (deg)', 't/c', 'Taper Ratio', 'Thrust (N)', 'Fuel Mass Frac', 'Altitude (m)', 'Takeoff Mass (kg)']
        
        # Y-axis limits matching reference image
        y_limits = [
            (9, 25),           # AR
            (90, 140),         # Sw
            (0, 40),           # Sweep (deg)
            (0.1, 0.25),      # t/c
            (0.25, 0.45),      # Taper Ratio
            (100000, 200000),  # Thrust (N)
            (0.1, 0.30),      # Fuel Mass Frac
            (7000, 13000),     # Altitude (m)
            (50000, 70000)     # Takeoff Mass (kg)
        ]

        for idx, (var_idx, lbl, (y_min, y_max)) in enumerate(zip(plot_indices, labels, y_limits)):
            row, col = divmod(idx, 3)
            ax = self.axs[row, col]
            y_vals = results_10d[:, var_idx]
            ax.plot(mach_values, y_vals, 'o-', color='#2E7D32', markersize=5, linewidth=2)
            ax.set_title(lbl, fontsize=11, fontweight='bold')
            ax.set_xlabel('M_N_c', fontsize=9)
            
            # Keep X-axis consistent across all plots
            ax.set_xlim(0.45, 0.95)
            ax.set_xticks(np.arange(0.5, 1.0, 0.1))       
            ax.tick_params(axis='both', labelsize=8)
            ax.set_ylabel(lbl, fontsize=9)
            
            # Apply fixed y-axis limits
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
            
            # Smart formatting based on value magnitudes
            if lbl in ['t/c', 'Taper Ratio']:
                ax.yaxis.set_major_formatter(__import__('matplotlib.ticker').ticker.FormatStrFormatter('%.2f'))
            elif lbl == 'AR':
                ax.yaxis.set_major_formatter(__import__('matplotlib.ticker').ticker.FormatStrFormatter('%.0f'))
            elif lbl in ['Thrust (N)', 'Altitude (m)', 'Takeoff Mass (kg)']:
                ax.ticklabel_format(style='plain', axis='y')
        
        self.canvas.draw()

if __name__ == '__main__':
    app = OptimizationGUI_Module2()
    app.mainloop()
