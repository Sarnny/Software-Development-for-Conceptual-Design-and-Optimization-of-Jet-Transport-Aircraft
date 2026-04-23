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

class OptimizationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Aircraft Design Optimization - Module 1')
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
        self.mass_vars = {k: tk.StringVar(value='num') for k in ['Mls', 'Mpower', 'Msys', 'Mfuel', 'Mfuse', 'Mpayload', 'Mop', 'Mtotal']}
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
        
        self.run_btn = ttk.Button(top_left, text='Run optimisation', command=self.start_optimization)
        self.run_btn.pack(pady=5, ipadx=10, ipady=3)

        self.clear_btn = ttk.Button(top_left, text='Clear Results', command=self.clear_results)
        self.clear_btn.pack(pady=5, ipadx=10, ipady=3)

        self.download_btn = ttk.Button(top_left, text='Download Plot', command=self.download_plot)
        self.download_btn.pack(pady=5, ipadx=10, ipady=3)

        self.next_module_btn = ttk.Button(top_left, text='Next Module ->', command=self.switch_to_module2)
        self.next_module_btn.pack(pady=5, ipadx=10, ipady=3)

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

        mid_left = tk.LabelFrame(self, text='Optimized result', font=('Arial', 16, 'bold'))
        mid_left.grid(row=1, column=0, sticky='new', padx=5, pady=0)
        for i, key in enumerate(self.var_keys):
            tk.Label(mid_left, text=key, font=('Arial', 11)).grid(row=i, column=0, sticky='w', padx=15, pady=1)
            tk.Label(mid_left, textvariable=self.opt_res_vars[i], font=('Arial', 11), width=15, anchor='w').grid(row=i, column=1, sticky='w', padx=15, pady=1)

        mid_right = tk.Frame(self, bd=1, relief='solid')
        mid_right.grid(row=1, column=1, rowspan=3, sticky='nsew', padx=5, pady=5)
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self._init_empty_plots()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bot_left = tk.LabelFrame(self, text='Mass Breakdown', font=('Arial', 16, 'bold'))
        bot_left.grid(row=2, column=0, sticky='new', padx=5, pady=0)
        
        tk.Label(bot_left, text='Variable Masses', font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky='w', padx=10, pady=2)
        tk.Label(bot_left, text='Fixed Masses', font=('Arial', 11, 'bold')).grid(row=0, column=2, sticky='w', padx=10, pady=2)

        vars_items = [('Mls', 1), ('Mpower', 2), ('Msys', 3), ('Mfuel', 4)]
        for k, r in vars_items:
            tk.Label(bot_left, text=k, font=('Arial', 11)).grid(row=r, column=0, sticky='w', padx=10, pady=1)
            tk.Label(bot_left, textvariable=self.mass_vars[k], font=('Arial', 11), width=10, anchor='w').grid(row=r, column=1, sticky='w', padx=5, pady=1)

        fixed_items = [('Mfuse', 1), ('Mpayload', 2), ('Mop', 3)]
        for k, r in fixed_items:
            tk.Label(bot_left, text=k, font=('Arial', 11)).grid(row=r, column=2, sticky='w', padx=10, pady=1)
            tk.Label(bot_left, textvariable=self.mass_vars[k], font=('Arial', 11), width=10, anchor='w').grid(row=r, column=3, sticky='w', padx=5, pady=1)

        tk.Label(bot_left, text='Mtotal', font=('Arial', 12, 'bold')).grid(row=4, column=2, sticky='w', padx=10, pady=2)
        tk.Label(bot_left, textvariable=self.mass_vars['Mtotal'], font=('Arial', 12, 'bold', 'underline'), width=10, anchor='w').grid(row=4, column=3, sticky='w', padx=5, pady=2)

        status_frame = tk.LabelFrame(self, text='Optimization status', font=('Arial', 16, 'bold'))
        status_frame.grid(row=3, column=0, sticky='new', padx=5, pady=0)
        
        tk.Label(status_frame, text='Success', font=('Arial', 11)).grid(row=0, column=0, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['success'], font=('Arial', 11), width=10, anchor='w').grid(row=0, column=1, sticky='w', padx=5, pady=2)
        tk.Label(status_frame, text='Iterations', font=('Arial', 11)).grid(row=0, column=2, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['iterations'], font=('Arial', 11), width=6, anchor='w').grid(row=0, column=3, sticky='w', padx=5, pady=2)

        tk.Label(status_frame, text='Message', font=('Arial', 11)).grid(row=1, column=0, sticky='w', padx=10, pady=2)
        tk.Label(status_frame, textvariable=self.status_vars['message'], font=('Arial', 11), wraplength=280, justify='left', anchor='w').grid(row=1, column=1, columnspan=3, sticky='w', padx=5, pady=2)
        
        tk.Label(status_frame, text='Constraints', font=('Arial', 11)).grid(row=2, column=0, sticky='w', padx=10, pady=2)
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

        self.axs[0,0].set_xlabel('Iteration', fontsize=10)
        self.axs[0,1].set_xlabel('Iteration', fontsize=10)
        self.axs[1,0].set_xlabel('Iteration', fontsize=10)
        self.axs[1,1].set_xlabel('Iteration', fontsize=10)
        self.axs[2,0].set_xlabel('Iteration', fontsize=10)
        self.axs[2,1].set_xlabel('Iteration', fontsize=10)

    def clear_results(self, event=None):
        if self.status_vars['success'].get() not in ['-', 'Running...']:
            for var in self.opt_res_vars: var.set('num')
            for k in self.mass_vars: self.mass_vars[k].set('num')
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
                title='Save Module 1 Plot',
                defaultextension='.png',
                initialfile='module1_plots.png',
                filetypes=[('PNG Image', '*.png'), ('JPEG Image', '*.jpg;*.jpeg'), ('PDF File', '*.pdf'), ('All Files', '*.*')],
            )
            if not file_path:
                return
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo('Saved', f'Plot saved successfully to:\n{file_path}')
        except Exception as e:
            messagebox.showerror('Save Error', f'Could not save plot: {e}')

    def switch_to_module2(self):
        import subprocess
        import os
        try:
            module2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GUI_Phase2.py')
            subprocess.Popen([sys.executable, module2_path])
        except Exception as e:
            messagebox.showerror('Error', f'Could not launch Module 2 App: {e}')

    def start_optimization(self):
        self.run_btn.config(state=tk.DISABLED)
        for var in self.opt_res_vars: var.set('running...')
        for k in self.mass_vars: self.mass_vars[k].set('...')
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
        start_t = time.time()
        try:
            x0_norm = normalize(x_init_real)
            bounds_norm = [(0.0, 1.0) for _ in range(10)]
            
            history_norm = [x0_norm.copy()]
            opt_method = self.optimizer_var.get()

            if opt_method == 'trust-constr':
                def callback(xk, state=None):
                    history_norm.append(xk.copy())
            else:
                def callback(xk):
                    history_norm.append(xk.copy())

            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout

            options = {'maxiter': 1000, 'disp': False}
            if opt_method == 'SLSQP':
                options['ftol'] = 1e-4
            elif opt_method == 'trust-constr':
                options['xtol'] = 1e-4

            sol = minimize(
                objective,
                x0_norm,
                method=opt_method,
                bounds=bounds_norm,
                constraints=constraints,
                callback=callback,
                options=options,
            )
            
            x_opt_real = unnormalize(sol.x)
            r_opt = evaluate_all(x_opt_real)
            Vdive = r_opt['Vdive']

            ext_time = time.time() - start_t
            
            print('\n==================================')
            print('  Optimisation Result Details')
            print('==================================')
            for lbl, val in zip(self.var_keys, x_opt_real): print(f'  {lbl:>8s} = {val:12.4f}')
            print(f'\n  Success: {sol.success}\n  Message: {sol.message}')
            MC.print_mass_breakdown(x_opt_real[0], x_opt_real[1], x_opt_real[2], x_opt_real[3], x_opt_real[4], x_opt_real[5], x_opt_real[6], x_opt_real[9], Vdive)
            
            c_diffs = [
                ('C1: Fuel Balance', r_opt['M_fuel_FFM'] - r_opt['M_fuel_req'], 'eq'),
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
                ('C12: Drag Divergence Mach Number', r_opt['M_DD_wing'] - x_opt_real[7], 'ineq'),
                ('C13: Mass Closure', r_opt['Mtotal'] - x_opt_real[9], 'eq')
            ]

            print('\n--------------------------------------------------------------------------------')
            print('POST-OPTIMIZATION CONSTRAINT CHECK')
            print('================================================================================')
            
            satisfied = 0
            total_constraints = len(c_diffs)
            
            for name, val, ctype in c_diffs:
                if ctype == 'eq' and abs(val) <= 1e-1:
                    is_pass = True
                elif ctype == 'ineq' and val >= -1e-1:
                    is_pass = True
                else:
                    is_pass = False

                if is_pass:
                    satisfied += 1
                    symbol = '✓'
                else:
                    symbol = 'x'
                
                print(f'{symbol} {name:<35s} | Diff: {val:14.6f}')

            print('================================================================================')
            if satisfied == total_constraints:
                print(f'Summary: {satisfied}/{total_constraints} constraints satisfied | ✓ All passed!')
            else:
                print(f'Summary: {satisfied}/{total_constraints} constraints satisfied | x Partial failure')
            print('================================================================================')
            
            constraints_str = f'{satisfied} / {total_constraints}'

            sys.stdout = old_stdout
            out_file = 'Optimization_Module1_Results.txt'
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(new_stdout.getvalue())

            Mls = MC.M_lifting_surface(x_opt_real[0], x_opt_real[1], x_opt_real[2], x_opt_real[3], x_opt_real[4], x_opt_real[9], Vdive)
            Mpower = MC.M_powerplant(x_opt_real[5])
            Msys = MC.M_systems(x_opt_real[9])
            Mfuel = MC.M_fuel(x_opt_real[6], x_opt_real[9])
            Mfuse = MC.M_fuselage()
            Mpayload = MC.M_payload()
            Mop = MC.M_operational()
            Mtotal_calc = r_opt['Mtotal']

            self.after(0, self._update_gui_post_calc, sol.success, sol.message, sol.nit, ext_time, constraints_str,
                       x_opt_real, Mls, Mpower, Msys, Mfuel, Mfuse, Mpayload, Mop, Mtotal_calc, history_norm)

        except Exception as e:
            import traceback; traceback.print_exc(); self.after(0, self._handle_error, str(e))

    def _update_gui_post_calc(self, success, message, nit, ext_time, constraints_str, x_opt, Mls, Mpower, Msys, Mfuel, Mfuse, Mpayload, Mop, Mtotal, history_norm):
        self.status_vars['success'].set(str(success))
        self.status_vars['message'].set(str(message))
        self.status_vars['iterations'].set(str(nit))
        self.status_vars['time'].set(f'{ext_time:.2f} s')
        self.status_vars['constraints'].set(constraints_str)

        # Warn if the optimizer did not converge to a fully feasible point.
        fully_feasible = False
        try:
            left, right = constraints_str.split('/')
            fully_feasible = int(left.strip()) == int(right.strip())
        except Exception:
            fully_feasible = False

        if (not bool(success)) or (not fully_feasible):
            warn_msg = (
                'Optimization warning:\n'
                'Solver could not  converge to a fully feasible optimum.\n\n'
                f'Success: {success}\n'
                f'Constraints satisfied: {constraints_str}\n'
                f'Solver message: {message}'
            )
            self.status_vars['message'].set('WARNING: No reliable optimum found')
            messagebox.showwarning('Convergence Warning', warn_msg)

        for i, val in enumerate(x_opt):
            self.opt_res_vars[i].set(f'{val:.4f}')

        m_dict = {'Mls': Mls, 'Mpower': Mpower, 'Msys': Msys, 'Mfuel': Mfuel, 
                  'Mfuse': Mfuse, 'Mpayload': Mpayload, 'Mop': Mop, 'Mtotal': Mtotal}
        for k, v in m_dict.items():
            self.mass_vars[k].set(f'{v:.1f}')

        self._plot_history(history_norm)
        self.run_btn.config(state=tk.NORMAL)

    def _handle_error(self, err_msg):
        warn_msg = (
            'Optimization could not find a valid solution from the current initial guess.\n\n'
            f'Details: {err_msg}\n\n'
            'Try a different initial guess or optimizer and run again.'
        )
        messagebox.showwarning('Optimization Warning', warn_msg)
        self.status_vars['success'].set('No solution')
        self.status_vars['message'].set('WARNING: Solver failed to find optimum')
        self.status_vars['constraints'].set('...')
        self.run_btn.config(state=tk.NORMAL)

    def _plot_history(self, history_norm):
        history_phys = np.array([unnormalize(xn) for xn in history_norm])
        iters = np.arange(len(history_phys))

        self.fig.clf()
        self.axs = self.fig.subplots(3, 2)
        self.fig.subplots_adjust(hspace=0.6, wspace=0.55, top=0.94, bottom=0.08, left=0.08, right=0.92)

        ax1 = self.axs[0, 0]
        ax1_twin = ax1.twinx()
        ax1.plot(iters, history_phys[:, 0], 'b.-', label='AR', markersize=4, lw=1)
        ax1_twin.plot(iters, history_phys[:, 1], 'm.-', label='Sw', markersize=4, lw=1)
        ax1.set_ylabel('AR')
        ax1_twin.set_ylabel('Sw (m²)')
        ax1.set_xlabel('Iteration', fontsize=10)
        ax1.legend(loc='lower center', fontsize=8)
        ax1_twin.legend(loc='upper right', fontsize=8)
        ax1.grid(True, ls='--', alpha=0.5)

        ax2 = self.axs[0, 1]
        ax2_twin = ax2.twinx()
        ax2.plot(iters, history_phys[:, 2], 'g.-', label='Sweep', markersize=4, lw=1)
        ax2_twin.plot(iters, history_phys[:, 4], 'r.-', label='Taper', markersize=4, lw=1)
        ax2.set_ylabel('Sweep (deg)')
        ax2_twin.set_ylabel('Taper Ratio')
        ax2.set_xlabel('Iteration', fontsize=10)
        ax2.legend(loc='lower left', fontsize=8)
        ax2_twin.legend(loc='lower right', fontsize=8)
        ax2.grid(True, ls='--', alpha=0.5)

        ax3 = self.axs[1, 0]
        ax3_twin = ax3.twinx()
        ax3.plot(iters, history_phys[:, 3], 'c.-', label='t/c', markersize=4, lw=1)
        ax3_twin.plot(iters, history_phys[:, 6], 'k.-', label='FMf', markersize=4, lw=1)
        ax3.set_ylabel('t/c')
        ax3_twin.set_ylabel('FMf')
        ax3.set_xlabel('Iteration', fontsize=10)
        ax3.legend(loc='lower left', fontsize=8)
        ax3_twin.legend(loc='lower right', fontsize=8)
        ax3.grid(True, ls='--', alpha=0.5)

        ax4 = self.axs[1, 1]
        ax4_twin = ax4.twinx()
        ax4.plot(iters, history_phys[:, 8], 'b.-', label='Altitude', markersize=4, lw=1)
        ax4_twin.plot(iters, history_phys[:, 7], 'm.-', label='Mach', markersize=4, lw=1)
        ax4.set_ylabel('Altitude (m)')
        ax4_twin.set_ylabel('Mach')
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.legend(loc='lower left', fontsize=8)
        ax4_twin.legend(loc='lower right', fontsize=8)
        ax4.grid(True, ls='--', alpha=0.5)

        ax5 = self.axs[2, 0]
        ax5.plot(iters, history_phys[:, 5], 'r.-', label='T0', markersize=4, lw=1)
        ax5.set_ylabel('Sea Level Thrust (N)')
        ax5.set_xlabel('Iteration', fontsize=10)
        ax5.legend(loc='lower right', fontsize=8)
        
        ax6 = self.axs[2, 1]
        ax6.plot(iters, history_phys[:, 9], 'g.-', label='Takeoff mass', markersize=4, lw=1)
        ax6.set_ylabel('Takeoff mass (kg)')
        ax6.set_xlabel('Iteration', fontsize=10)
        ax6.legend(loc='lower right', fontsize=8)

        self.canvas.draw()

if __name__ == '__main__':
    app = OptimizationGUI()
    app.mainloop()
