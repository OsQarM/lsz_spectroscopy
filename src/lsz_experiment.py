import numpy as np
import qutip as qt


# Module-level generator. Call set_global_seed(seed) once at the start of a
# sweep to make every subsequent LSZ_experiment draw reproducible (but distinct)
# noise. Each constructor pulls from this generator unless the user passes
# their own `rng`.
_GLOBAL_RNG = np.random.default_rng()


def set_global_seed(seed):
    """Seed the module-level RNG used by LSZ_experiment for ramp/wait errors.

    Call this once before a sweep. Each experiment instance will then draw a
    different noise realization, but the whole sequence is reproducible.
    """
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)


class LSZ_experiment():

    def __init__(self, n_qubits, epsilon, H_target_dictionary, ramp_time, wait_time, dt, up_assymetry_factors = None,
                 down_assymetry_factors=None,
                 ramp_noise=False, wait_noise=False, gamma_dec_list=None, gamma_dep_list=None,
                 initial_state=None, ramp_error = 0.0, wait_error = 0.0, rng=None,
                 noise_seed=None):
        '''
        Params:
        n_qubits: number of qubits
        epsilon: Coefficient of X Hamiltonian
        H_target_dictionary: coefficients of target Hamiltonian. Keys:
            'one_body'  -> list of local Z weights (length n_qubits)
            'local_x'   -> list of local X weights (length n_qubits) added to target H
            'two_body'  -> list of ZZ couplings (length n_qubits*(n_qubits-1)/2)
        ramp_time: time taken to ramp up
        wait_time: Waiting time in the middle of the LSZ algorithm (0 if just doing LZ)
        dt: time step
        assymetry_factors: list of relative ramp speeds for Z component of qubits
        initial_state: qutip ket used to initialize the time evolution and the diagnostics.
            If None, defaults to the tensor product of |-> states.
        '''
        
        #General parameters and Hamiltonian definition
        self.n_qubits = n_qubits
        self.e = epsilon
        self.Ht_dict = H_target_dictionary

        #Schedule times and timestep size
        self.to = 0
        self.tw = wait_time
        self.tr = ramp_time
        self.dt = dt

        self.tf = 2*self.tr + self.tw
        self.slope = 1/self.tr

        #T1 and T2 noise parameters
        self.rnoise = ramp_noise
        self.wnoise = wait_noise
        self.dec_rates = gamma_dec_list
        self.dep_rates = gamma_dep_list

        #Ramp asymetry
        if up_assymetry_factors == None:
            self.up_assymetry_list = np.ones(n_qubits)
        else:
            self.up_assymetry_list = up_assymetry_factors

        if down_assymetry_factors is None:
            self.down_assymetry_list = np.ones(n_qubits)
        else:
            self.down_assymetry_list = down_assymetry_factors

        #Random ramp/wait errors. Each LSZ_experiment instance pre-draws its
        #own Gaussian noise grid so that H(t) stays a deterministic function
        #for qutip's ODE solver (different bin -> different sample).
        self.ramp_e = ramp_error
        self.wait_e = wait_error

        # If `noise_seed` is given, this experiment re-seeds its own generator
        # from that fixed value, so the drawn noise grid is identical every time
        # the class is instantiated with the same seed (e.g. across a sweep,
        # giving the same schedule error at every wait time). `noise_seed` takes
        # precedence over `rng`.
        if noise_seed is not None:
            rng = np.random.default_rng(noise_seed)
        elif rng is None:
            rng = _GLOBAL_RNG
        elif not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # Per-qubit sample per dt-bin for the slope multiplier and the
        # wait-plateau amplitude. We over-allocate by one bin so the lookup
        # at t == tf lands safely in range.
        n_bins_total = int(np.ceil(self.tf / self.dt)) + 1
        if self.ramp_e > 0:
            self.ramp_noise_grid = rng.standard_normal((n_bins_total, n_qubits))
        else:
            self.ramp_noise_grid = np.zeros((n_bins_total, n_qubits))

        if self.wait_e > 0:
            self.wait_noise_grid = rng.standard_normal((n_bins_total, n_qubits))
        else:
            self.wait_noise_grid = np.zeros((n_bins_total, n_qubits))


        #Initialize operators
        self.npsx, self.npsy, self.npsz = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
        self.qtsx, self.qtsy, self.qtsz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

        self.ket0 = qt.basis(2, 0)
        self.ket1 = qt.basis(2, 1)

        self.qtsx_list, self.qtsy_list, self.qtsz_list = self.initialize_qt_operators()
        self.npsx_list, self.npsy_list, self.npsz_list = self.initialize_np_operators()
        
        #Build Hamiltonians
        self.H0_numpy = self.build_H0_numpy()
        self.H0_qutip = self.build_H0_qutip()

        self.Ht_numpy_common = self.build_Ht_numpy_common()
        self.Ht_qutip_common = self.build_Ht_qutip_common()
        
        self.Hz_numpy_list = self.build_Hz_numpy_list()
        self.Hz_qutip_list = self.build_Hz_qutip_list()

        #Create noise operators
        self.dep_ops, self.dec_ops = None, None
        if self.rnoise or self.wnoise:
            self.dep_ops, self.dec_ops = self.build_noise_operators()
        
        #Custom initial state
        if initial_state is None:
            self.initial_state = qt.tensor([1/np.sqrt(2)*(self.ket0 - self.ket1)]*self.n_qubits)
        else:
            self.initial_state = initial_state

    def kron_n(self, ops):
        out = ops[0]
        for A in ops[1:]:
            out = np.kron(out, A)
        return out

    def build_H0_numpy(self):
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        for i in range(self.n_qubits):
            H += self.e*self.npsx_list[i]
        return H

    def build_H0_qutip(self):
        H = 0
        for i in range(self.n_qubits):
            H += self.e*self.qtsx_list[i]
        return H

    def build_Ht_numpy_common(self):
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)

        if self.n_qubits > 1:
            zz_terms = self.Ht_dict['two_body']
            k = 0
            for i in range(self.n_qubits-1):
                for j in range(i+1, self.n_qubits):
                    H += zz_terms[k] * self.npsz_list[i] @ self.npsz_list[j]
                    k += 1

        local_x_terms = self.Ht_dict.get('local_x', np.zeros(self.n_qubits))
        for i, weight in enumerate(local_x_terms):
            H += weight * self.npsx_list[i]

        return H

    def build_Ht_qutip_common(self):
        H = 0
        if self.n_qubits > 1:
            zz_terms = self.Ht_dict['two_body']
            k = 0
            for i in range(self.n_qubits-1):
                for j in range(i+1, self.n_qubits):
                    H += zz_terms[k] * self.qtsz_list[i] * self.qtsz_list[j]
                    k += 1

        local_x_terms = self.Ht_dict.get('local_x', np.zeros(self.n_qubits))
        for i, weight in enumerate(local_x_terms):
            H += weight * self.qtsx_list[i]

        return H
    
    def build_Hz_numpy_list(self):
        H_list = []

        local_z_terms = self.Ht_dict['local_z']
        for i, weight in enumerate(local_z_terms):
            H_list.append(weight*self.npsz_list[i])
        
        return H_list
    
    def build_Hz_qutip_list(self):
        H_list = []

        local_z_terms = self.Ht_dict['local_z']
        for i, weight in enumerate(local_z_terms):
            H_list.append(weight*self.qtsz_list[i])

        return H_list

    def initialize_np_operators(self):
        sx_list, sy_list, sz_list = [], [], []
        sx, sy, sz = self.npsx, self.npsy, self.npsz
        for i in range(self.n_qubits):
            op_list = [np.eye(2) for _ in range(self.n_qubits)]
            op_list[i] = sx
            sx_list.append(self.kron_n(op_list))
            op_list[i] = sy
            sy_list.append(self.kron_n(op_list))
            op_list[i] = sz
            sz_list.append(self.kron_n(op_list))

        return sx_list, sy_list, sz_list

    def initialize_qt_operators(self):
        sx_list, sy_list, sz_list = [], [], []
        sx, sy, sz = self.qtsx, self.qtsy, self.qtsz
        for i in range(self.n_qubits):
            op_list = [qt.qeye(2)] * self.n_qubits
            op_list[i] = sx
            sx_list.append(qt.tensor(op_list))
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        return sx_list, sy_list, sz_list

    def build_noise_operators(self):
        if self.dep_rates:
            dep_ops = []
            for i in range(self.n_qubits):
                dep_ops.append(np.sqrt(self.dep_rates[i])*self.qtsz_list[i])
        else:
            dep_ops = None

        if self.dec_rates:
            dec_ops = []
            for i in range(self.n_qubits):
                op_list = [qt.qeye(2)] * self.n_qubits
                op_list[i] = qt.sigmam()
                dec_ops.append(np.sqrt(self.dec_rates[i])*qt.tensor(op_list))
        else:
            dec_ops = None

        return dep_ops, dec_ops

    def trapezoid(self, t, up_assym=1, down_assym=1, qubit_idx=None):
        # Linearly interpolate the noise between adjacent bins. A piecewise-
        # constant lookup (one flat value per dt-bin) makes H(t) a staircase
        # with a jump at every bin boundary; qutip's adaptive ODE solver then
        # rejects and shrinks its step at each discontinuity, thrashing down to
        # sub-dt steps. Interpolating keeps H(t) continuous (same random samples,
        # just connected) so the solver takes large steps again.
        n_bins = self.ramp_noise_grid.shape[0]
        x = np.clip(t / self.dt, 0, n_bins - 1)
        lo = int(np.floor(x))
        hi = min(lo + 1, n_bins - 1)
        frac = x - lo

        #Generate errors
        if qubit_idx is not None:
            r_lo = self.ramp_noise_grid[lo, qubit_idx]
            r_hi = self.ramp_noise_grid[hi, qubit_idx]
            w_lo = self.wait_noise_grid[lo, qubit_idx]
            w_hi = self.wait_noise_grid[hi, qubit_idx]
            slope_offset = self.ramp_e * (r_lo + frac * (r_hi - r_lo))
            cap = 1.0 + self.wait_e * (w_lo + frac * (w_hi - w_lo))
        else:
            slope_offset = 0.0
            cap = 1.0

        # Endpoints pinned to 0 for all ramps 
        if t <= self.to or t >= self.tf:
            return 0.0

        #Return 1 (+ error) during wait time
        if self.tr <= t <= self.tr + self.tw:
            return cap

        # Return trapezoid (rising or falling + error). The asymmetry factor
        # makes the ramp steeper so it reaches 1 *earlier*; clamping the clean
        # ramp to 1 keeps a fast ramp from overshooting past 1 and holds it at
        # the top until the wait window. The slope error is added *after* the
        # clamp so the noise -- unlike the asymmetry -- is still free to push
        # the value above 1.
        if t < self.tr:
            return np.minimum(1.0, up_assym * self.slope * t) + slope_offset
        else:
            return np.minimum(1.0, down_assym * self.slope * (self.tf - t)) + slope_offset
    
    # def trapezoid(self, t, assym=1):
    #     return np.minimum(1, assym*self.slope * np.minimum(t, self.tf - t))

    def H_numpy(self, t):
        s = self.trapezoid(t)
        #Add evolution of H0 and interaction+transverse terms
        H = (1-s)*self.H0_numpy + s*self.Ht_numpy_common
        #Add local Z fields with assymetries (per-qubit wait-plateau noise)
        for i, Hi in enumerate(self.Hz_numpy_list):
            H += self.trapezoid(t, up_assym=self.up_assymetry_list[i],
                                down_assym=self.down_assymetry_list[i],
                                qubit_idx=i) * Hi
        return H

    def H_qutip(self):
        #Add evolution of H0 and interaction+transverse terms
        H = [[self.H0_qutip, lambda t: 1 - self.trapezoid(t)],
            [self.Ht_qutip_common, lambda t: self.trapezoid(t)]]
        #Add local Z fields with assymetries (per-qubit wait-plateau noise)
        H += [[Hi, lambda t, i=i: self.trapezoid(t,
                                                 up_assym=self.up_assymetry_list[i],
                                                 down_assym=self.down_assymetry_list[i],
                                                 qubit_idx=i)]
            for i, Hi in enumerate(self.Hz_qutip_list)]
        return H

    def show_schedule(self, n_steps):
        t_list = np.linspace(self.to, self.tf, n_steps)
        common_schedule = np.array([self.trapezoid(t) for t in t_list])
        hz_schedules = [
            np.array([self.trapezoid(t, up_assym=self.up_assymetry_list[i],
                                     down_assym=self.down_assymetry_list[i],
                                     qubit_idx=i) for t in t_list])
            for i in range(len(self.Hz_numpy_list))
        ]
        return t_list, common_schedule, hz_schedules


    def show_spectrum(self, n_steps):
        t_list = np.linspace(self.to, self.tf, n_steps)
        eg_val_list = []
        eg_vec_list = []

        for i, t in enumerate(t_list):
            H = self.H_numpy(t)
            egvals, egvecs = np.linalg.eigh(H)
            eg_val_list.append(egvals)
            eg_vec_list.append(egvecs)

        return t_list, eg_val_list, eg_vec_list

    def time_evolution(self):
        t_list_r1 = np.linspace(self.to, self.tr, max(100, int(self.tr / self.dt)))
        t_list_w = np.linspace(self.tr, self.tr + self.tw, max(100, int(self.tw / self.dt)))
        t_list_r2 = np.linspace(self.tr + self.tw, self.tf, max(100, int(self.tr / self.dt)))

        psi0 = self.initial_state

        H = self.H_qutip()

        dep_ops = self.dep_ops or []
        dec_ops = self.dec_ops or []
        ramp_c_ops = dep_ops + dec_ops if self.rnoise else []
        wait_c_ops = dep_ops + dec_ops if self.wnoise else []

        # Once any earlier segment is solved with mesolve its output is a
        # density matrix, which sesolve cannot accept. So a segment must use
        # mesolve if it has its own noise OR if any preceding segment did.
        if self.rnoise:
            sim_r1 = qt.mesolve(H, psi0, t_list_r1, c_ops=ramp_c_ops)
        else:
            sim_r1 = qt.sesolve(H, psi0, t_list_r1)

        upstream_noisy = self.rnoise
        if self.wnoise or upstream_noisy:
            sim_w = qt.mesolve(H, sim_r1.states[-1], t_list_w, c_ops=wait_c_ops)
        else:
            sim_w = qt.sesolve(H, sim_r1.states[-1], t_list_w)

        upstream_noisy = self.rnoise or self.wnoise
        if self.rnoise or upstream_noisy:
            sim_r2 = qt.mesolve(H, sim_w.states[-1], t_list_r2, c_ops=ramp_c_ops)
        else:
            sim_r2 = qt.sesolve(H, sim_w.states[-1], t_list_r2)

        return sim_r1, sim_w, sim_r2

    def calculate_populations(self, sim_res):
        H_f = self.H_numpy(self.tf)
        eigvals_f, eigvecs_f = np.linalg.eigh(H_f)

        n_states = 2**self.n_qubits
        populations = np.zeros(n_states)
        final = sim_res.states[-1]

        if final.type == 'ket':
            final_state = final.full().flatten()
            for k in range(n_states):
                overlap = np.dot(eigvecs_f[:, k].conj(), final_state)
                populations[k] = np.real(overlap * overlap.conj())
        else:
            rho = final.full()
            for k in range(n_states):
                v = eigvecs_f[:, k].reshape(-1, 1)
                populations[k] = np.real(v.conj().T @ rho @ v).item()

        return populations

    def calculate_eigenbasis_populations(self, sim_res):
        """Populations in the eigenbasis of H(tr) (the post-quench target H)
        at every timestep.

        Returns array of shape (n_timesteps, 2**n_qubits).
        """
        H_target = self.H_numpy(self.tr)
        _, eigvecs = np.linalg.eigh(H_target)

        n_states = 2**self.n_qubits
        n_steps = len(sim_res.states)
        pops = np.zeros((n_steps, n_states))

        for s, state in enumerate(sim_res.states):
            if state.type == 'ket':
                psi = state.full().flatten()
                overlaps = eigvecs.conj().T @ psi
                pops[s, :] = np.real(overlaps * overlaps.conj())
            else:
                rho = state.full()
                M = eigvecs.conj().T @ rho @ eigvecs
                pops[s, :] = np.real(np.diag(M))

        return pops


def run_experiment_sweep(nqubits, epsilon, H_target_dict, ramp_time, tw_l, dt, up_assym_list = None,
                         down_assym_list=None,
                         r_noise=False, w_noise=False,
                         gamma_dec_list=None, gamma_dep_list=None, initial_state=None,
                         ramp_error=0.0, wait_error=0.0, rng=None,
                         noise_seed=None,
                         show_progress=True):
    """Run LSZ experiment over a list of wait times, returning P(ground) vs tw.

    `ramp_error` / `wait_error` are Gaussian noise sizes on the ramp slope and
    wait-plateau amplitude. Each experiment instance draws its own realization
    from the module-level RNG (call `set_global_seed(seed)` once before the
    sweep for reproducibility) or from a user-supplied `rng`.

    Pass `noise_seed` (any fixed int) to hold the schedule error *identical*
    across every wait time in the sweep: each experiment re-seeds its own
    generator from that value, so the same noise grid is drawn every iteration.
    `noise_seed` takes precedence over `rng`.

    Progress is reported weighted by each run's total simulated time
    (2*ramp_time + tw), not by iteration count, so the percentage tracks
    actual wall-clock work rather than how many wait-times have been done.
    """
    import sys
    import time

    if rng is not None and not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    tw_arr = np.asarray(tw_l, dtype=float)
    weights = 2 * ramp_time + tw_arr           # simulated-time per iteration
    total_work = float(weights.sum())
    done_work = 0.0
    t_start = time.perf_counter()

    pc_list = []
    for i, wait_time in enumerate(tw_arr):
        experiment = LSZ_experiment(nqubits, epsilon, H_target_dict, ramp_time, float(wait_time), dt, up_assym_list,
                                    down_assymetry_factors=down_assym_list,
                                    ramp_noise=r_noise, wait_noise=w_noise,
                                    gamma_dec_list=gamma_dec_list, gamma_dep_list=gamma_dep_list,
                                    initial_state=initial_state,
                                    ramp_error=ramp_error, wait_error=wait_error, rng=rng,
                                    noise_seed=noise_seed)
        _, _, ramp_2_sim = experiment.time_evolution()

        #Population of GS of H0
        # populations = experiment.calculate_populations(ramp_2_sim)
        # pc_list.append(np.real(populations[0]))

        #Population of Psi0
        psi_init = experiment.initial_state.full().flatten()
        final = ramp_2_sim.states[-1]
        if final.type == 'ket':
            psi_f = final.full().flatten()
            pc = np.abs(np.vdot(psi_init, psi_f))**2
        else:
            rho = final.full()
            pc = np.real(np.vdot(psi_init, rho @ psi_init))
        pc_list.append(float(pc))

        done_work += weights[i]
        if show_progress:
            frac = done_work / total_work if total_work > 0 else 1.0
            elapsed = time.perf_counter() - t_start
            eta = elapsed * (1.0 - frac) / frac if frac > 0 else 0.0
            bar_len = 30
            filled = int(bar_len * frac)
            bar = '#' * filled + '-' * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {100*frac:6.2f}%  "
                             f"iter {i+1}/{len(tw_arr)}  "
                             f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s")
            sys.stdout.flush()

    if show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return np.array(pc_list)
