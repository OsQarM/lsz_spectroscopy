import numpy as np
import qutip as qt


class LSZ_experiment():

    def __init__(self, n_qubits, epsilon, H_target_dictionary, ramp_time, wait_time, dt,
                 ramp_noise=False, wait_noise=False, gamma_dec_list=None, gamma_dep_list=None):
        '''
        Params:
        epsilon: Coefficient of X Hamiltonian
        alpha: Coefficient of target Hamiltonian
        total_time: Total time of the algorithm
        wait_time: Waiting time in the middle of the LSZ algorithm (0 if just doing LZ)
        max_s: maximum of s parameter. Will control slope of schedule
        '''

        self.n_qubits = n_qubits
        self.e = epsilon
        self.Ht_dict = H_target_dictionary

        self.to = 0
        self.tw = wait_time
        self.tr = ramp_time
        self.dt = dt

        self.tf = 2*self.tr + self.tw
        self.slope = 1/self.tr

        self.rnoise = ramp_noise
        self.wnoise = wait_noise
        self.dec_rates = gamma_dec_list
        self.dep_rates = gamma_dep_list

        self.npsx, self.npsy, self.npsz = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
        self.qtsx, self.qtsy, self.qtsz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

        self.ket0 = qt.basis(2, 0)
        self.ket1 = qt.basis(2, 1)

        self.qtsx_list, self.qtsy_list, self.qtsz_list = self.initialize_qt_operators()
        self.npsx_list, self.npsy_list, self.npsz_list = self.initialize_np_operators()

        self.H0_numpy = self.build_H0_numpy()
        self.H0_qutip = self.build_H0_qutip()

        self.Ht_numpy = self.build_Ht_numpy()
        self.Ht_qutip = self.build_Ht_qutip()

        self.dep_ops, self.dec_ops = None, None
        if self.rnoise or self.wnoise:
            self.dep_ops, self.dec_ops = self.build_noise_operators()

    def kron_n(self, ops):
        out = ops[0]
        for A in ops[1:]:
            out = np.kron(out, A)
        return out

    def build_H0_numpy(self):
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        for i in range(self.n_qubits):
            H += -self.e*self.npsx_list[i]
        return H

    def build_H0_qutip(self):
        H = 0
        for i in range(self.n_qubits):
            H += -self.e*self.qtsx_list[i]
        return H

    def build_Ht_numpy(self):
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        local_z_terms = self.Ht_dict['one_body']

        for i, weight in enumerate(local_z_terms):
            H += weight*self.npsz_list[i]

        if self.n_qubits > 1:
            zz_terms = self.Ht_dict['two_body']
            k = 0
            for i in range(self.n_qubits-1):
                for j in range(i+1, self.n_qubits):
                    H += zz_terms[k] * self.npsz_list[i] @ self.npsz_list[j]
                    k += 1

        return H

    def build_Ht_qutip(self):
        H = 0
        local_z_terms = self.Ht_dict['one_body']

        for i, weight in enumerate(local_z_terms):
            H += weight*self.qtsz_list[i]

        if self.n_qubits > 1:
            zz_terms = self.Ht_dict['two_body']
            k = 0
            for i in range(self.n_qubits-1):
                for j in range(i+1, self.n_qubits):
                    H += zz_terms[k] * self.qtsz_list[i] * self.qtsz_list[j]
                    k += 1

        return H

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

    def trapezoid_s(self, t):
        return np.minimum(1, self.slope * np.minimum(t, self.tf - t))

    def H_numpy(self, t):
        s = self.trapezoid_s(t)
        return (1-s)*self.H0_numpy + s*self.Ht_numpy

    def H_qutip(self):
        H = [[self.H0_qutip, lambda t, args: 1 - self.trapezoid_s(t)],
             [self.Ht_qutip, lambda t, args: self.trapezoid_s(t)]]
        return H

    def show_schedule(self, n_steps):
        t_list = np.linspace(self.to, self.tf, n_steps)
        s_list = np.zeros(len(t_list))

        for i, t in enumerate(t_list):
            s_list[i] = self.trapezoid_s(t)

        return t_list, s_list

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

        psi0 = qt.tensor([1/np.sqrt(2)*(self.ket0 + self.ket1)]*self.n_qubits)

        H = self.H_qutip()

        dep_ops = self.dep_ops or []
        dec_ops = self.dec_ops or []
        ramp_c_ops = dep_ops + dec_ops if self.rnoise else []
        wait_c_ops = dep_ops + dec_ops if self.wnoise else []

        if self.rnoise:
            sim_r1 = qt.mesolve(H, psi0, t_list_r1, c_ops=ramp_c_ops)
        else:
            sim_r1 = qt.sesolve(H, psi0, t_list_r1)

        if self.wnoise:
            sim_w = qt.mesolve(H, sim_r1.states[-1], t_list_w, c_ops=wait_c_ops)
        else:
            sim_w = qt.sesolve(H, sim_r1.states[-1], t_list_w)

        downstream_noisy = self.wnoise or self.rnoise
        if self.rnoise:
            sim_r2 = qt.mesolve(H, sim_w.states[-1], t_list_r2, c_ops=ramp_c_ops)
        elif downstream_noisy:
            sim_r2 = qt.mesolve(H, sim_w.states[-1], t_list_r2, c_ops=[])
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


def run_experiment_sweep(nqubits, epsilon, H_target_dict, ramp_time, tw_l, dt,
                         r_noise=False, w_noise=False,
                         gamma_dec_list=None, gamma_dep_list=None):
    """Run LSZ experiment over a list of wait times, returning P(ground) vs tw."""
    pc_list = []
    for wait_time in tw_l:
        experiment = LSZ_experiment(nqubits, epsilon, H_target_dict, ramp_time, wait_time, dt,
                                    ramp_noise=r_noise, wait_noise=w_noise,
                                    gamma_dec_list=gamma_dec_list, gamma_dep_list=gamma_dep_list)
        _, _, ramp_2_sim = experiment.time_evolution()
        populations = experiment.calculate_populations(ramp_2_sim)
        pc_list.append(np.real(populations[0]))
    return np.array(pc_list)
