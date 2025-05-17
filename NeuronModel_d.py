import numpy as np
from scipy.integrate import odeint


class NeuronModel:
    def __init__(self, bifurcation_type='saddle-node', C_m=1.0, g_Na=None, g_K=None, g_L=None,
                 E_Na=60.0, E_K=-90.0, E_L=None, V_mid_m=None, V_mid_n=None, k_m=None, k_n=5.0):
        """
        Simplified Hodgkin-Huxley model with two state variables (persistent I_Na + I_K).

        Parameters from Izhikevich, E. M. (2007). Dynamical systems in neuroscience.

        Parameters:
        -----------
        bifurcation_type : str
            Type of bifurcation ('saddle-node', 'SNIC', 'subcritical_Hopf', 'supercritical_Hopf')
        C_m : float
            Membrane capacitance (μF/cm²)
        g_Na : float
            Sodium conductance (mS/cm²)
        g_K : float
            Potassium conductance (mS/cm²)
        g_L : float
            Leak conductance (mS/cm²)
        E_Na : float
            Sodium reversal potential (mV)
        E_K : float
            Potassium reversal potential (mV)
        E_L : float
            Leak reversal potential (mV)
        V_mid_m : float
            Sodium activation midpoint (mV)
        V_mid_n : float
            Potassium activation midpoint (mV)
        k_m : float
            Sodium activation slope
        k_n : float
            Potassium activation slope
        """
        self.bifurcation_type = bifurcation_type
        assert bifurcation_type in ['saddle-node', 'SNIC', 'subcritical_Hopf', 'supercritical_Hopf']

        # Set potassium threshold based on bifurcation type
        self.potassium_threshold = 'high' if bifurcation_type in ['saddle-node', 'SNIC'] else 'low'

        # Membrane capacitance
        self.C_m = C_m

        # Set conductances (use provided values or defaults based on bifurcation type)
        if bifurcation_type == 'subcritical_Hopf':
            self.g_Na = g_Na if g_Na is not None else 4.0
            self.g_K = g_K if g_K is not None else 4.0
            self.g_L = g_L if g_L is not None else 1.0
        else:
            self.g_Na = g_Na if g_Na is not None else 20.0
            self.g_K = g_K if g_K is not None else 10.0
            self.g_L = g_L if g_L is not None else 8.0

        # Reversal potentials
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L if E_L is not None else (-80.0 if self.potassium_threshold == 'high' else -78.0)

        # Kinetic parameters for gating variables
        if bifurcation_type == 'subcritical_Hopf':
            self.V_mid_m = V_mid_m if V_mid_m is not None else -30.0
            self.k_m = k_m if k_m is not None else 7.0
        else:
            self.V_mid_m = V_mid_m if V_mid_m is not None else -20.0
            self.k_m = k_m if k_m is not None else 15.0

        self.V_mid_n = V_mid_n if V_mid_n is not None else (-25.0 if self.potassium_threshold == 'high' else -45.0)
        self.k_n = k_n

    def m_inf(self, V):
        """Steady-state value of m (sodium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_m) / self.k_m))

    def n_inf(self, V):
        """Steady-state value of n (potassium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_n) / self.k_n))

    def tau_n(self, V):
        if self.bifurcation_type == 'SNIC':
            return 1
        if self.bifurcation_type == 'saddle-node':
            return 0.152
        if self.bifurcation_type == 'subcritical_Hopf':
            return 1
        if self.bifurcation_type == 'supercritical_Hopf':
            return 1

    def I_Na(self, V):
        """Sodium current"""
        return self.g_Na * self.m_inf(V) * (V - self.E_Na)

    def I_K(self, V, n):
        """Potassium current"""
        return self.g_K * n * (V - self.E_K)

    def I_L(self, V):
        """Leak current"""
        return self.g_L * (V - self.E_L)

    @staticmethod
    def create_step_current(t, step_time, step_duration, baseline, amplitude):
        """
        Create a step current waveform
        """
        I = np.ones_like(t) * baseline
        step_mask = (t >= step_time) & (t < step_time + step_duration)
        I[step_mask] = baseline + amplitude
        return I

    @staticmethod
    def create_ramp_current(t, ramp_start, ramp_duration, baseline, final_amplitude):
        """
        Create a linear ramp current
        """
        I = np.ones_like(t) * baseline
        ramp_mask = (t >= ramp_start) & (t < ramp_start + ramp_duration)
        ramp_t = t[ramp_mask] - ramp_start
        I[ramp_mask] = baseline + (final_amplitude - baseline) * (ramp_t / ramp_duration)
        I[t >= ramp_start + ramp_duration] = final_amplitude
        return I

    def find_equlibrium_points(self, I_ext, x_range, num_points=50000):
        """
        Find and analyze equilibrium points of the neuron model.
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        v_null = self.V_nullcline(x, I_ext)
        n_null = self.n_nullcline(x)

        diff = v_null - n_null
        sign_changes = np.where(np.diff(np.signbit(diff)))[0]

        equilibria = []

        for idx in sign_changes:
            x_intersect = np.interp(0, [diff[idx], diff[idx + 1]], [x[idx], x[idx + 1]])
            y_intersect = np.interp(x_intersect, [x[idx], x[idx + 1]], [v_null[idx], v_null[idx + 1]])

            eps = 1e-8
            V, n = x_intersect, y_intersect

            dV1, dn1 = self.dALLdt([V + eps, n], 0, lambda t: I_ext)
            dV2, dn2 = self.dALLdt([V - eps, n], 0, lambda t: I_ext)
            dV3, dn3 = self.dALLdt([V, n + eps], 0, lambda t: I_ext)
            dV4, dn4 = self.dALLdt([V, n - eps], 0, lambda t: I_ext)

            J = np.array([
                [(dV1 - dV2) / (2 * eps), (dV3 - dV4) / (2 * eps)],
                [(dn1 - dn2) / (2 * eps), (dn3 - dn4) / (2 * eps)]
            ])

            eigenvals = np.linalg.eigvals(J)

            if np.all(np.real(eigenvals) < 0):
                stability = 'stable'
            elif np.all(np.real(eigenvals) > 0):
                stability = 'unstable'
            else:
                stability = 'saddle'

            equilibria.append({
                'point': (V, n),
                'stability': stability,
                'eigenvalues': eigenvals,
                'jacobian': J
            })

        return equilibria

    def dALLdt(self, X, t, I_ext_t):
        """
        Calculate derivatives for the two state variables
        """
        V, n = X

        if callable(I_ext_t):
            I = I_ext_t(t)
        else:
            idx = int(t / self.dt) if hasattr(self, 'dt') else 0
            I = I_ext_t[min(idx, len(I_ext_t) - 1)]

        dVdt = (I - self.I_Na(V) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dndt = (self.n_inf(V) - n) / self.tau_n(V)

        return [dVdt, dndt]

    def dALLdt_backwards(self, X, t, I_ext_t):
        return [-x for x in self.dALLdt(X, t, I_ext_t)]

    def V_nullcline(self, V, I_ext):
        """V nullcline"""
        return (I_ext - self.I_Na(V) - self.I_L(V)) / (self.g_K * (V - self.E_K))

    def n_nullcline(self, V):
        """n nullcline"""
        return self.n_inf(V)

    def simulate(self, T, dt, X0, I_ext):
        """
        Basic simulation without perturbations or special pulse handling
        """
        self.dt = dt
        t = np.arange(0, T, dt)

        if isinstance(I_ext, (np.ndarray, list)) and len(I_ext) != len(t):
            raise ValueError("If I_ext is an array, it must have the same length as time points")

        solution = odeint(self.dALLdt, X0, t, args=(I_ext,))
        return t, solution

    def simulate_with_perturbations(self, T, dt, X0, I_ext, perturbations, smoothing_points=10):
        """
        Simulate with perturbations in both voltage and gating variable
        """
        perturbations = sorted(perturbations, key=lambda x: x[0])
        t_base = np.arange(0, T, dt)
        t_dense = list(t_base)

        if smoothing_points > 0:
            for pert_time, _, _ in perturbations:
                if pert_time >= T:
                    continue
                eps = dt / smoothing_points
                t_dense.extend([pert_time + i * eps for i in range(-smoothing_points, smoothing_points + 1)])

        t_dense = sorted(set(t_dense))
        t_dense = np.array(t_dense)

        n_points = len(t_dense)
        solution = np.zeros((n_points, 2))
        current_state = np.array(X0)
        current_idx = 0

        for pert_time, delta_V, delta_N in perturbations:
            if pert_time >= T:
                break
            next_idx = np.searchsorted(t_dense, pert_time)
            segment_t = t_dense[current_idx:next_idx + 1]

            if len(segment_t) > 0:
                segment_solution = odeint(self.dALLdt, current_state, segment_t, args=(I_ext,))
                solution[current_idx:next_idx + 1] = segment_solution
                current_state = segment_solution[-1].copy()

            current_state[0] += delta_V
            if delta_N is not None:
                current_state[1] += delta_N

            if smoothing_points > 0:
                smooth_idx_start = next_idx - smoothing_points
                smooth_idx_end = next_idx + smoothing_points + 1
                if smooth_idx_start >= 0 and smooth_idx_end < len(t_dense):
                    smooth_t = np.linspace(-3, 3, 2 * smoothing_points + 1)
                    sigmoid = 1 / (1 + np.exp(-smooth_t))
                    pre_state = solution[smooth_idx_start]
                    post_state = current_state
                    for i, s in enumerate(sigmoid):
                        idx = smooth_idx_start + i
                        solution[idx] = pre_state + s * (post_state - pre_state)

            current_idx = next_idx + 1

        if current_idx < n_points:
            final_t = t_dense[current_idx:]
            final_solution = odeint(self.dALLdt, current_state, final_t, args=(I_ext,))
            solution[current_idx:] = final_solution

        if smoothing_points > 0:
            solution_interp = np.array([
                np.interp(t_base, t_dense, solution[:, i])
                for i in range(2)
            ]).T
            return t_base, solution_interp

        return t_dense, solution

    def create_pulse_train(self, t, pulse_times, pulse_width, baseline, amplitude):
        """
        Create a train of short current pulses
        """

        def I(t_eval):
            if isinstance(t_eval, (list, np.ndarray)):
                result = np.ones_like(t_eval) * baseline
                for pulse_time in pulse_times:
                    mask = (t_eval >= pulse_time) & (t_eval <= pulse_time + pulse_width)
                    result[mask] = baseline + amplitude
                return result
            else:
                for pulse_time in pulse_times:
                    if pulse_time <= t_eval <= pulse_time + pulse_width:
                        return baseline + amplitude
                return baseline

        I.pulse_times = pulse_times
        I.pulse_width = pulse_width
        return I

    def simulate_with_pulses(self, T, dt, X0, I_ext):
        """
        Simulate with precise handling of brief current pulses
        """
        self.dt = dt
        t_eval = np.arange(0, T, dt)

        if not callable(I_ext) or not hasattr(I_ext, 'pulse_times'):
            raise ValueError("I_ext must be a function created by create_pulse_train")

        t_dense = list(t_eval)
        eps = dt / 1000
        n_extra = 50

        for t_pulse in I_ext.pulse_times:
            t_dense.extend([t_pulse - eps, t_pulse - eps / 2, t_pulse])
            pulse_duration = np.linspace(t_pulse, t_pulse + I_ext.pulse_width, n_extra)
            t_dense.extend(pulse_duration)
            t_dense.extend([t_pulse + I_ext.pulse_width,
                            t_pulse + I_ext.pulse_width + eps / 2,
                            t_pulse + I_ext.pulse_width + eps])

        t_dense = sorted(set(t_dense))
        t_dense = np.array(t_dense)

        solution_dense = odeint(self.dALLdt, X0, t_dense, args=(I_ext,), rtol=1e-8, atol=1e-8, hmax=dt / 10)
        solution = np.array([np.interp(t_eval, t_dense, solution_dense[:, i]) for i in range(2)]).T
        return t_eval, solution

    def find_separatrix(self, I_ext, x_range=(-90, 20), num_points=10000, eps=1e-6, t_max=100, dt=0.005):
        """
        Find the separatrix by computing the stable manifold of the saddle point.
        """
        equilibria = self.find_equlibrium_points(I_ext, x_range)
        saddle_point = None
        for eq in equilibria:
            if eq['stability'] == 'saddle':
                saddle_point = eq
                break

        if saddle_point is None:
            raise ValueError("No saddle point found. This might not be a saddle-node bifurcation case.")

        eigenvals = saddle_point['eigenvalues']
        eigenvecs = np.linalg.eig(saddle_point['jacobian'])[1]
        stable_idx = np.argmin(np.real(eigenvals))
        stable_eigenvec = np.real(eigenvecs[:, stable_idx])
        stable_eigenvec = stable_eigenvec / np.linalg.norm(stable_eigenvec)

        V_saddle, n_saddle = saddle_point['point']
        points_pos = np.array([V_saddle + eps * stable_eigenvec[0], n_saddle + eps * stable_eigenvec[1]])
        points_neg = np.array([V_saddle - eps * stable_eigenvec[0], n_saddle - eps * stable_eigenvec[1]])

        t = np.arange(0, t_max, dt)
        sol_pos = odeint(self.dALLdt_backwards, points_pos, t, args=(lambda t: I_ext,))
        sol_neg = odeint(self.dALLdt_backwards, points_neg, t, args=(lambda t: I_ext,))

        V_separatrix = np.concatenate([sol_neg[::-1, 0], sol_pos[:, 0]])
        n_separatrix = np.concatenate([sol_neg[::-1, 1], sol_pos[:, 1]])
        mask = n_separatrix >= 0
        V_separatrix, n_separatrix = V_separatrix[mask], n_separatrix[mask]
        return V_separatrix, n_separatrix

    def find_unstable_limit_cycle(self, I_ext, dt=0.025, T_max=2000):
        """
        Find the unstable limit cycle for the subcritical Hopf bifurcation
        """
        self.dt = dt
        I_ext_array = np.ones(int(T_max / dt)) * I_ext
        try:
            X0 = self.get_stable_equlibrium_location(I_ext)
            X0[0] += 0.02
        except:
            raise ValueError(
                "No stable point found. This might not be a Hopf bifurcation case below a bifurcation point.")

        t = np.arange(0, T_max, dt)
        solution = odeint(self.dALLdt_backwards, X0, t, args=(I_ext_array,))
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(solution[:, 0], distance=int(0.1 / dt))
        last_cycle_start = peaks[-3]
        last_cycle_end = peaks[-1]
        return solution[last_cycle_start:last_cycle_end, 0], solution[last_cycle_start:last_cycle_end, 1]

    def get_stable_equlibrium_location(self, I_ext):
        equilibria = self.find_equlibrium_points(I_ext, [-90, 20])
        stable_eq = [eq for eq in equilibria if eq['stability'] == 'stable'][0]
        X0 = [stable_eq['point'][0], stable_eq['point'][1]]
        return X0

    def get_unstable_equlibrium_location(self, I_ext):
        equilibria = self.find_equlibrium_points(I_ext, [-90, 20])
        unstable_eq = [eq for eq in equilibria if eq['stability'] == 'unstable'][0]
        X0 = [unstable_eq['point'][0], unstable_eq['point'][1]]
        return X0

    def get_saddle_equlibrium_location(self, I_ext):
        equilibria = self.find_equlibrium_points(I_ext, [-90, 20])
        try:
            saddle_eq = [eq for eq in equilibria if eq['stability'] == 'saddle'][0]
            X0 = [saddle_eq['point'][0], saddle_eq['point'][1]]
            return X0
        except:
            raise ValueError("No saddle point found")

    def find_limit_cycle(self, I_ext, dt=0.01, T_max=200, T_start=None):
        """
        Return one cycle of the limit cycle after discarding transients (saddle-node bifurcation)
        """
        t = np.arange(0, T_max, dt)
        try:
            X0 = self.get_unstable_equlibrium_location(I_ext)
            X0[0] += 0.1
            _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)

            if T_start is None:
                half_idx = len(t) // 2
                solution = solution[half_idx:]
            else:
                start_idx = np.searchsorted(t, T_start)
                solution = solution[start_idx:]
            return solution[:, 0], solution[:, 1]
        except:
            return [], []

    def find_spiking_orbit_subcritical_Hopf(self, I_ext, dt=0.01, T_max=200):
        """
        Return one cycle of the limit cycle (subcritical Hopf bifurcation)
        """
        t = np.arange(0, T_max, dt)
        X0 = [0, 0.05]
        _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(solution[:, 0], distance=int(0.1 / dt))
        return solution[peaks[-2]:peaks[-1], 0], solution[peaks[-2]:peaks[-1], 1]

    def find_invariant_circle(self, I_ext, dt=0.01, T_max=50):
        """
        Return one cycle of the limit cycle (SNIC bifurcation)
        """
        t = np.arange(0, T_max, dt)
        X0 = self.get_saddle_equlibrium_location(I_ext)
        X0[0] += 0.1
        _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
        return solution[:, 0], solution[:, 1]

    def find_aligned_limit_cycle(self, I_ext, dt=0.01, T_max=200, n_cycles=3, align_phase=True):
        """
        Return phase-aligned voltage oscillations for smoother animations
        """
        try:
            X0 = self.get_unstable_equlibrium_location(I_ext)
            X0[0] += 0.1
            _, solution = self.simulate(T_max, dt, X0, lambda t: I_ext)
            half_idx = len(solution) // 2
            V = solution[half_idx:, 0]

            if align_phase:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(V, distance=int(0.1 / dt))
                if len(peaks) < 2:
                    return None, None
                period = np.mean(np.diff(peaks))
                start_idx = peaks[0]
                end_idx = start_idx + int(period * n_cycles)
                if end_idx > len(V):
                    end_idx = len(V)
                V_aligned = V[start_idx:end_idx]
                t_aligned = np.arange(len(V_aligned)) * dt
                return t_aligned, V_aligned
            else:
                return np.arange(len(V)) * dt, V
        except:
            return None, None