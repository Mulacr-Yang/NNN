import numpy as np
import os
from tqdm import tqdm
from termcolor import colored
from NeuronModel_d import NeuronModel
from pathlib import Path

VOLTAGE_LIMITS = [-90, 20]
SPIKE_THRESHOLD = -50
VOLTAGE_THRESHOLD = -65.0


def make_folder(name, root_dir='./png'):
    path = Path(root_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_animation_folder(base_dir='./png', prefix="multi_neuron"):
    from datetime import datetime
    folder_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def detect_spikes(V, dt, V_th=SPIKE_THRESHOLD):
    spikes = []
    for i in range(1, len(V)):
        if V[i] > V_th and V[i - 1] <= V_th:
            spikes.append(i * dt)
    return spikes


def compute_spike_frequency(spike_times, T):
    if len(spike_times) < 2:
        return 0.0
    return (len(spike_times) - 1) / (T / 1000.0)


def generate_neuron_parameters(bifurcation_type, neuron_idx):
    base_params = {
        'SNIC': {
            'C_m': 1.0, 'g_Na': 20.0, 'g_K': 10.0, 'g_L': 8.0,
            'E_Na': 60.0, 'E_K': -90.0, 'E_L': -80.0,
            'V_mid_m': -20.0, 'V_mid_n': -25.0, 'k_m': 15.0, 'k_n': 5.0
        },
        'saddle-node': {
            'C_m': 1.0, 'g_Na': 20.0, 'g_K': 10.0, 'g_L': 8.0,
            'E_Na': 60.0, 'E_K': -90.0, 'E_L': -80.0,
            'V_mid_m': -20.0, 'V_mid_n': -25.0, 'k_m': 15.0, 'k_n': 5.0
        },
        'supercritical_Hopf': {
            'C_m': 1.0, 'g_Na': 20.0, 'g_K': 10.0, 'g_L': 8.0,
            'E_Na': 60.0, 'E_K': -90.0, 'E_L': -78.0,
            'V_mid_m': -20.0, 'V_mid_n': -45.0, 'k_m': 15.0, 'k_n': 5.0
        },
        'subcritical_Hopf': {
            'C_m': 1.0, 'g_Na': 4.0, 'g_K': 4.0, 'g_L': 1.0,
            'E_Na': 60.0, 'E_K': -90.0, 'E_L': -78.0,
            'V_mid_m': -30.0, 'V_mid_n': -45.0, 'k_m': 7.0, 'k_n': 5.0
        }
    }

    params = base_params[bifurcation_type].copy()
    np.random.seed(neuron_idx)
    for key in ['C_m', 'g_Na', 'g_K', 'g_L', 'V_mid_m', 'V_mid_n', 'k_m', 'k_n']:
        params[key] *= np.random.uniform(0.9, 1.1)
    return params


def save_neuron_parameters(neurons, layer_name, save_folder):
    output_path = os.path.join(save_folder, f'{layer_name}_parameters.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, neuron in enumerate(neurons):
            f.write(f"Neuron {i + 1} ({neuron.bifurcation_type}):\n")
            for key in ['C_m', 'g_Na', 'g_K', 'g_L', 'E_Na', 'E_K', 'E_L', 'V_mid_m', 'V_mid_n', 'k_m', 'k_n']:
                value = getattr(neuron, key)
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
    print(colored(f"Saved {layer_name} parameters to {output_path}", 'green'))


def generate_input_currents(num_samples=100, T=200, dt=0.01):
    np.random.seed(42)
    t = np.arange(0, T, dt)
    currents = []
    for _ in range(num_samples):
        baseline = 0.0
        amplitude = np.random.uniform(0.0, 10.0)
        step_time = np.random.uniform(50.0, 150.0)
        step_duration = 50.0
        I_ext = NeuronModel.create_step_current(t, step_time, step_duration, baseline, amplitude)
        currents.append(I_ext)
    return currents, t


def simulate_network(hidden_neurons, output_neurons, I_ext, T=200, dt=0.01):
    N_hidden = len(hidden_neurons)
    N_output = len(output_neurons)

    t = np.arange(0, T, dt)
    nt = len(t)

    V_hidden = np.zeros((N_hidden, nt))
    n_hidden = np.zeros((N_hidden, nt))
    V_output = np.zeros((N_output, nt))
    n_output = np.zeros((N_output, nt))

    g_syn_hh = np.zeros((N_hidden, N_hidden, nt))  # Hidden to hidden
    g_syn_ho = np.zeros((N_hidden, N_output, nt))  # Hidden to output

    # Randomize initial states for each simulation
    np.random.seed()  # Reset seed for random initial states
    initial_states_hidden = [
        [np.random.uniform(-80.0, -70.0), np.random.uniform(0.2, 0.5)] for _ in range(N_hidden)
    ]
    initial_states_output = [
        [-80.0, 0.3], [-80.0, 0.4]
    ]

    for j in range(N_hidden):
        V_hidden[j, 0] = initial_states_hidden[j][0]
        n_hidden[j, 0] = initial_states_hidden[j][1]
    for j in range(N_output):
        V_output[j, 0] = initial_states_output[j][0]
        n_output[j, 0] = initial_states_output[j][1]

    g_syn_max = 2.0
    tau_syn = 5.0
    E_syn_exc = 0.0
    E_syn_inh = -70.0

    syn_type_hh = np.ones((N_hidden, N_hidden)) * E_syn_exc
    syn_type_ho = np.ones((N_hidden, N_output)) * E_syn_exc

    for i in range(nt - 1):
        for j in range(N_hidden):
            I_syn_total = 0.0
            for k in range(N_hidden):
                if k != j:  # No self-connection
                    I_syn_total += g_syn_hh[k, j, i] * (syn_type_hh[k, j] - V_hidden[j, i])
            X = [V_hidden[j, i], n_hidden[j, i]]
            dV_dt, dn_dt = hidden_neurons[j].dALLdt(X, t[i], lambda t: I_ext[i] + I_syn_total)
            V_hidden[j, i + 1] = V_hidden[j, i] + dt * dV_dt
            n_hidden[j, i + 1] = n_hidden[j, i] + dt * dn_dt

        for j in range(N_output):
            I_syn_total = 0.0
            for k in range(N_hidden):
                I_syn_total += g_syn_ho[k, j, i] * (syn_type_ho[k, j] - V_output[j, i])
            X = [V_output[j, i], n_output[j, i]]
            dV_dt, dn_dt = output_neurons[j].dALLdt(X, t[i], lambda t: I_syn_total)
            V_output[j, i + 1] = V_output[j, i] + dt * dV_dt
            n_output[j, i + 1] = n_output[j, i] + dt * dn_dt

        for j in range(N_hidden):
            if V_hidden[j, i] <= SPIKE_THRESHOLD and V_hidden[j, i + 1] > SPIKE_THRESHOLD:
                for k in range(N_hidden):
                    if k != j:
                        g_syn_hh[j, k, i] += g_syn_max
                for k in range(N_output):
                    g_syn_ho[j, k, i] += g_syn_max
            for k in range(N_hidden):
                if k != j:
                    g_syn_hh[j, k, i + 1] = g_syn_hh[j, k, i] + dt * (-g_syn_hh[j, k, i] / tau_syn)
            for k in range(N_output):
                g_syn_ho[j, k, i + 1] = g_syn_ho[j, k, i] + dt * (-g_syn_ho[j, k, i] / tau_syn)

        if i % int(50 / dt) == 0 and i > 0:
            for j in range(N_hidden):
                spikes = detect_spikes(V_hidden[j, :i + 1], dt)
                freq = compute_spike_frequency(spikes, t[i])
                for k in range(N_hidden):
                    if k != j:
                        syn_type_hh[j, k] = E_syn_exc if freq > 10.0 else E_syn_inh
                for k in range(N_output):
                    syn_type_ho[j, k] = E_syn_exc if freq > 10.0 else E_syn_inh

    return t, V_hidden, n_hidden, V_output, n_output, g_syn_hh, g_syn_ho


def evaluate_output(V_output, n_output):
    return V_output[:, -1], n_output[:, -1]


def main(num_runs=100, num_samples=100):
    hidden_neurons = [
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 0)),
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 1)),
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 2)),
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 3)),
        NeuronModel(bifurcation_type='saddle-node', **generate_neuron_parameters('saddle-node', 4)),
        NeuronModel(bifurcation_type='saddle-node', **generate_neuron_parameters('saddle-node', 5)),
        NeuronModel(bifurcation_type='saddle-node', **generate_neuron_parameters('saddle-node', 6)),
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 7)),
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 8)),
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 9))
    ]
    output_neurons = [
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 10)),
        NeuronModel(bifurcation_type='subcritical_Hopf', **generate_neuron_parameters('subcritical_Hopf', 11))
    ]

    save_folder = setup_animation_folder()
    save_neuron_parameters(hidden_neurons, 'hidden', save_folder)
    save_neuron_parameters(output_neurons, 'output', save_folder)

    currents, t = generate_input_currents(num_samples=num_samples, T=200, dt=0.01)

    for run_idx in tqdm(range(num_runs), desc="Running simulation sets"):
        output_path = os.path.join(save_folder, f'output_endpoints_{run_idx}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Sample,V1,n1,V2,n2\n")
            for sample_idx, I_ext in enumerate(currents):
                t, V_hidden, n_hidden, V_output, n_output, g_syn_hh, g_syn_ho = simulate_network(
                    hidden_neurons, output_neurons, I_ext, T=200, dt=0.01
                )
                V_end, n_end = evaluate_output(V_output, n_output)
                f.write(f"{sample_idx},{V_end[0]:.4f},{n_end[0]:.4f},{V_end[1]:.4f},{n_end[1]:.4f}\n")
        print(colored(f"Saved endpoints to {output_path}", 'green'))


if __name__ == "__main__":
    main(num_runs=10, num_samples=10)