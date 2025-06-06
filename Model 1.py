import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from termcolor import colored
from NeuronModel_d import *
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


def setup_phase_portrait_ax(show_ax=True):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1])
    ax.set_ylim(-0.05, 1.05)
    if show_ax:
        ax.set_xlabel('$V$ (mV)', color='black', fontsize=14)
        ax.set_ylabel('$n$', color='black', fontsize=14)
        ax.tick_params(colors='black')
    else:
        ax.axis('off')
    return fig, ax


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


def compute_vector_field(neuron, V_range, n_range, I_syn=0.0):
    V_grid, n_grid = np.meshgrid(V_range, n_range)
    dV_dt = np.zeros_like(V_grid)
    dn_dt = np.zeros_like(n_grid)
    for i in range(V_grid.shape[0]):
        for j in range(V_grid.shape[1]):
            X = [V_grid[i, j], n_grid[i, j]]
            dV, dn = neuron.dALLdt(X, 0, lambda t: I_syn)
            dV_dt[i, j] = dV
            dn_dt[i, j] = dn
    return V_grid, n_grid, dV_dt, dn_dt


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


def generate_input_data(num_samples=10, T=200, dt=0.01):
    np.random.seed(42)
    data = []
    labels = []

    for _ in range(num_samples):
        initial_states = []
        for _ in range(6):
            V = np.random.uniform(-80.0, -50.0)
            n = np.random.uniform(0.0, 1.0)
            initial_states.append([n, V])
        initial_states = np.array(initial_states)  # Shape: (6, 2)
        avg_V = np.mean(initial_states[:, 1])
        label = 1 if avg_V > VOLTAGE_THRESHOLD else 0
        data.append(initial_states)
        labels.append(label)

    t = np.arange(0, T, dt)
    return data, labels, t


def simulate_network(input_neurons, hidden_neurons, output_neurons, initial_states, T=200, dt=0.01):
    N_input = len(input_neurons)
    N_hidden = len(hidden_neurons)
    N_output = len(output_neurons)

    t = np.arange(0, T, dt)
    nt = len(t)

    V_input = np.zeros((N_input, nt))
    n_input = np.zeros((N_input, nt))
    V_hidden = np.zeros((N_hidden, nt))
    n_hidden = np.zeros((N_hidden, nt))
    V_output = np.zeros((N_output, nt))
    n_output = np.zeros((N_output, nt))

    g_syn_ih = np.zeros((N_input, N_hidden, nt))
    g_syn_ho = np.zeros((N_hidden, N_output, nt))

    for j in range(N_input):
        n_input[j, 0] = initial_states[j][0]
        V_input[j, 0] = initial_states[j][1]

    initial_states_hidden = [
        [-80.0, 0.3], [-80.0, 0.4], [-70.0, 0.2], [-70.0, 0.3]
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

    syn_type_ih = np.ones((N_input, N_hidden)) * E_syn_exc
    syn_type_ho = np.ones((N_hidden, N_output)) * E_syn_exc

    for j in range(N_input):
        X0 = [V_input[j, 0], n_input[j, 0]]
        _, solution = input_neurons[j].simulate(T, dt, X0, lambda t: 0.0)
        V_input[j, :] = solution[:, 0]
        n_input[j, :] = solution[:, 1]

    for i in range(nt - 1):
        for j in range(N_hidden):
            I_syn_total = 0.0
            for k in range(N_input):
                I_syn_total += g_syn_ih[k, j, i] * (syn_type_ih[k, j] - V_hidden[j, i])
            X = [V_hidden[j, i], n_hidden[j, i]]
            dV_dt, dn_dt = hidden_neurons[j].dALLdt(X, t[i], lambda t: I_syn_total)
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

        for j in range(N_input):
            if V_input[j, i] <= SPIKE_THRESHOLD and V_input[j, i + 1] > SPIKE_THRESHOLD:
                for k in range(N_hidden):
                    g_syn_ih[j, k, i] += g_syn_max
            for k in range(N_hidden):
                g_syn_ih[j, k, i + 1] = g_syn_ih[j, k, i] + dt * (-g_syn_ih[j, k, i] / tau_syn)

        for j in range(N_hidden):
            if V_hidden[j, i] <= SPIKE_THRESHOLD and V_hidden[j, i + 1] > SPIKE_THRESHOLD:
                for k in range(N_output):
                    g_syn_ho[j, k, i] += g_syn_max
            for k in range(N_output):
                g_syn_ho[j, k, i + 1] = g_syn_ho[j, k, i] + dt * (-g_syn_ho[j, k, i] / tau_syn)

        if i % int(50 / dt) == 0 and i > 0:
            for j in range(N_input):
                spikes = detect_spikes(V_input[j, :i + 1], dt)
                freq = compute_spike_frequency(spikes, t[i])
                for k in range(N_hidden):
                    syn_type_ih[j, k] = E_syn_exc if freq > 10.0 else E_syn_inh
            for j in range(N_hidden):
                spikes = detect_spikes(V_hidden[j, :i + 1], dt)
                freq = compute_spike_frequency(spikes, t[i])
                for k in range(N_output):
                    syn_type_ho[j, k] = E_syn_exc if freq > 10.0 else E_syn_inh

    return t, V_input, n_input, V_hidden, n_hidden, V_output, n_output, g_syn_ih, g_syn_ho


def plot_timeseries(t, V_input, V_hidden, V_output, input_neurons, save_folder, sample_idx):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    for i, neuron in enumerate(input_neurons):
        label = f'{neuron.bifurcation_type} Neuron {i + 1}'
        ax.plot(t, V_input[i], label=label, color='black', alpha=0.5)
    for i in range(len(V_hidden)):
        ax.plot(t, V_hidden[i], label=f'Hidden Neuron {i + 1}', color='gray', alpha=0.7)
    for i in range(len(V_output)):
        ax.plot(t, V_output[i], label=f'Output Neuron {i + 1}', color='blue' if i == 0 else 'red', linewidth=2)
    ax.set_xlabel('Time (ms)', color='black')
    ax.set_ylabel('$V$ (mV)', color='black')
    ax.tick_params(colors='black')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black', loc='upper right')
    save_path = os.path.join(save_folder, f'timeseries_sample_{sample_idx}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(colored(f"Saved timeseries to {save_path}", 'green'))


def plot_phase_portrait(V_input, n_input, V_hidden, n_hidden, V_output, n_output, input_neurons, hidden_neurons,
                        output_neurons, save_folder, sample_idx):
    V_range = np.linspace(VOLTAGE_LIMITS[0], VOLTAGE_LIMITS[1], 20)
    n_range = np.linspace(-0.05, 1.05, 20)

    for i, neuron in enumerate(input_neurons):
        fig, ax = setup_phase_portrait_ax()
        V_grid, n_grid, dV_dt, dn_dt = compute_vector_field(neuron, V_range, n_range, I_syn=10.0)
        ax.streamplot(V_grid, n_grid, dV_dt, dn_dt, color='gray', linewidth=0.5, density=1.0)
        label = f'{neuron.bifurcation_type} Neuron {i + 1}'
        ax.plot(V_input[i], n_input[i], color='black', lw=2, label=label)
        ax.plot(V_input[i][0], n_input[i][0], 'o', color='green', markersize=8, label='Start')
        ax.plot(V_input[i][-1], n_input[i][-1], '^', color='red', markersize=8, label='End')
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
        save_path = os.path.join(save_folder, f'phase_portrait_input_{i + 1}_sample_{sample_idx}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(colored(f"Saved phase portrait to {save_path}", 'green'))

    for i in range(len(hidden_neurons)):
        fig, ax = setup_phase_portrait_ax()
        V_grid, n_grid, dV_dt, dn_dt = compute_vector_field(hidden_neurons[i], V_range, n_range, I_syn=10.0)
        ax.streamplot(V_grid, n_grid, dV_dt, dn_dt, color='gray', linewidth=0.5, density=1.0)
        ax.plot(V_hidden[i], n_hidden[i], color='gray', lw=2, label=f'Hidden Neuron {i + 1}')
        ax.plot(V_hidden[i][0], n_hidden[i][0], 'o', color='green', markersize=8, label='Start')
        ax.plot(V_hidden[i][-1], n_hidden[i][-1], '^', color='red', markersize=8, label='End')
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
        save_path = os.path.join(save_folder, f'phase_portrait_hidden_{i + 1}_sample_{sample_idx}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(colored(f"Saved phase portrait to {save_path}", 'green'))

    for i in range(len(output_neurons)):
        fig, ax = setup_phase_portrait_ax()
        V_grid, n_grid, dV_dt, dn_dt = compute_vector_field(output_neurons[i], V_range, n_range, I_syn=10.0)
        ax.streamplot(V_grid, n_grid, dV_dt, dn_dt, color='gray', linewidth=0.5, density=1.0)
        ax.plot(V_output[i], n_output[i], color='blue' if i == 0 else 'red', lw=2, label=f'Output Neuron {i + 1}')
        ax.plot(V_output[i][0], n_output[i][0], 'o', color='green', markersize=8, label='Start')
        ax.plot(V_output[i][-1], n_output[i][-1], '^', color='red', markersize=8, label='End')
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
        save_path = os.path.join(save_folder, f'phase_portrait_output_{i + 1}_sample_{sample_idx}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(colored(f"Saved phase portrait to {save_path}", 'green'))


def evaluate_output(V_output, T, dt):
    spikes_1 = detect_spikes(V_output[0], dt)
    spikes_2 = detect_spikes(V_output[1], dt)
    freq_1 = compute_spike_frequency(spikes_1, T)
    freq_2 = compute_spike_frequency(spikes_2, T)
    return 1 if freq_1 > freq_2 else 0


def main():
    input_neurons = [
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 0)),
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 1)),
        NeuronModel(bifurcation_type='saddle-node', **generate_neuron_parameters('saddle-node', 2)),
        NeuronModel(bifurcation_type='saddle-node', **generate_neuron_parameters('saddle-node', 3)),
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 4)),
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 5))
    ]
    hidden_neurons = [
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 6)),
        NeuronModel(bifurcation_type='subcritical_Hopf', **generate_neuron_parameters('subcritical_Hopf', 7)),
        NeuronModel(bifurcation_type='SNIC', **generate_neuron_parameters('SNIC', 8)),
        NeuronModel(bifurcation_type='saddle-node', **generate_neuron_parameters('saddle-node', 9))
    ]
    output_neurons = [
        NeuronModel(bifurcation_type='supercritical_Hopf', **generate_neuron_parameters('supercritical_Hopf', 10)),
        NeuronModel(bifurcation_type='subcritical_Hopf', **generate_neuron_parameters('subcritical_Hopf', 11))
    ]

    save_folder = setup_animation_folder()
    save_neuron_parameters(input_neurons, 'input', save_folder)
    save_neuron_parameters(hidden_neurons, 'hidden', save_folder)
    save_neuron_parameters(output_neurons, 'output', save_folder)

    data, labels, t = generate_input_data(num_samples=100, T=200, dt=0.01)

    correct = 0
    for idx, (initial_states, true_label) in enumerate(zip(data, labels)):
        t, V_input, n_input, V_hidden, n_hidden, V_output, n_output, g_syn_ih, g_syn_ho = simulate_network(
            input_neurons, hidden_neurons, output_neurons, initial_states, T=200, dt=0.01
        )

        # plot_timeseries(t, V_input, V_hidden, V_output, input_neurons, save_folder, idx)
        # plot_phase_portrait(V_input, n_input, V_hidden, n_hidden, V_output, n_output,
        #                    input_neurons, hidden_neurons, output_neurons, save_folder, idx)

        pred_label = evaluate_output(V_output, T=200, dt=0.01)
        if pred_label == true_label:
            correct += 1

    accuracy = correct / len(data)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
