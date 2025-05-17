import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from termcolor import colored
from NeuronModel_d import *
from pathlib import Path
from pyswarm import pso

VOLTAGE_LIMITS = [-90, 20]
SPIKE_THRESHOLD = -50  # 尖峰阈值 (mV)
VOLTAGE_THRESHOLD = -65.0  # 高/低电压阈值 (mV)


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
    """检测尖峰时刻"""
    spikes = []
    for i in range(1, len(V)):
        if V[i] > V_th and V[i - 1] <= V_th:
            spikes.append(i * dt)
    return spikes


def compute_spike_frequency(spike_times, T):
    """计算尖峰频率 (Hz)"""
    if len(spike_times) < 2:
        return 0.0
    return (len(spike_times) - 1) / (T / 1000.0)


def generate_neuron_parameters(neuron_idx):
    """为每个神经元生成自由的参数（初始值）"""
    bifurcation_types = ['SNIC', 'saddle-node', 'supercritical_Hopf', 'subcritical_Hopf']
    bifurcation_type = np.random.choice(bifurcation_types)

    np.random.seed(neuron_idx)
    params = {
        'bifurcation_type': bifurcation_type,
        'C_m': np.random.uniform(0.5, 1.5),
        'g_Na': np.random.uniform(5.0, 20.0),
        'g_K': np.random.uniform(3.0, 12.0),
        'g_L': np.random.uniform(1.0, 8.0),
        'E_Na': 60.0,
        'E_K': -90.0,
        'E_L': np.random.uniform(-82.0, -76.0),
        'V_mid_m': np.random.uniform(-35.0, -15.0),
        'V_mid_n': np.random.uniform(-50.0, -20.0),
        'k_m': np.random.uniform(5.0, 20.0),
        'k_n': np.random.uniform(3.0, 7.0)
    }
    return params


def params_to_neurons(params, N_input=6, N_hidden=4, N_output=2):
    """将 PSO 参数向量转换为神经元列表"""
    param_names = ['C_m', 'g_Na', 'g_K', 'g_L', 'E_L', 'V_mid_m', 'V_mid_n', 'k_m', 'k_n']
    bifurcation_types = ['SNIC', 'saddle-node', 'supercritical_Hopf', 'subcritical_Hopf']

    input_neurons = []
    hidden_neurons = []
    output_neurons = []

    params_per_neuron = len(param_names)
    for i in range(N_input):
        neuron_params = {
            'bifurcation_type': np.random.choice(bifurcation_types),
            'E_Na': 60.0,
            'E_K': -90.0
        }
        for j, name in enumerate(param_names):
            neuron_params[name] = params[i * params_per_neuron + j]
        input_neurons.append(NeuronModel(**neuron_params))

    for i in range(N_hidden):
        neuron_params = {
            'bifurcation_type': np.random.choice(bifurcation_types),
            'E_Na': 60.0,
            'E_K': -90.0
        }
        for j, name in enumerate(param_names):
            neuron_params[name] = params[(N_input + i) * params_per_neuron + j]
        hidden_neurons.append(NeuronModel(**neuron_params))

    for i in range(N_output):
        neuron_params = {
            'bifurcation_type': np.random.choice(bifurcation_types),
            'E_Na': 60.0,
            'E_K': -90.0
        }
        for j, name in enumerate(param_names):
            neuron_params[name] = params[(N_input + N_hidden + i) * params_per_neuron + j]
        output_neurons.append(NeuronModel(**neuron_params))

    return input_neurons, hidden_neurons, output_neurons


def save_neuron_parameters(neurons, layer_name, save_folder):
    """保存神经元参数到文件"""
    output_path = os.path.join(save_folder, f'{layer_name}_parameters.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, neuron in enumerate(neurons):
            f.write(f"Neuron {i + 1} ({neuron.bifurcation_type}):\n")
            for key in ['C_m', 'g_Na', 'g_K', 'g_L', 'E_Na', 'E_K', 'E_L', 'V_mid_m', 'V_mid_n', 'k_m', 'k_n']:
                value = getattr(neuron, key)
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
    print(colored(f"Saved {layer_name} parameters to {output_path}", 'green'))


def generate_input_data(num_samples=8, T=100, dt=0.02, seed=42):
    """生成 [n, V] 初始条件数据"""
    np.random.seed(seed)
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


def simulate_network(input_neurons, hidden_neurons, output_neurons, initial_states, T=100, dt=0.02):
    """模拟三层神经网络"""
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
        try:
            _, solution = input_neurons[j].simulate(T, dt, X0, lambda t: 0.0)
            V_input[j, :] = solution[:, 0]
            n_input[j, :] = solution[:, 1]
        except RuntimeError:
            return None

    for i in range(nt - 1):
        for j in range(N_hidden):
            I_syn_total = 0.0
            for k in range(N_input):
                I_syn_total += g_syn_ih[k, j, i] * (syn_type_ih[k, j] - V_hidden[j, i])
            X = [V_hidden[j, i], n_hidden[j, i]]
            try:
                dV_dt, dn_dt = hidden_neurons[j].dALLdt(X, t[i], lambda t: I_syn_total)
                V_hidden[j, i + 1] = V_hidden[j, i] + dt * dV_dt
                n_hidden[j, i + 1] = n_hidden[j, i] + dt * dn_dt
            except RuntimeError:
                return None

        for j in range(N_output):
            I_syn_total = 0.0
            for k in range(N_hidden):
                I_syn_total += g_syn_ho[k, j, i] * (syn_type_ho[k, j] - V_output[j, i])
            X = [V_output[j, i], n_output[j, i]]
            try:
                dV_dt, dn_dt = output_neurons[j].dALLdt(X, t[i], lambda t: I_syn_total)
                V_output[j, i + 1] = V_output[j, i] + dt * dV_dt
                n_output[j, i + 1] = n_output[j, i] + dt * dn_dt
            except RuntimeError:
                return None

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


def evaluate_output(V_output, T, dt):
    """评估输出层，基于尖峰频率分类"""
    spikes_1 = detect_spikes(V_output[0], dt)
    spikes_2 = detect_spikes(V_output[1], dt)
    freq_1 = compute_spike_frequency(spikes_1, T)
    freq_2 = compute_spike_frequency(spikes_2, T)
    return 1 if freq_1 > freq_2 else 0


def objective_function(params, data, labels, T=100, dt=0.02, save_folder=None):
    """PSO 目标函数：返回负准确率"""
    input_neurons, hidden_neurons, output_neurons = params_to_neurons(params)
    correct = 0
    for initial_states, true_label in zip(data, labels):
        result = simulate_network(input_neurons, hidden_neurons, output_neurons, initial_states, T, dt)
        if result is None:
            return 0.0
        t, _, _, _, _, V_output, _, _, _ = result
        pred_label = evaluate_output(V_output, T, dt)
        if pred_label == true_label:
            correct += 1
    accuracy = correct / len(data)
    print(colored(f"Current accuracy: {accuracy * 100:.2f}%", 'cyan'))
    # 保存当前参数（如果准确率高）
    if save_folder and accuracy > 0.7:  # 仅保存较高准确率的参数
        save_neuron_parameters(input_neurons, f'input_interim_{accuracy:.2f}', save_folder)
        save_neuron_parameters(hidden_neurons, f'hidden_interim_{accuracy:.2f}', save_folder)
        save_neuron_parameters(output_neurons, f'output_interim_{accuracy:.2f}', save_folder)
    return -accuracy


def train_network(data, labels, save_folder, T=100, dt=0.02):
    """使用 PSO 训练网络参数"""
    N_input, N_hidden, N_output = 6, 4, 2
    params_per_neuron = 9
    total_params = (N_input + N_hidden + N_output) * params_per_neuron

    lb = []
    ub = []
    for _ in range(N_input + N_hidden + N_output):
        lb.extend([0.5, 5.0, 3.0, 1.0, -82.0, -35.0, -50.0, 5.0, 3.0])
        ub.extend([1.5, 20.0, 12.0, 8.0, -76.0, -15.0, -20.0, 20.0, 7.0])

    print(colored("Starting PSO training...", 'blue'))
    best_params, best_cost = pso(
        func=lambda x: objective_function(x, data, labels, T, dt, save_folder),
        lb=lb,
        ub=ub,
        swarmsize=15,
        maxiter=15,
        omega=0.5,
        phip=2.0,
        phig=2.0,
        debug=True
    )

    best_accuracy = -best_cost
    print(colored(f"Training completed. Best training accuracy: {best_accuracy * 100:.2f}%", 'green'))

    input_neurons, hidden_neurons, output_neurons = params_to_neurons(best_params)
    save_neuron_parameters(input_neurons, 'input_best', save_folder)
    save_neuron_parameters(hidden_neurons, 'hidden_best', save_folder)
    save_neuron_parameters(output_neurons, 'output_best', save_folder)

    return input_neurons, hidden_neurons, output_neurons, best_accuracy


def main():
    save_folder = setup_animation_folder()
    T, dt = 100, 0.02

    train_data, train_labels, t = generate_input_data(num_samples=8, T=T, dt=dt, seed=42)
    test_data, test_labels, _ = generate_input_data(num_samples=8, T=T, dt=dt, seed=43)

    input_neurons, hidden_neurons, output_neurons, train_accuracy = train_network(
        train_data, train_labels, save_folder, T, dt
    )

    correct = 0
    for initial_states, true_label in zip(test_data, test_labels):
        result = simulate_network(input_neurons, hidden_neurons, output_neurons, initial_states, T, dt)
        if result is None:
            continue
        t, _, _, _, _, V_output, _, _, _ = result
        pred_label = evaluate_output(V_output, T, dt)
        if pred_label == true_label:
            correct += 1
    test_accuracy = correct / len(test_data)

    print(f"Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()