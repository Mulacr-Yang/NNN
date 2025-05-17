import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_input_data(num_samples=100, T=200, dt=0.01):
    data = []
    labels = []

    for _ in range(num_samples):
        initial_states = []
        for _ in range(32):
            V = np.random.uniform(-80.0, -50.0)
            n = np.random.uniform(0.0, 1.0)
            initial_states.append([n, V])
        initial_states = np.array(initial_states)  # Shape: (32, 2)
        avg_V = np.mean(initial_states[:, 1])
        label = 1 if avg_V > -70.0 else 0
        data.append(initial_states)
        labels.append(label)

    t = np.arange(0, T, dt)
    return np.array(data), np.array(labels), t


def preprocess_data(X, y):
    X_n = X[:, :, 0]  # Shape: (100, 32)
    X_V = X[:, :, 1]  # Shape: (100, 32)

    scaler = MinMaxScaler()
    X_V_scaled = scaler.fit_transform(X_V)  # Shape: (100, 32)

    X_scaled = np.stack([X_n, X_V_scaled], axis=-1)  # Shape: (100, 32, 2)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    return X_tensor, y_tensor

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=63, batch_first=True)
        self.fc = nn.Linear(63, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.rnn(x)  # h_n: (1, batch, 63)
        x = self.relu(h_n[-1])  # (batch, 63)
        x = self.sigmoid(self.fc(x))
        return x


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=63, batch_first=True)
        self.fc = nn.Linear(63, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, 63)
        x = self.relu(h_n[-1])  # (batch, 63)
        x = self.sigmoid(self.fc(x))
        return x


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy


def main():
    X, y, _ = generate_input_data(num_samples=100, T=200, dt=0.01)

    X_tensor, y_tensor = preprocess_data(X, y)

    rnn_model = RNN().to(device)
    rnn_accuracy = evaluate_model(rnn_model, X_tensor, y_tensor)
    print(f"RNN Accuracy: {rnn_accuracy * 100:.2f}%")

    lstm_model = LSTM().to(device)
    lstm_accuracy = evaluate_model(lstm_model, X_tensor, y_tensor)
    print(f"LSTM Accuracy: {lstm_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
