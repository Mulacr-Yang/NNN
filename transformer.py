import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_input_data(num_samples=100, T=200, dt=0.01):
    """生成 [n, V] 初始条件数据"""
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


def preprocess_data(X, y, normalize=False):
    """数据预处理：可选标准化，转换为张量"""
    if normalize:
        X_n = X[:, :, 0]
        X_V = X[:, :, 1]
        scaler = MinMaxScaler()
        X_V_scaled = scaler.fit_transform(X_V)
        X_scaled = np.stack([X_n, X_V_scaled], axis=-1)
    else:
        X_scaled = X

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X_tensor, y_tensor


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    """Transformer 模型，满足神经元约束"""

    def __init__(self, d_model=64, nhead=2, num_encoder_layers=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(d_model, 63)  # 63 神经元
        self.fc2 = nn.Linear(63, 1)  # 1 神经元
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 32, 2)
        x = self.embedding(x)  # (batch, 32, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, 32, d_model)
        x = x.mean(dim=1)  # 平均池化: (batch, d_model)
        x = self.relu(self.fc1(x))  # (batch, 63)
        x = self.sigmoid(self.fc2(x))  # (batch, 1)
        return x


def train_model(model, X, y, epochs=100, batch_size=32):
    """训练模型"""
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def evaluate_model(model, X, y):
    """评估模型准确率"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy


def main():
    # 生成数据
    X, y, _ = generate_input_data(num_samples=100, T=200, dt=0.01)

    # 预处理数据
    X_norm, y_norm = preprocess_data(X, y, normalize=True)
    X_non_norm, y_non_norm = preprocess_data(X, y, normalize=False)

    # 初始预测（标准化）
    transformer_initial_norm = TransformerModel().to(device)
    acc_initial_norm = evaluate_model(transformer_initial_norm, X_norm, y_norm)
    print(f"Transformer Initial Prediction (Normalized) Accuracy: {acc_initial_norm * 100:.2f}%")

    # 初始预测（非标准化）
    transformer_initial_non_norm = TransformerModel().to(device)
    acc_initial_non_norm = evaluate_model(transformer_initial_non_norm, X_non_norm, y_non_norm)
    print(f"Transformer Initial Prediction (Non-Normalized) Accuracy: {acc_initial_non_norm * 100:.2f}%")

    # 训练和预测（标准化）
    transformer_trained_norm = TransformerModel().to(device)
    train_model(transformer_trained_norm, X_norm, y_norm)
    acc_trained_norm = evaluate_model(transformer_trained_norm, X_norm, y_norm)
    print(f"Transformer Trained Prediction (Normalized) Accuracy: {acc_trained_norm * 100:.2f}%")

    # 训练和预测（非标准化）
    transformer_trained_non_norm = TransformerModel().to(device)
    train_model(transformer_trained_non_norm, X_non_norm, y_non_norm)
    acc_trained_non_norm = evaluate_model(transformer_trained_non_norm, X_non_norm, y_non_norm)
    print(f"Transformer Trained Prediction (Non-Normalized) Accuracy: {acc_trained_non_norm * 100:.2f}%")


if __name__ == "__main__":
    main()