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


class BasicBlock(nn.Module):
    """ResNet-18 的 BasicBlock，1D 版本"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 模型，适配 1D 输入"""

    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 32, 2) -> (batch, 2, 32)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


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
    resnet_initial_norm = ResNet18().to(device)
    acc_initial_norm = evaluate_model(resnet_initial_norm, X_norm, y_norm)
    print(f"ResNet-18 Initial Prediction (Normalized) Accuracy: {acc_initial_norm * 100:.2f}%")

    # 初始预测（非标准化）
    resnet_initial_non_norm = ResNet18().to(device)
    acc_initial_non_norm = evaluate_model(resnet_initial_non_norm, X_non_norm, y_non_norm)
    print(f"ResNet-18 Initial Prediction (Non-Normalized) Accuracy: {acc_initial_non_norm * 100:.2f}%")

    # 训练和预测（标准化）
    resnet_trained_norm = ResNet18().to(device)
    train_model(resnet_trained_norm, X_norm, y_norm)
    acc_trained_norm = evaluate_model(resnet_trained_norm, X_norm, y_norm)
    print(f"ResNet-18 Trained Prediction (Normalized) Accuracy: {acc_trained_norm * 100:.2f}%")

    # 训练和预测（非标准化）
    resnet_trained_non_norm = ResNet18().to(device)
    train_model(resnet_trained_non_norm, X_non_norm, y_non_norm)
    acc_trained_non_norm = evaluate_model(resnet_trained_non_norm, X_non_norm, y_non_norm)
    print(f"ResNet-18 Trained Prediction (Non-Normalized) Accuracy: {acc_trained_non_norm * 100:.2f}%")


if __name__ == "__main__":
    main()