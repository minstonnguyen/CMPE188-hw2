import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "hw2_linreg_california_rmsprop",
        "description": "Linear regression on California Housing with RMSProp and batch norm",
        "input_dim": 8,
        "output_dim": 1,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_ratio=0.8, batch_size=256):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_model(device=None):
    device = device or get_device()
    return MLPRegressor().to(device)


def train(model, train_loader, val_loader, device, epochs=60, lr=1e-2):
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/count:.6f}")


def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
            targets.append(yb)
    y_pred = torch.cat(preds)
    y_true = torch.cat(targets)
    mse = torch.mean((y_pred - y_true) ** 2).item()
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "r2": r2}


def predict(model, X, device):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {metrics['mse']}\nR2: {metrics['r2']}\n")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: MLP Regressor (California Housing + RMSProp + BatchNorm)")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device, epochs=60, lr=1e-2)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print(f"\nTrain MSE: {train_m['mse']:.4f}, R2: {train_m['r2']:.4f}")
    print(f"Val   MSE: {val_m['mse']:.4f}, R2: {val_m['r2']:.4f}")

    if val_m["r2"] > 0.55:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL: R2={val_m['r2']:.4f}, MSE={val_m['mse']:.4f}")
        sys.exit(1)
