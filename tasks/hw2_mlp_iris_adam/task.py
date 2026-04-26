import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "hw2_mlp_iris_adam",
        "description": "Two-layer MLP on Iris with Adam optimizer and dropout",
        "input_dim": 4,
        "output_dim": 3,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_ratio=0.8, batch_size=16):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden1=32, hidden2=16, output_dim=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_model(device=None):
    device = device or get_device()
    return MLPClassifier().to(device)


def train(model, train_loader, val_loader, device, epochs=150, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0:
            v = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch+1}/{epochs}  Acc: {v['accuracy']:.4f}  F1: {v['f1_macro']:.4f}"
            )


def evaluate(model, data_loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(yb.numpy())
    y_pred = np.array(all_pred)
    y_true = np.array(all_true)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict(model, X, device):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(device)
    with torch.no_grad():
        return model(X).argmax(dim=1).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: MLP Iris (Adam + Dropout)")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)
    print(
        f"\nTrain  MSE: {train_m['mse']:.4f}  R2: {train_m['r2']:.4f}  "
        f"Acc: {train_m['accuracy']:.4f}  F1: {train_m['f1_macro']:.4f}"
    )
    print(
        f"Val    MSE: {val_m['mse']:.4f}  R2: {val_m['r2']:.4f}  "
        f"Acc: {val_m['accuracy']:.4f}  F1: {val_m['f1_macro']:.4f}"
    )

    if val_m["accuracy"] > 0.90:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL: acc={val_m['accuracy']:.4f}")
        sys.exit(1)
