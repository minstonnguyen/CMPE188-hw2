import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "hw2_logreg_breast_cancer_adagrad",
        "description": "MLP classifier on Breast Cancer with Adagrad and early stopping",
        "input_dim": 30,
        "output_dim": 2,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_ratio=0.8, batch_size=32):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
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


def build_model(device=None):
    device = device or get_device()
    return nn.Sequential(
        nn.Linear(30, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    ).to(device)


def _val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            total_loss += nn.functional.cross_entropy(model(xb), yb, reduction="sum").item()
            n += len(yb)
    return total_loss / n


def train(model, train_loader, val_loader, device, epochs=300, lr=0.05, patience=20):
    model.train()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

        vl = _val_loss(model, val_loader, device)
        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Val Loss: {vl:.6f}  (patience {wait}/{patience})")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


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
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
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
    print("Task: MLP Classifier (Breast Cancer + Adagrad + Early Stopping)")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print(
        f"\nTrain MSE: {train_m['mse']:.4f}, R2: {train_m['r2']:.4f}, "
        f"Acc: {train_m['accuracy']:.4f}, F1: {train_m['f1']:.4f}"
    )
    print(
        f"Val   MSE: {val_m['mse']:.4f}, R2: {val_m['r2']:.4f}, "
        f"Acc: {val_m['accuracy']:.4f}, F1: {val_m['f1']:.4f}"
    )
    print(f"Val   Precision: {val_m['precision']:.4f}, Recall: {val_m['recall']:.4f}")

    if val_m["accuracy"] > 0.93:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL: acc={val_m['accuracy']:.4f}")
        sys.exit(1)
