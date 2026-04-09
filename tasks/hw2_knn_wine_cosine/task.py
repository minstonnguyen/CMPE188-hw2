import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "hw2_knn_wine_cosine",
        "description": "KNN on Wine dataset with cosine distance metric",
        "input_dim": 13,
        "output_dim": 3,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_ratio=0.8, batch_size=32):
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_wine()
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class CosineKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _cosine_distance(self, A, B):
        A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
        B_norm = B / (B.norm(dim=1, keepdim=True) + 1e-8)
        similarity = A_norm @ B_norm.T
        return 1.0 - similarity

    def predict(self, X):
        dist = self._cosine_distance(X, self.X_train)
        _, topk_idx = torch.topk(dist, self.k, dim=1, largest=False)
        topk_labels = self.y_train[topk_idx]
        preds = torch.mode(topk_labels, dim=1).values
        return preds


def build_model(device=None, k=5):
    return CosineKNN(k=k)


def train(model, train_loader, val_loader, device, **kwargs):
    X_all, y_all = [], []
    for xb, yb in train_loader:
        X_all.append(xb)
        y_all.append(yb)
    X_train = torch.cat(X_all).to(device)
    y_train = torch.cat(y_all).to(device)
    model.fit(X_train, y_train)
    print(f"Stored {len(X_train)} training samples (k={model.k})")


def evaluate(model, data_loader, device):
    all_pred, all_true = [], []
    for xb, yb in data_loader:
        xb = xb.to(device)
        pred = model.predict(xb).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(yb.numpy())
    y_pred = np.array(all_pred)
    y_true = np.array(all_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict(model, X, device):
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(device)
    return model.predict(X).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"X_train": model.X_train.cpu(), "y_train": model.y_train.cpu(), "k": model.k},
        os.path.join(output_dir, "model.pt"),
    )


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: KNN (Wine + Cosine Distance)")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device, k=5)
    train(model, train_loader, val_loader, device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print(f"\nTrain Acc: {train_m['accuracy']:.4f}, F1: {train_m['f1_macro']:.4f}")
    print(f"Val   Acc: {val_m['accuracy']:.4f}, F1: {val_m['f1_macro']:.4f}")

    if val_m["accuracy"] > 0.85:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL: acc={val_m['accuracy']:.4f}")
        sys.exit(1)
