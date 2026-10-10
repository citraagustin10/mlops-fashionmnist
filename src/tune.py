import hashlib
import json
import random
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEVICE = torch.device("cpu")
SEED = 42

# Dataset kecil untuk tuning awal agar lebih ringan di CPU.
TRAIN_LIMIT = 2000
TEST_LIMIT = 500
EPOCHS = 2


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def dataset_hash():
    """Hash isi file dataset lokal yang digunakan eksperimen."""
    digest = hashlib.sha256()
    files = sorted((DATA_DIR / "FashionMNIST" / "raw").glob("*"))
    if not files:
        files = sorted((DATA_DIR / "FashionMNIST" / "processed").glob("*"))
    for path in files:
        if path.is_file():
            digest.update(path.name.encode())
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def load_data():
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    train_set = Subset(train_set, range(min(TRAIN_LIMIT, len(train_set))))
    test_set = Subset(test_set, range(min(TEST_LIMIT, len(test_set))))
    return train_set, test_set


def train_and_evaluate(params, train_set, test_set, run_seed):
    seed_everything(run_seed)
    train_loader = DataLoader(
        train_set, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = SimpleCNN(dropout=params["dropout"]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(DEVICE)).argmax(dim=1)
            correct += (predictions == labels.to(DEVICE)).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fashion-MNIST-CNN")

    print("Menyiapkan Fashion-MNIST...")
    train_set, test_set = load_data()
    data_digest = dataset_hash()
    print("Hash dataset:", data_digest)
    print(f"Data tuning: {len(train_set)} train, {len(test_set)} test")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64]
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }

        # Satu MLflow run per trial: memudahkan compare view.
        with mlflow.start_run(run_name=f"optuna-trial-{trial.number:02d}"):
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("seed", SEED)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("train_samples", len(train_set))
            mlflow.log_param("test_samples", len(test_set))
            mlflow.log_param("dataset_sha256", data_digest)
            mlflow.log_params(params)

            accuracy = train_and_evaluate(
                params, train_set, test_set, SEED + trial.number
            )
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.set_tag("stage", "hyperparameter_tuning")
            mlflow.set_tag("dataset", "Fashion-MNIST")

            trial.set_user_attr("mlflow_run_name", f"optuna-trial-{trial.number:02d}")
            print(
                f"Trial {trial.number + 1}/20 | "
                f"accuracy={accuracy:.4f} | params={params}"
            )
            return accuracy

    study = optuna.create_study(
        direction="maximize",
        study_name="fashion-mnist-cnn-tuning",
    )
    study.optimize(objective, n_trials=20)

    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "dataset_sha256": data_digest,
    }

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    summary_path = reports_dir / "tuning_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTUNING SELESAI")
    print("Jumlah trial:", len(study.trials))
    print("Akurasi terbaik:", round(study.best_value, 4))
    print("Parameter terbaik:", study.best_params)
    print("Ringkasan disimpan di:", summary_path)


if __name__ == "__main__":
    main()