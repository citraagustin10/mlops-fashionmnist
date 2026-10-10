import hashlib
import json
import random
import re
import statistics
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "FashionMNIST"
DVC_FILE = PROJECT_DIR / "data" / "FashionMNIST.dvc"
DEVICE = torch.device("cpu")

# Parameter terbaik dari hasil tuning 20 trial
BEST_PARAMS = {
    "learning_rate": 0.0027792992750462365,
    "batch_size": 64,
    "dropout": 0.24444525375713033,
}

SEEDS = [42, 123, 2026]
EPOCHS = 2


def calculate_dataset_hash(dataset_dir):
    """Hitung SHA-256 berdasarkan nama relatif dan isi file dataset."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {dataset_dir}"
        )

    sha256 = hashlib.sha256()
    file_count = 0

    for file_path in sorted(dataset_dir.rglob("*")):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(
            dataset_dir
        ).as_posix()

        sha256.update(relative_path.encode("utf-8"))

        with file_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024), b""
            ):
                sha256.update(chunk)

        file_count += 1

    if file_count == 0:
        raise RuntimeError(
            f"Tidak ada file dataset di {dataset_dir}"
        )

    return sha256.hexdigest()


def get_dvc_dataset_hash(dvc_file):
    """Ambil hash direktori dataset dari file pointer DVC."""
    if not dvc_file.is_file():
        raise FileNotFoundError(
            f"File DVC tidak ditemukan: {dvc_file}"
        )

    content = dvc_file.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^\s*-\s*md5:\s*([0-9a-fA-F]+(?:\.dir)?)\s*$",
        content,
    )

    if not match:
        raise ValueError(
            f"Hash MD5 tidak ditemukan di {dvc_file}"
        )

    return match.group(1)


class SimpleCNN(nn.Module):
    def __init__(self, dropout):
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


def evaluate_seed(seed, train_set, test_set):
    random.seed(seed)
    torch.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=BEST_PARAMS["batch_size"],
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
    )

    model = SimpleCNN(BEST_PARAMS["dropout"]).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=BEST_PARAMS["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        mlflow.log_metric(
            "train_loss",
            average_loss,
            step=epoch,
        )
        print(
            f"Seed {seed} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(DEVICE)).argmax(dim=1)
            correct += (
                predictions == labels.to(DEVICE)
            ).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    mlflow.log_metric("test_accuracy", accuracy)

    # Simpan model seed 123 untuk proses registrasi model.
    if seed == 123:
        example_input = torch.zeros(1, 1, 28, 28)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input,
            serialization_format="pickle",
        )

    return accuracy


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fashion-MNIST-CNN")

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=str(PROJECT_DIR / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(PROJECT_DIR / "data"),
        train=False,
        download=True,
        transform=transform,
    )

    train_set = Subset(train_dataset, range(2000))
    test_set = Subset(test_dataset, range(500))

    # Hitung hash sekali; catat nilainya pada setiap run.
    dataset_sha256 = calculate_dataset_hash(DATA_DIR)
    dvc_dataset_hash = get_dvc_dataset_hash(DVC_FILE)

    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"DVC dataset hash: {dvc_dataset_hash}")

    results = []

    for seed in SEEDS:
        with mlflow.start_run(
            run_name=f"best-config-seed-{seed}"
        ):
            mlflow.log_params(BEST_PARAMS)
            mlflow.log_param("dataset", "Fashion-MNIST")
            mlflow.log_param("dataset_sha256", dataset_sha256)
            mlflow.log_param("dvc_dataset_hash", dvc_dataset_hash)
            mlflow.log_param("seed", seed)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("train_samples", len(train_set))
            mlflow.log_param("test_samples", len(test_set))
            mlflow.log_param("device", str(DEVICE))
            mlflow.set_tag("stage", "three_seed_evaluation")

            accuracy = evaluate_seed(seed, train_set, test_set)
            results.append(accuracy)

            print(
                f"Seed {seed}: test_accuracy={accuracy:.4f}"
            )

    mean_accuracy = statistics.mean(results)
    std_accuracy = statistics.stdev(results)

    summary = {
        "seeds": SEEDS,
        "accuracies": results,
        "mean_accuracy": mean_accuracy,
        "std_accuracy_sample": std_accuracy,
        "mean_plus_minus_std": (
            f"{mean_accuracy:.4f} ± {std_accuracy:.4f}"
        ),
        "best_params": BEST_PARAMS,
        "dataset_sha256": dataset_sha256,
        "dvc_dataset_hash": dvc_dataset_hash,
    }

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "three_seed_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nRINGKASAN 3 SEED")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Standard deviation: {std_accuracy:.4f}")
    print(
        f"Mean ± std: {mean_accuracy:.4f} "
        f"± {std_accuracy:.4f}"
    )
    print(f"Ringkasan disimpan di: {output_path}")


if __name__ == "__main__":
    main()