
import hashlib
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# --------------------------------------------------
# Konfigurasi
# --------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Fashion-MNIST-CNN"
DATA_DIR = "data/FashionMNIST"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1
SEED = 42
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 500


# --------------------------------------------------
# Hash dataset SHA-256
# --------------------------------------------------
def calculate_dataset_hash(dataset_dir=DATA_DIR):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {dataset_dir}"
        )

    sha256 = hashlib.sha256()
    file_count = 0

    for root, dirs, files in os.walk(dataset_dir):
        dirs.sort()

        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(
                file_path, dataset_dir
            ).replace("\\", "/")

            sha256.update(relative_path.encode("utf-8"))

            with open(file_path, "rb") as file:
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


# --------------------------------------------------
# Model CNN sederhana
# --------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------
# Training dan evaluasi baseline
# --------------------------------------------------
def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    # Unduh dataset jika belum tersedia.
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # Gunakan subset kecil agar eksperimen cepat.
    train_data = Subset(
        train_dataset, range(TRAIN_SAMPLES)
    )
    test_data = Subset(
        test_dataset, range(TEST_SAMPLES)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Hubungkan ke MLflow tracking server.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Hitung hash setelah dataset tersedia.
    dataset_hash = calculate_dataset_hash()

    with mlflow.start_run(
        run_name="baseline-cnn-test"
    ):
        # Parameter eksperimen.
        mlflow.log_param("dataset", "Fashion-MNIST")
        mlflow.log_param(
            "dataset_sha256", dataset_hash
        )
        mlflow.log_param(
            "train_samples", len(train_data)
        )
        mlflow.log_param(
            "test_samples", len(test_data)
        )
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param(
            "learning_rate", LEARNING_RATE
        )
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("device", str(device))

        print(f"Dataset SHA-256: {dataset_hash}")
        print("Memulai training baseline...")

        # Training.
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            average_loss = (
                total_loss / len(train_loader)
            )

            # Evaluasi.
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    predictions = outputs.argmax(dim=1)

                    correct += (
                        predictions == labels
                    ).sum().item()
                    total += labels.size(0)

            accuracy = correct / total

            # Catat metrik untuk setiap epoch.
            mlflow.log_metric(
                "train_loss",
                average_loss,
                step=epoch,
            )
            mlflow.log_metric(
                "test_accuracy",
                accuracy,
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Loss: {average_loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

        # Simpan model ke artifact MLflow.
        example_input = torch.zeros(
            1, 1, 28, 28
        )

        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input,
            serialization_format="pickle",
        )

        print("\nTraining baseline selesai.")
        print(f"Training loss: {average_loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Dataset SHA-256: {dataset_hash}")
        print("Parameter, metrik, dan model dicatat ke MLflow.")


if __name__ == "__main__":
    main()
