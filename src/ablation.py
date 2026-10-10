import json
import random
import statistics
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED = 42
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.0027792992750462365
BEST_DROPOUT = 0.24444525375713033
DEVICE = torch.device("cpu")


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


def run_experiment(name, dropout, augmentation):
    random.seed(SEED)
    torch.manual_seed(SEED)

    transform_list = []
    if augmentation:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    train_set = datasets.FashionMNIST(
        root=str(PROJECT_DIR / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.FashionMNIST(
        root=str(PROJECT_DIR / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_set = Subset(train_set, range(2000))
    test_set = Subset(test_set, range(500))
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = SimpleCNN(dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": dropout,
            "augmentation": augmentation,
            "train_samples": len(train_set),
            "test_samples": len(test_set),
        })
        mlflow.set_tag("stage", "ablation")

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            mlflow.log_metric(
                "train_loss", total_loss / len(train_loader), step=epoch
            )

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                predictions = model(images.to(DEVICE)).argmax(dim=1)
                correct += (predictions == labels.to(DEVICE)).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"{name}: test_accuracy={accuracy:.4f}")
        return accuracy


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fashion-MNIST-CNN")

    results = {}
    results["baseline"] = run_experiment(
        "ablation-baseline", BEST_DROPOUT, False
    )
    results["no_dropout"] = run_experiment(
        "ablation-no-dropout", 0.0, False
    )
    results["with_augmentation"] = run_experiment(
        "ablation-with-augmentation", BEST_DROPOUT, True
    )

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / "ablation_summary.json"
    output_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\nRINGKASAN ABLATION")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")
    print("Ringkasan disimpan di:", output_path)


if __name__ == "__main__":
    main()