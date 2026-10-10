import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    # Subset kecil untuk uji awal agar lebih cepat.
    train_data = Subset(train_data, range(2000))
    test_data = Subset(test_data, range(500))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Fashion-MNIST-CNN")

    with mlflow.start_run(run_name="baseline-cnn-test"):
        mlflow.log_param("dataset", "Fashion-MNIST")
        mlflow.log_param("train_samples", len(train_data))
        mlflow.log_param("test_samples", len(test_data))
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 1)
        mlflow.log_param("seed", 42)

        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels.to(device)).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        average_loss = total_loss / len(train_loader)

        mlflow.log_metric("train_loss", average_loss)
        mlflow.log_metric("test_accuracy", accuracy)
        example_input = torch.zeros(1, 1, 28, 28)
        mlflow.pytorch.log_model(
    model,
    name="model",
    input_example=example_input,
    serialization_format="pickle",
)

        print(f"Training selesai. Loss: {average_loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print("Model dan metrik sudah dicatat ke MLflow.")


if __name__ == "__main__":
    main()