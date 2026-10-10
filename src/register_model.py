from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from torchvision import datasets, transforms


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Fashion-MNIST-CNN"
RUN_NAME = "best-config-seed-123"
MODEL_NAME = "FashionMNIST-CNN"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Eksperimen tidak ditemukan: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"Run tidak ditemukan: {RUN_NAME}")

    run = runs[0]
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print("Run sumber:", RUN_NAME)
    print("Run ID:", run_id)
    print("Akurasi run:", run.data.metrics.get("test_accuracy"))

    # Registrasikan model dari run yang sudah menyimpan artifact model.
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    version = registered.version
    print("Model terdaftar, versi:", version)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )
    print(f"Alias champion diarahkan ke versi {version}")

    # Muat model melalui alias sesuai persyaratan tugas.
    champion_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.pytorch.load_model(champion_uri)
    model.eval()

    project_dir = Path(__file__).resolve().parents[1]
    test_set = datasets.FashionMNIST(
        root=str(project_dir / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    image, label = test_set[0]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0)).argmax(dim=1).item()

    print("URI model:", champion_uri)
    print("Label sebenarnya:", label)
    print("Hasil prediksi:", prediction)
    print("Prediksi berhasil menggunakan model champion.")


if __name__ == "__main__":
    main()