import os
from ultralytics import YOLO
import torch

torch.multiprocessing.set_start_method('spawn', force=True)

if __name__ == "__main__":
    # Încarcă un model preantrenat sau începe de la zero
    model = YOLO("yolov8n.pt")  # model preantrenat
    # model = YOLO("yolov8n.yaml")  # model nou

    # Verifică dacă GPU este disponibil
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Antrenează modelul
    results = model.train(
        data="data.yaml",  # calea către fișierul data.yaml
        epochs=10,
        imgsz=640,
        batch=8,
        workers=4,
        device="0" if torch.cuda.is_available() else "cpu"  # folosește GPU dacă este disponibil
    )

    # Evaluează modelul pe setul de validare
    results = model.val()

    # Salvează modelul
    model.export(format="onnx")


#pentru maine la ora 13