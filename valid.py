import os
from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Configure model and paths
model_path = r"C:\Users\user\runs\detect\train18\weights\best.pt"
image_path = r"C:\Users\user\PycharmProjects\PythonProject23\data\images\val\Harta 21.tif"

# Check if model exists, switch to ONNX if needed
if not os.path.exists(model_path):
    model_path = r"C:\Users\user\runs\detect\train18\weights\best.onnx"
    print(f"Using ONNX model: {model_path}")
else:
    print(f"Using PyTorch model: {model_path}")

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

# Use CPU for inference
device = torch.device('cpu')

# Load model
try:
    model = YOLO(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Run inference on the image with higher quality visualization
try:
    # Încarcă imaginea originală pentru a verifica dimensiunile
    original_img = cv2.imread(image_path)
    height, width = original_img.shape[:2]
    print(f"Original image dimensions: {width}x{height}")

    # Efectuează predicția
    results = model(image_path, conf=0.25)  # Poți ajusta pragul de confidență aici

    # Obține și afișează rezultatele la rezoluție mare
    for result in results:
        # Creează o imagine de plot de înaltă rezoluție
        img_with_boxes = original_img.copy()

        # Pentru fiecare detecție, desenează căsuța
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Desenează căsuța cu un contur mai gros pentru vizibilitate mai bună
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Adaugă eticheta cu clasa și confidența
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convertește pentru matplotlib (BGR la RGB)
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        # Afișează imaginea la dimensiune mare
        plt.figure(figsize=(20, 16))  # Mărește dimensiunea figurii
        plt.imshow(img_with_boxes_rgb)
        plt.axis('off')
        plt.title('Detection Results (High Quality)')
        plt.tight_layout()
        plt.show()

        # Salvează rezultatul la rezoluție înaltă
        high_res_save_path = os.path.join(os.path.dirname(image_path), "high_res_prediction.png")
        cv2.imwrite(high_res_save_path, img_with_boxes)
        print(f"High resolution results saved to {high_res_save_path}")

        # Afișează informațiile despre detecții
        print(f"Found {len(boxes)} objects")
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"Object {i + 1}: Class {cls}, Confidence: {conf:.2f}, Coords: {coords}")

except Exception as e:
    print(f"Error during inference: {e}")