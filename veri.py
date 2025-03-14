import os
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from datetime import datetime

# Configure model and paths
model_path = r"C:\Users\user\runs\detect\train18\weights\best.pt"
image_path = r"C:\Users\user\PycharmProjects\PythonProject23\data\images\val\Harta 57.tif"

# Folder pentru imagini segmentate
output_folder = r"C:\Users\user\PycharmProjects\PythonProject23\segmented_regions"
os.makedirs(output_folder, exist_ok=True)

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

# Run inference and extract segments
try:
    # Încarcă imaginea originală
    original_img = cv2.imread(image_path)
    if original_img is None:
        # Încearcă să citești imaginea TIFF cu alt method
        from PIL import Image

        pil_img = Image.open(image_path)
        original_img = np.array(pil_img.convert('RGB'))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    height, width = original_img.shape[:2]
    print(f"Original image dimensions: {width}x{height}")

    # Obține numele fișierului pentru etichetare
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Efectuează predicția
    results = model(image_path, conf=0.25)  # Poți ajusta pragul de confidență

    # Creează un fișier CSV pentru a salva metadatele
    csv_path = os.path.join(output_folder, f"{base_filename}_segments_info.csv")
    with open(csv_path, 'w') as f:
        f.write("segment_id,class_id,confidence,x1,y1,x2,y2,width,height,timestamp\n")

    # Pentru fiecare detecție, extrage și salvează segmentul
    segments_count = 0
    for result in results:
        boxes = result.boxes
        print(f"Found {len(boxes)} segments")

        for i, box in enumerate(boxes):
            # Obține coordonatele cutiei de încadrare
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Adaugă padding opțional (mărește zona pentru context)
            padding = 10
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(width, x2 + padding)
            y2_pad = min(height, y2 + padding)

            # Extrage segmentul din imaginea originală
            segment = original_img[y1_pad:y2_pad, x1_pad:x2_pad]

            # Creează un nume de fișier informativ
            segment_filename = f"{base_filename}_class{cls}_conf{conf:.2f}_seg{i + 1}.png"
            segment_path = os.path.join(output_folder, segment_filename)

            # Salvează segmentul
            cv2.imwrite(segment_path, segment)
            segments_count += 1

            # Salvează metadatele în CSV
            segment_width = x2_pad - x1_pad
            segment_height = y2_pad - y1_pad
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(csv_path, 'a') as f:
                f.write(f"{i + 1},{cls},{conf:.4f},{x1_pad},{y1_pad},{x2_pad},{y2_pad},"
                        f"{segment_width},{segment_height},{timestamp}\n")

            print(f"Saved segment {i + 1} to {segment_path}")

    # Crează și o imagine cu toate căsuțele marcate pentru referință
    marked_img = original_img.copy()
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Desenează căsuța și adaugă text
            cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(marked_img, f"ID:{i + 1} C:{cls} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Salvează imaginea cu marcaje
    marked_img_path = os.path.join(output_folder, f"{base_filename}_all_segments.png")
    cv2.imwrite(marked_img_path, marked_img)

    print(f"\nProcesare finalizată:")
    print(f"- {segments_count} segmente salvate în {output_folder}")
    print(f"- Informații despre segmente salvate în {csv_path}")
    print(f"- Imagine cu toate segmentele marcate: {marked_img_path}")

except Exception as e:
    print(f"Error during processing: {e}")