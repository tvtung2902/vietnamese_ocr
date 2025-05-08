import torch
import cv2
import pathlib
from tool.config import Cfg
from tool.predictor import Predictor
from PIL import Image
import matplotlib.pyplot as plt

# Patch to allow loading Unix paths on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
yolo_model.conf = 0.4  # confidence threshold

# Load configuration for the OCR model
config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
config['device'] = 'cpu'
detector = Predictor(config)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect with YOLOv5
    results = yolo_model(frame)
    detections = results.xyxy[0]  # Tensor: [x1, y1, x2, y2, conf, cls]

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = results.names[int(cls)]

        # Convert coordinates to integers before cropping
        x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])

        # Crop the detected region for OCR
        cropped_img = frame[y1:y2, x1:x2]
        pil_img = Image.fromarray(cropped_img)

        # Perform OCR prediction on cropped image
        ocr_result = detector.predict(pil_img)
        print(f"OCR result for {label}: {ocr_result}")

        # Draw bounding box and label on the original image
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label + ": " + ocr_result, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow('YOLOv5 + OCR Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
