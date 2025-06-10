import torch
import cv2
import pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tool.config import Cfg
from tool.predictor import Predictor

# Patch pathlib for compatibility on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
yolo_model.conf = 0.4  # Confidence threshold

# Load OCR config
config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
config['device'] = 'cpu'
print(config)
detector = Predictor(config)

# Load font for Vietnamese text
font_path = "C:/Windows/Fonts/arial.ttf"  # Replace if needed
font = ImageFont.truetype(font_path, size=20)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    results = yolo_model(frame)
    detections = results.xyxy[0]

    # Convert frame to RGB and create PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = results.names[int(cls)]
        x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])

        # Crop region and apply OCR
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        pil_crop = Image.fromarray(cropped)
        ocr_result = detector.predict(pil_crop)
        print(f"OCR result for {label}: {ocr_result}")

        # Draw rectangle and OCR text on image
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        draw.text((x1, y1 - 25), f"{label}: {ocr_result}", font=font, fill=(0, 255, 0))

    # Convert back to BGR for OpenCV display
    final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 + OCR (Unicode)', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
