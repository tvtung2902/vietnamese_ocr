import torch
import cv2
import pathlib
from tool.config import Cfg
from tool.predictor import Predictor
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# Patch to allow loading Unix paths on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
yolo_model.conf = 0.4  # confidence threshold

# Load OCR configuration
config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
config['device'] = 'cpu'
detector = Predictor(config)

# Load image
img_path = './sample/1.png'
frame = cv2.imread(img_path)

# Detect objects with YOLO
results = yolo_model(frame)
detections = results.xyxy[0]

# Convert frame to PIL for Unicode rendering
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)
draw = ImageDraw.Draw(pil_img)

# Load a Unicode font (ensure this path is valid on your system)
font_path = "C:/Windows/Fonts/arial.ttf"  # You can change this to a font that supports Vietnamese
font = ImageFont.truetype(font_path, size=20)

# Process each detection
for det in detections:
    x1, y1, x2, y2, conf, cls = det[:6]
    label = results.names[int(cls)]

    x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
    cropped_img = frame[y1:y2, x1:x2]
    
    # OCR prediction
    pil_crop = Image.fromarray(cropped_img)
    ocr_result = detector.predict(pil_crop)
    print(f"OCR result for {label} at [{x1}, {y1}, {x2}, {y2}]: {ocr_result}")

    # Draw bounding box
    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
    draw.text((x1, y1 - 25), f"{label}: {ocr_result}", font=font, fill=(0, 255, 0))

# Convert back to OpenCV image
final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Show result
cv2.imshow('YOLOv5 + OCR (Unicode)', final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
