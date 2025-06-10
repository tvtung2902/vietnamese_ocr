import torch
import cv2
import pathlib
from tool.config import Cfg
from tool.predictor import Predictor
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# Patch để hỗ trợ đường dẫn Unix trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load mô hình YOLOv5 và OCR config chỉ một lần (nên đặt bên ngoài hàm để không load lại mỗi lần gọi)
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
yolo_model.conf = 0.4

config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
config['device'] = 'cpu'
detector = Predictor(config)

font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, size=20)

def process_image_with_yolo_ocr(pil_image: Image.Image):
    # Convert PIL image to OpenCV format (BGR)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # YOLO detection
    results = yolo_model(frame)
    detections = results.xyxy[0]

    # Convert frame to RGB PIL for drawing
    pil_img = pil_image.copy()
    draw = ImageDraw.Draw(pil_img)

    # Process detections
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = results.names[int(cls)]

        x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
        cropped_img = frame[y1:y2, x1:x2]
        
        pil_crop = Image.fromarray(cropped_img)
        ocr_result = detector.predict(pil_crop)
        print(f"OCR result for {label} at [{x1}, {y1}, {x2}, {y2}]: {ocr_result}")

        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        draw.text((x1, y1 - 25), f"{label}: {ocr_result}", font=font, fill=(0, 255, 0))

    # Convert back to OpenCV to show or return
    final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return final_frame