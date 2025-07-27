from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import pathlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import io
import time
from functools import wraps

from tool.config import Cfg
from tool.predictor import Predictor

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app)

class ModelLoader:
    def __init__(self):
        self.yolo_model = None
        self.detector = None
        self.font = None
        self.device = None
        self.load_time = None
        self.initialize_models()

    def initialize_models(self):
        start_time = time.time()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n🔧 Đang khởi tạo mô hình trên {self.device.upper()}...")

        try:
            self.yolo_model = torch.hub.load(
                'ultralytics/yolov5', 
                'custom', 
                path='./yolov5/weight-detection.pt',
                force_reload=False
            ).to(self.device)
            self.yolo_model.conf = 0.4
            self.yolo_model.eval()
        except Exception as e:
            raise

        # Tải OCR model
        try:
            config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
            config['device'] = self.device
            print(config)
            self.detector = Predictor(config)
        except Exception as e:
            raise

        # Tải font
        font_path = "C:/Windows/Fonts/arial.ttf"
        try:
            self.font = ImageFont.truetype(font_path, size=20)
        except IOError:
            self.font = ImageFont.load_default()

        self.load_time = time.time() - start_time

model_loader = ModelLoader()

def sort_key(box_info):
    y1, x1, x2, y2, label, cropped_img = box_info
    return (round(y1 / 10) * 10, x1)

def handle_errors(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return wrapped

@app.route('/health', methods=['GET'])
def health_check():
    """Kiểm tra trạng thái server và mô hình"""
    status = {
        "status": "running",
        "models_loaded": bool(model_loader.yolo_model and model_loader.detector),
        "device": model_loader.device,
        "load_time": model_loader.load_time
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    start_time = time.time()
    image_stream = io.BytesIO(file.read())
    pil_img = Image.open(image_stream).convert('RGB')
    frame = np.array(pil_img)
    frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        results = model_loader.yolo_model(frame_cv2)
    
    detections = results.xyxy[0]
    data_to_sort = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = model_loader.yolo_model.names[int(cls)]
        x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])

        cropped_img = frame_cv2[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        data_to_sort.append((y1, x1, x2, y2, label, cropped_img))

    sorted_data = sorted(data_to_sort, key=sort_key)
    cropped_images = [Image.fromarray(item[5]) for item in sorted_data]
    results_list = []

    if cropped_images:
        ocr_results = model_loader.detector.predict_batch(cropped_images)
        
        for (y1, x1, x2, y2, label, _), text in zip(sorted_data, ocr_results):
            # results_list.append({
            #     "label": label,
            #     "confidence": float(conf.item()),
            #     "bbox": [x1, y1, x2, y2],
            #     "text": text
            # })
            results_list.append(text)

    processing_time = time.time() - start_time
    response = {
        "texts": results_list,
        "processing_time": f"{processing_time:.3f}s",
        "objects_detected": len(results_list)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, threaded=True)