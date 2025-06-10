from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2
import numpy as np
import torch
import pathlib
from tool.config import Cfg
from tool.predictor import Predictor
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
yolo_model.conf = 0.4

config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
config['device'] = 'cpu'
detector = Predictor(config)

def extract_text_from_image(pil_image: Image.Image):
    # Chuyển ảnh từ PIL -> OpenCV (RGB -> BGR)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Dự đoán đối tượng với YOLO
    results = yolo_model(frame)
    detections = results.xyxy[0]

    detected_items = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
        cropped_img = frame[y1:y2, x1:x2]

        # Dự đoán OCR trên vùng cắt
        pil_crop = Image.fromarray(cropped_img)
        ocr_result = detector.predict(pil_crop)

        # Thêm kết quả vào danh sách với vị trí để sắp xếp
        detected_items.append((y1, x1, ocr_result))  # y1 trước để ưu tiên sắp xếp theo chiều dọc

    # Sắp xếp theo: từ trên xuống dưới (y1), và trong cùng một dòng thì từ trái sang phải (x1)
    detected_items.sort(key=lambda item: (item[0] // 10, item[1]))  # //10 để gom nhóm các dòng gần nhau

    # Trả về danh sách chỉ chứa văn bản, đã được sắp xếp đúng thứ tự
    texts = [item[2] for item in detected_items]
    return texts


# --- FastAPI Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_data = await file.read()
    pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")

    texts = extract_text_from_image(pil_image)
    return JSONResponse(content={"texts": texts})