import torch
import cv2
import pathlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tool.config import Cfg
from tool.predictor import Predictor

# Patch cho Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
yolo_model.conf = 0.4

# Load OCR config
config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
config['device'] = 'cpu'
detector = Predictor(config)

# Load ảnh
# img_path = './sample/20160722_0202_26749_1_tg_7.png'
img_path = './sample/img_1.png'

frame = cv2.imread(img_path)

# YOLO detection
results = yolo_model(frame)
detections = results.xyxy[0]

# Convert frame to PIL for drawing
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)
draw = ImageDraw.Draw(pil_img)

# Load font
font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, size=20)

# Collect all boxes with positions
boxes = []
for det in detections:
    x1, y1, x2, y2, conf, cls = det[:6]
    label = results.names[int(cls)]
    x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
    boxes.append((y1, x1, x2, y2, label))  # sắp xếp theo y trước, x sau

# Sắp xếp các box: theo dòng (y1), sau đó theo x1 nếu cùng dòng
def sort_key(box):
    y1, x1, x2, y2, _ = box
    return (round(y1 / 10) * 10, x1)  # nhóm theo dòng (10 px)

boxes = sorted(boxes, key=sort_key)

# Xử lý từng box theo thứ tự đã sắp xếp
for y1, x1, x2, y2, label in boxes:
    cropped_img = frame[y1:y2, x1:x2]
    if cropped_img.size == 0:
        continue

    pil_crop = Image.fromarray(cropped_img)
    ocr_result = detector.predict(pil_crop)
    print(f"{label} at [{x1}, {y1}, {x2}, {y2}] → {ocr_result}")

    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
    draw.text((x1, y1 - 25), f"{label}: {ocr_result}", font=font, fill=(255, 0, 0))

# Hiển thị kết quả
final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
cv2.imshow('YOLOv5 + OCR (Sorted)', final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import torch
# import cv2
# import pathlib
# import numpy as np
# from PIL import Image, ImageFont, ImageDraw
# from tool.config import Cfg
# from tool.predictor import Predictor
# import os
# from glob import glob

# # Patch cho Windows nếu bạn đang dùng hệ điều hành này
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # Load YOLOv5 model (custom weight)
# yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)
# yolo_model.conf = 0.4

# # Load OCR config
# config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
# config['device'] = 'cpu'
# detector = Predictor(config)

# # Font để vẽ chữ lên ảnh
# font_path = "C:/Windows/Fonts/arial.ttf"
# font = ImageFont.truetype(font_path, size=20)

# # Hàm sắp xếp box theo dòng (y), sau đó theo vị trí ngang (x)
# def sort_key(box):
#     y1, x1, x2, y2, _ = box
#     return (round(y1 / 10) * 10, x1)

# # Lặp qua tất cả ảnh trong thư mục
# img_dir = 'sample/data'
# img_extensions = ('*.jpg', '*.png', '*.jpeg', '*.bmp')
# image_files = []
# for ext in img_extensions:
#     image_files.extend(glob(os.path.join(img_dir, ext)))

# # Xử lý từng ảnh
# for img_path in image_files:
#     print(f"\n🖼️ Processing: {img_path}")
#     frame = cv2.imread(img_path)
#     if frame is None:
#         print(f"⚠️ Could not read {img_path}, skipping.")
#         continue

#     # YOLO object detection
#     results = yolo_model(frame)
#     detections = results.xyxy[0]

#     # Chuyển ảnh sang PIL để dễ vẽ chữ
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(frame_rgb)
#     draw = ImageDraw.Draw(pil_img)

#     # Thu thập box
#     boxes = []
#     for det in detections:
#         x1, y1, x2, y2, conf, cls = det[:6]
#         label = results.names[int(cls)]
#         x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
#         boxes.append((y1, x1, x2, y2, label))

#     boxes = sorted(boxes, key=sort_key)

#     # OCR từng box
#     for y1, x1, x2, y2, label in boxes:
#         cropped_img = frame[y1:y2, x1:x2]
#         if cropped_img.size == 0:
#             continue

#         pil_crop = Image.fromarray(cropped_img)
#         ocr_result = detector.predict(pil_crop)
#         print(f"{label} at [{x1}, {y1}, {x2}, {y2}] → {ocr_result}")

#         draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
#         draw.text((x1, y1 - 25), f"{label}: {ocr_result}", font=font, fill=(255, 0, 0))

#     # Hiển thị kết quả
#     final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#     cv2.imshow('YOLOv5 + OCR (Sorted)', final_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
