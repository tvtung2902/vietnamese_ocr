import torch
import cv2
import pathlib

# Patch to allow loading Unix paths on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/weight-detection.pt', force_reload=True)

yolo_model.conf = 0.4  # confidence threshold

# ----- Open webcam -----
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

        # Draw bounding box and label
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow('YOLOv5 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()