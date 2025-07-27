# 🇻🇳 Vietnamese OCR System with YOLOv5 + VietOCR

This project is a complete Vietnamese OCR pipeline that **combines object detection and sequence modeling** to detect and recognize text in images. The system uses **YOLOv5** for text region detection and **VietOCR** (Transformer-based) for recognizing the text content.

---

## Overview of the System

The pipeline consists of two main stages:

1. **Text Detection** (YOLOv5):
   - We use **YOLOv5** (You Only Look Once version 5) as the **text detector**.
   - It is a fast and accurate real-time object detection algorithm.
   - The model is trained (or fine-tuned) to detect text areas in input images and return bounding boxes around them.
   - These cropped regions are then passed to the next stage for recognition.

2. **Text Recognition** (VietOCR):
   - We use **VietOCR**, a deep learning-based text recognizer optimized for Vietnamese.
   - VietOCR architecture consists of two main components:
     - **VGG-style CNN** for **feature extraction**: Converts input images into visual feature sequences.
     - **Transformer encoder-decoder**: Translates the visual features into Vietnamese character sequences.
   - Unlike traditional OCRs using CTC, VietOCR uses **Seq2Seq with attention**, making it highly accurate for Vietnamese with diacritics and complex scripts.

---

### References
- [VietOCR - quanpn90/VietOCR](https://github.com/quanpn90/VietOCR)
- [YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5)
