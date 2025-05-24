# Vietnamese-OCR

**Vietnamese-OCR** is an advanced Optical Character Recognition (OCR) system designed specifically to accurately extract Vietnamese text from images. Leveraging a powerful multi-stage pipeline, this project combines state-of-the-art deep learning techniques in image preprocessing, object detection, and sequence recognition to deliver high accuracy on challenging Vietnamese scripts.

---

## Project Overview

Recognizing Vietnamese text from images is a complex task due to the language’s unique diacritics and character combinations. To tackle this, Vietnamese-OCR employs a carefully designed pipeline consisting of three key components:

1. **Image Preprocessing:**  
   The input images undergo a series of preprocessing steps such as grayscale conversion, noise reduction, contrast enhancement, and normalization. This step ensures that the input data is clean and optimized for subsequent model inference, improving overall recognition robustness.

2. **Text Region Detection with YOLOv5:**  
   Utilizing the state-of-the-art YOLOv5 object detection framework, the system identifies precise bounding boxes around text regions or individual characters within the image. YOLOv5 is chosen for its speed and accuracy, enabling real-time or near real-time detection.

3. **Text Recognition using VGG + Transformer:**  
   Once text regions are detected, each region is processed by a Convolutional Neural Network based on the VGG architecture to extract meaningful visual features. These features are then fed into a Transformer-based sequence model that decodes the features into Vietnamese text sequences, effectively handling complex diacritics and variable-length text.

---

## Pipeline Breakdown

| Stage                | Description                                                                                      |
|----------------------|------------------------------------------------------------------------------------------------|
| **Preprocessing**    | Image enhancement techniques including grayscale conversion, noise filtering, and normalization.|
| **Object Detection** | YOLOv5 detects bounding boxes of text lines or characters, ensuring accurate localization.      |
| **Text Recognition** | Feature extraction with VGG and sequence decoding using Transformer for precise transcription.   |

---

## Features

- **Robust to varied image quality:** Handles images with different lighting, resolution, and noise levels.  
- **Accurate Vietnamese text recognition:** Designed to handle Vietnamese-specific characters and diacritics.  
- **End-to-end deep learning pipeline:** From raw images to decoded text strings.  
- **Modular architecture:** Each stage can be improved or replaced independently.  
- **Open-source and extensible:** Easy to adapt for related OCR tasks or languages.

---

## Installation

To get started with Vietnamese-OCR, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/vietnamese-ocr.git
cd vietnamese-ocr
pip install -r requirements.txt

