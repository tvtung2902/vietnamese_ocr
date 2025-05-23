# Vietnamese-OCR

Vietnamese-OCR is an OCR system designed to recognize Vietnamese text from images. The system uses a three-stage pipeline combining image preprocessing, object detection, and sequence recognition for high accuracy on Vietnamese scripts.

---

## Pipeline Overview

| Step                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| **Preprocessing**    | Clean and enhance input images (grayscale, noise removal, etc.) |
| **Text Detection**   | Use YOLOv5 to locate text regions or characters in the image     |
| **Text Recognition** | Apply VGG + Transformer to recognize and decode Vietnamese text  |

---

## Installation

To set up the project, clone the repo and install dependencies:

```bash
git clone https://github.com/tvtung2902/vietnamese-ocr.git
cd vietnamese-ocr
pip install -r requirements.txt
