# YOLO Video Tool

A web-based tool for extracting, annotating, and reviewing images from videos using YOLO models.

## Features

- Upload YOLO models
- Extract frames from videos based on YOLO detections
- Annotate extracted images with YOLO format labels
- Review and edit annotations
- Add/delete annotations and frames
- Change annotation classes

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p models videos extracted-images annotated-images
```

4. Run the application:
```bash
python main.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Upload Model**
   - Click "Upload Model" and select your YOLO model file (.pt)
   - The model will be saved in the `models` directory

2. **Extract Images**
   - Click "Extract Images from Video"
   - Select a video file
   - Choose a YOLO model
   - Set frame skip (optional)
   - Enter output folder name
   - Select classes to extract
   - Click "Start Extraction"

3. **Annotate Images**
   - Click "Annotate Extracted Images"
   - Select the extracted images folder
   - Choose a YOLO model
   - Map original classes to new class numbers
   - Click "Start Annotation"

4. **Review Annotations**
   - Click "Review/Change Annotations"
   - Select the annotated folder
   - Choose a class to review
   - Use the interface to:
     - View images and annotations
     - Change annotation classes
     - Delete annotations
     - Add new annotations
     - Delete frames

## Directory Structure

- `models/`: YOLO model files
- `videos/`: Input video files
- `extracted-images/`: Extracted frames
- `annotated-images/`: Annotated images and labels (includes modified annotations)

## Notes

- The application uses YOLO format for annotations (class x_center y_center width height)
- All coordinates are normalized to [0,1]
- Images are saved in JPG format
- Labels are saved in TXT format

## Download Example YOLO Model

You can download a sample YOLO model (yolo11x.pt) from the following link:

[yolo11x.pt (Ultralytics YOLOv8.3.0)](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

After downloading, use the **Upload Model** feature in the application to upload this file. The model will be saved in the `models/` directory (which is gitignored by default).
