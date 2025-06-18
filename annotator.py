import os
import cv2
from ultralytics import YOLO
import json
import numpy as np

class ImageAnnotator:
    def __init__(self):
        self.extracted_dir = 'extracted-images'
        self.annotated_dir = 'annotated-images'

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes in YOLO format (x_center, y_center, width, height)
        
        Args:
            box1: (x_center, y_center, width, height)
            box2: (x_center, y_center, width, height)
            
        Returns:
            float: IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to corner coordinates
        x1_min, x1_max = x1 - w1/2, x1 + w1/2
        y1_min, y1_max = y1 - h1/2, y1 + h1/2
        x2_min, x2_max = x2 - w2/2, x2 + w2/2
        y2_min, y2_max = y2 - h2/2, y2 + h2/2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def filter_overlapping_detections(self, boxes, classes, iou_threshold=0.5):
        """
        Filter out overlapping detections with high IoU
        
        Args:
            boxes: List of (x, y, w, h) boxes
            classes: List of class labels
            iou_threshold: IoU threshold for overlap detection
            
        Returns:
            tuple: (filtered_boxes, filtered_classes)
        """
        if len(boxes) <= 1:
            return boxes, classes
        
        # Convert to list for easier manipulation
        boxes = list(boxes)
        classes = list(classes)
        
        # Sort by area (larger boxes first) to prioritize them
        areas = [w * h for _, _, w, h in boxes]
        sorted_indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
        
        filtered_indices = []
        
        for i in sorted_indices:
            should_keep = True
            
            # Check overlap with already kept boxes
            for kept_idx in filtered_indices:
                iou = self.calculate_iou(boxes[i], boxes[kept_idx])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                filtered_indices.append(i)
        
        # Return filtered results in original order
        filtered_boxes = [boxes[i] for i in sorted(filtered_indices)]
        filtered_classes = [classes[i] for i in sorted(filtered_indices)]
        
        return filtered_boxes, filtered_classes

    def process(self, folder_name, model_path, class_mappings, iou_threshold=0.5):
        """
        Process extracted images and create YOLO format annotations
        
        Args:
            folder_name (str): Name of the folder containing extracted images
            model_path (str): Path to the YOLO model file
            class_mappings (str): JSON string of class mappings (only for selected classes)
            iou_threshold (float): IoU threshold for overlap filtering (default: 0.5)
            
        Returns:
            dict: Result containing status and message
        """
        try:
            # Validate inputs
            if not os.path.exists(model_path):
                return {'error': 'Model file not found'}
            if not folder_name:
                return {'error': 'No folder name provided'}

            # Setup paths
            input_folder = os.path.join(self.extracted_dir, folder_name)
            if not os.path.exists(input_folder):
                return {'error': f'Input folder "{folder_name}" not found'}

            out_folder = os.path.join(self.annotated_dir, folder_name)
            images_out = os.path.join(out_folder, 'images')
            labels_out = os.path.join(out_folder, 'labels')
            os.makedirs(images_out, exist_ok=True)
            os.makedirs(labels_out, exist_ok=True)

            # Load model
            model = YOLO(model_path)

            # Parse class mappings
            try:
                class_map = json.loads(class_mappings)
                # Convert string keys to integers
                class_map = {int(k): int(v) for k, v in class_map.items()}
            except json.JSONDecodeError:
                return {'error': 'Invalid class mappings format'}
            except ValueError:
                return {'error': 'Invalid class number format'}

            if not class_map:
                return {'error': 'No classes selected for annotation'}

            # Process images
            image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files.sort()
            total_images = len(image_files)
            img_num = 0
            total_filtered = 0

            for idx, img_name in enumerate(image_files):
                if idx % 10 == 0:  # Print progress every 10 images
                    print(f"Processing image {idx + 1}/{total_images}")
                    
                img_path = os.path.join(input_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_name}")
                    continue

                results = model(img)
                label_boxes = []
                label_classes = []

                for r in results:
                    for box, cls in zip(r.boxes.xywh, r.boxes.cls):
                        cls = int(cls)
                        # Only process if class is in selected mappings
                        if cls in class_map:
                            x, y, w, h = box
                            label_boxes.append((x, y, w, h))
                            # Use mapped class number
                            label_classes.append(class_map[cls])

                # Filter overlapping detections
                original_count = len(label_boxes)
                if label_boxes:
                    label_boxes, label_classes = self.filter_overlapping_detections(
                        label_boxes, label_classes, iou_threshold
                    )
                    filtered_count = original_count - len(label_boxes)
                    total_filtered += filtered_count
                    
                    if filtered_count > 0:
                        print(f"Filtered {filtered_count} overlapping detections in {img_name}")

                if label_boxes:
                    h_img, w_img = img.shape[:2]
                    label_lines = []
                    for cls, (x, y, w, h) in zip(label_classes, label_boxes):
                        label_lines.append(f"{cls} {x/w_img:.6f} {y/h_img:.6f} {w/w_img:.6f} {h/h_img:.6f}")

                    out_img_name = f"{folder_name}_{img_num}.jpg"
                    cv2.imwrite(os.path.join(images_out, out_img_name), img)
                    label_name = os.path.splitext(out_img_name)[0] + '.txt'
                    with open(os.path.join(labels_out, label_name), 'w') as f:
                        f.write('\n'.join(label_lines))
                    img_num += 1

                # Delete the original image after processing
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Warning: Could not delete {img_path}: {e}")

            # Clean up empty input folder
            try:
                if os.path.isdir(input_folder):
                    os.rmdir(input_folder)
            except Exception as e:
                print(f"Warning: Could not delete input folder: {e}")

            return {
                'success': True,
                'message': f'Annotation complete! {img_num} images and labels saved to {out_folder}. Filtered {total_filtered} overlapping detections.',
                'processed_count': img_num,
                'filtered_count': total_filtered,
                'output_folder': out_folder
            }

        except Exception as e:
            return {'error': f'Error during annotation: {str(e)}'} 