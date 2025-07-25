import os
import glob
import json
from PIL import Image, ImageDraw

class AnnotationReviewer:
    def __init__(self):
        self.annotated_dir = 'annotated-images'

    def get_class_counts(self, folder_name):
        """Get class counts for a folder"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        labels_dir = os.path.join(folder_path, 'labels')
        
        if not os.path.exists(labels_dir):
            return {}
            
        class_counts = {}
        for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = parts[0]
                        class_counts[cls] = class_counts.get(cls, 0) + 1
        return class_counts

    def get_images(self, folder_name, class_names):
        """Get images and annotations for specified classes"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        images_dir = os.path.join(folder_path, 'images')
        labels_dir = os.path.join(folder_path, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            return {'error': 'Images or labels directory not found'}
            
        images_data = []
        for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
            img_name = os.path.splitext(os.path.basename(label_file))[0] + '.jpg'
            img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            annotations = []
            with open(label_file) as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, bw, bh = parts
                        if cls in class_names:
                            annotations.append({
                                'id': idx,
                                'class': cls,
                                'x': float(x),
                                'y': float(y),
                                'width': float(bw),
                                'height': float(bh)
                            })
            
            if annotations:  # Only include images that have annotations for selected classes
                images_data.append({
                    'image_name': img_name,
                    'image_path': img_path,
                    'label_file': label_file,
                    'annotations': annotations
                })
        
        return {
            'success': True,
            'images': images_data
        }

    def update_annotation(self, folder_name, image_name, annotation_id, new_class):
        """Update annotation class"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        label_file = os.path.join(folder_path, 'labels', os.path.splitext(image_name)[0] + '.txt')
        
        if not os.path.exists(label_file):
            return {'error': 'Label file not found'}
            
        try:
            annotation_id = int(annotation_id)
            new_lines = []
            with open(label_file) as f:
                for idx, line in enumerate(f):
                    if idx == annotation_id:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = new_class
                            new_lines.append(' '.join(parts))
                        else:
                            new_lines.append(line.strip())
                    else:
                        new_lines.append(line.strip())
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(new_lines))
            
            return {'success': True, 'message': 'Annotation updated successfully'}
        except Exception as e:
            return {'error': f'Error updating annotation: {str(e)}'}

    def update_annotation_box(self, folder_name, image_name, annotation_id, box_data):
        """Update annotation box coordinates"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        label_file = os.path.join(folder_path, 'labels', os.path.splitext(image_name)[0] + '.txt')
        
        if not os.path.exists(label_file):
            return {'error': 'Label file not found'}
            
        try:
            # Parse box data (should be in YOLO format: x_center, y_center, width, height)
            box_parts = box_data.split(',')
            if len(box_parts) != 4:
                return {'error': 'Invalid box data format'}
            
            x_center, y_center, width, height = map(float, box_parts)
            
            # Validate coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                return {'error': 'Box coordinates must be normalized (0-1)'}
            
            annotation_id = int(annotation_id)
            new_lines = []
            with open(label_file) as f:
                for idx, line in enumerate(f):
                    if idx == annotation_id:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Keep the class but update the box coordinates
                            class_name = parts[0]
                            new_lines.append(f'{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
                        else:
                            new_lines.append(line.strip())
                    else:
                        new_lines.append(line.strip())
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(new_lines))
            
            return {'success': True, 'message': 'Annotation box updated successfully'}
        except Exception as e:
            return {'error': f'Error updating annotation box: {str(e)}'}

    def delete_annotation(self, folder_name, image_name, annotation_id):
        """Delete an annotation"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        label_file = os.path.join(folder_path, 'labels', os.path.splitext(image_name)[0] + '.txt')
        
        if not os.path.exists(label_file):
            return {'error': 'Label file not found'}
            
        try:
            annotation_id = int(annotation_id)
            new_lines = []
            with open(label_file) as f:
                for idx, line in enumerate(f):
                    if idx != annotation_id:
                        new_lines.append(line.strip())
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(new_lines))
            
            return {'success': True, 'message': 'Annotation deleted successfully'}
        except Exception as e:
            return {'error': f'Error deleting annotation: {str(e)}'}

    def add_annotation(self, folder_name, image_name, box_data, class_name):
        """Add a new annotation"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        label_file = os.path.join(folder_path, 'labels', os.path.splitext(image_name)[0] + '.txt')
        image_path = os.path.join(folder_path, 'images', image_name)
        
        if not os.path.exists(label_file):
            return {'error': 'Label file not found'}
        if not os.path.exists(image_path):
            return {'error': 'Image file not found'}
            
        try:
            # Parse box data (should be in YOLO format: x_center, y_center, width, height)
            box_parts = box_data.split(',')
            if len(box_parts) != 4:
                return {'error': 'Invalid box data format'}
            
            x_center, y_center, width, height = map(float, box_parts)
            
            # Validate coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                return {'error': 'Box coordinates must be normalized (0-1)'}
            
            # Add to label file
            with open(label_file, 'a') as f:
                f.write(f'\n{class_name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
            
            return {'success': True, 'message': 'Annotation added successfully'}
        except Exception as e:
            return {'error': f'Error adding annotation: {str(e)}'}

    def delete_frame(self, folder_name, image_name):
        """Delete an entire frame (image and label)"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        image_path = os.path.join(folder_path, 'images', image_name)
        label_file = os.path.join(folder_path, 'labels', os.path.splitext(image_name)[0] + '.txt')
        
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(label_file):
                os.remove(label_file)
            
            return {'success': True, 'message': 'Frame deleted successfully'}
        except Exception as e:
            return {'error': f'Error deleting frame: {str(e)}'}

    def get_annotated_image(self, folder_name, image_name, selected_classes=None):
        """Get image with annotations drawn on it"""
        folder_path = os.path.join(self.annotated_dir, folder_name)
        image_path = os.path.join(folder_path, 'images', image_name)
        label_file = os.path.join(folder_path, 'labels', os.path.splitext(image_name)[0] + '.txt')
        
        if not os.path.exists(image_path):
            return {'error': 'Image file not found'}
            
        try:
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            annotations = []
            with open(label_file) as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, bw, bh = parts
                        if selected_classes is None or cls in selected_classes:
                            x, y, bw, bh = float(x), float(y), float(bw), float(bh)
                            x1 = (x - bw/2) * w
                            y1 = (y - bh/2) * h
                            x2 = (x + bw/2) * w
                            y2 = (y + bh/2) * h
                            
                            color = 'red' if selected_classes and cls in selected_classes else 'blue'
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw.text((x1, y1 - 12), f"ID {idx}", fill=color)
                            
                            annotations.append({
                                'id': idx,
                                'class': cls,
                                'x': x, 'y': y, 'width': bw, 'height': bh,
                                'bbox': [x1, y1, x2, y2]
                            })
            
            # Save annotated image directly to the annotated-images folder
            output_path = os.path.join(folder_path, 'images', f"review_{image_name}")
            img.save(output_path)
            
            return {
                'success': True,
                'annotated_image_path': output_path,
                'annotations': annotations
            }
        except Exception as e:
            return {'error': f'Error processing image: {str(e)}'} 