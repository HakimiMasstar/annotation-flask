import os
import cv2
from ultralytics import YOLO

class ImageExtractor:
    def __init__(self):
        self.videos_dir = 'videos'
        self.models_dir = 'models'
        self.extracted_dir = 'extracted-images'

    def process(self, video_path, model_path, selected_classes, frame_skip=1, folder_name=None):
        """
        Process a video file and extract frames based on YOLO detections
        
        Args:
            video_path (str): Path to the video file
            model_path (str): Path to the YOLO model file
            selected_classes (list): List of class indices to extract
            frame_skip (int): Number of frames to skip between processing
            folder_name (str): Name of the output folder
            
        Returns:
            dict: Result containing status and message
        """
        try:
            # Validate inputs
            if not os.path.exists(video_path):
                return {'error': 'Video file not found'}
            if not os.path.exists(model_path):
                return {'error': 'Model file not found'}
            if not selected_classes:
                return {'error': 'No classes selected'}
            if not folder_name:
                return {'error': 'No folder name provided'}
            if any(c in folder_name for c in r'<>:"/\\|?*'):
                return {'error': 'Folder name contains invalid characters'}

            # Create output directory
            out_folder = os.path.join(self.extracted_dir, folder_name)
            if os.path.exists(out_folder):
                return {'error': f'Folder "{folder_name}" already exists'}

            os.makedirs(out_folder, exist_ok=False)

            # Load model
            model = YOLO(model_path)

            # Process video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            saved = 0
            img_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    results = model(frame)
                    for r in results:
                        # Convert class indices to integers for comparison
                        detected_classes = [int(cls) for cls in r.boxes.cls]
                        # Check if any of the selected classes are in the detected classes
                        if any(cls in detected_classes for cls in map(int, selected_classes)):
                            out_path = os.path.join(out_folder, f'{folder_name}_{img_num}.jpg')
                            cv2.imwrite(out_path, frame)
                            saved += 1
                            img_num += 1
                            break

                frame_idx += 1
                # Print progress every 100 frames
                if frame_idx % 100 == 0:
                    print(f"Processing frame {frame_idx}/{frame_count}")

            cap.release()
            return {
                'success': True,
                'message': f'Extraction complete! {saved} images saved to {out_folder}',
                'saved_count': saved,
                'output_folder': out_folder
            }

        except Exception as e:
            return {'error': f'Error during extraction: {str(e)}'} 