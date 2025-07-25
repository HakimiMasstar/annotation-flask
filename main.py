from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import shutil
from extractor import ImageExtractor
from annotator import ImageAnnotator
from reviewer import AnnotationReviewer
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max file size

# Ensure required directories exist
for dir_name in ['uploads', 'models', 'videos', 'extracted-images', 'annotated-images']:
    os.makedirs(dir_name, exist_ok=True)

def secure_filename(filename):
    """Basic filename sanitization"""
    # Remove any directory components
    filename = os.path.basename(filename)
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_model_classes')
def get_model_classes():
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({'error': 'No model specified'}), 400
        
    model_path = os.path.join('models', model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
        
    try:
        model = YOLO(model_path)
        class_names = model.names.values() if hasattr(model.names, 'values') else model.names
        return jsonify({
            'success': True,
            'classes': list(class_names)
        })
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500

@app.route('/get_folder_classes')
def get_folder_classes():
    folder_name = request.args.get('folder')
    if not folder_name:
        return jsonify({'error': 'No folder specified'}), 400
        
    folder_path = os.path.join('annotated-images', folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404
        
    try:
        # Get all unique class numbers from label files
        classes = set()
        labels_dir = os.path.join(folder_path, 'labels')
        if os.path.exists(labels_dir):
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                classes.add(parts[0])
        
        return jsonify({
            'success': True,
            'classes': sorted(list(classes), key=lambda x: int(x) if x.isdigit() else x)
        })
    except Exception as e:
        return jsonify({'error': f'Error loading classes: {str(e)}'}), 500

@app.route('/get_folder_class_counts')
def get_folder_class_counts():
    folder_name = request.args.get('folder')
    if not folder_name:
        return jsonify({'error': 'No folder specified'}), 400
        
    folder_path = os.path.join('annotated-images', folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404
        
    try:
        reviewer = AnnotationReviewer()
        class_counts = reviewer.get_class_counts(folder_name)
        
        return jsonify({
            'success': True,
            'class_counts': class_counts
        })
    except Exception as e:
        return jsonify({'error': f'Error loading class counts: {str(e)}'}), 500

@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        if 'model' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['model']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('models', filename))
            return jsonify({'message': f'Model uploaded successfully: {filename}'})
    return render_template('upload_model.html')

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        video = request.files['video']
        model_name = request.form.get('model')
        classes = request.form.getlist('classes[]')
        frame_skip = int(request.form.get('frame_skip', 1))
        folder_name = request.form.get('folder_name')
        
        if not all([video.filename, model_name, classes, folder_name]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Save video
        video_path = os.path.join('videos', secure_filename(video.filename))
        video.save(video_path)
        
        # Initialize extractor and process
        extractor = ImageExtractor()
        result = extractor.process(
            video_path=video_path,
            model_path=os.path.join('models', model_name),
            selected_classes=classes,
            frame_skip=frame_skip,
            folder_name=folder_name
        )
        
        return jsonify(result)
        
    # GET request - show form
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('extract.html', models=models)

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if request.method == 'POST':
        folder_name = request.form.get('folder_name')
        model_name = request.form.get('model')
        class_mappings = request.form.get('class_mappings')
        iou_threshold = float(request.form.get('iou_threshold', 0.5))
        
        if not all([folder_name, model_name, class_mappings]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Initialize annotator and process
        annotator = ImageAnnotator()
        result = annotator.process(
            folder_name=folder_name,
            model_path=os.path.join('models', model_name),
            class_mappings=class_mappings,
            iou_threshold=iou_threshold
        )
        
        return jsonify(result)
        
    # GET request - show form
    folders = [f for f in os.listdir('extracted-images') if os.path.isdir(os.path.join('extracted-images', f))]
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('annotate.html', folders=folders, models=models)

@app.route('/review', methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        action = request.form.get('action')
        folder_name = request.form.get('folder_name')
        
        if not folder_name:
            return jsonify({'error': 'Missing folder name'}), 400
            
        reviewer = AnnotationReviewer()
        
        if action == 'get_images':
            class_names = request.form.getlist('class_names[]')
            return jsonify(reviewer.get_images(folder_name, class_names))
        elif action == 'update_annotation':
            image_name = request.form.get('image_name')
            annotation_id = request.form.get('annotation_id')
            new_class = request.form.get('new_class')
            return jsonify(reviewer.update_annotation(folder_name, image_name, annotation_id, new_class))
        elif action == 'delete_annotation':
            image_name = request.form.get('image_name')
            annotation_id = request.form.get('annotation_id')
            return jsonify(reviewer.delete_annotation(folder_name, image_name, annotation_id))
        elif action == 'add_annotation':
            image_name = request.form.get('image_name')
            box_data = request.form.get('box_data')
            class_name = request.form.get('class_name')
            return jsonify(reviewer.add_annotation(folder_name, image_name, box_data, class_name))
        elif action == 'update_annotation_box':
            image_name = request.form.get('image_name')
            annotation_id = request.form.get('annotation_id')
            box_data = request.form.get('box_data')
            return jsonify(reviewer.update_annotation_box(folder_name, image_name, annotation_id, box_data))
        elif action == 'delete_frame':
            image_name = request.form.get('image_name')
            return jsonify(reviewer.delete_frame(folder_name, image_name))
            
    # GET request - show form
    folders = [f for f in os.listdir('annotated-images') if os.path.isdir(os.path.join('annotated-images', f))]
    return render_template('review.html', folders=folders)

@app.route('/get_image/<path:image_path>')
def get_image(image_path):
    return send_file(image_path)

if __name__ == '__main__':
    app.run(debug=True)
