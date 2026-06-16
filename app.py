from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import time
import shutil

# Import from core
from core.image_processing import process_pipeline
from core.model import load_model, predict_tumor

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML model at startup
clf_model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_uploads():
    """Removes old files in the uploads folder to keep it clean"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/', methods=['GET'])
def index():
    # If model failed to load, pass a warning to the template
    model_status = "Loaded" if clf_model is not None else "Not Found (Run train_model.py first)"
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        # Clean up previous uploads to prevent clutter
        cleanup_uploads()
        
        # Save original file securely
        filename = secure_filename(file.filename)
        # Use timestamp to ensure uniqueness and order
        unique_prefix = str(int(time.time()))
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_prefix}_input.jpg")
        file.save(save_path)
        
        # 1. Run Image Processing Pipeline
        try:
            pipeline_results = process_pipeline(save_path, app.config['UPLOAD_FOLDER'], unique_prefix)
        except Exception as e:
            flash(f"Error processing image: {str(e)}")
            return redirect(url_for('index'))
            
        # 2. Make Prediction using ML Model
        features = pipeline_results['features']
        prediction_result, confidence = predict_tumor(features, clf_model)
        
        # Add prediction to results
        pipeline_results['prediction'] = prediction_result
        pipeline_results['confidence'] = confidence
        
        # Convert absolute paths to relative paths for HTML rendering
        # They are currently something like static/uploads/1234_1_orig.jpg
        # Flask wants just the relative path from the static folder if we use url_for,
        # but since we serve them directly, just having 'static/uploads/...' is fine if we use them directly in src="/static/uploads/..."
        # But to be safe and clean, we'll strip the leading parts if needed, or just pass them as is.
        # process_pipeline returned paths like 'static/uploads/...' or 'static\uploads\...'
        # Ensure forward slashes for web
        for key, val in pipeline_results.items():
            if isinstance(val, str) and '\\' in val:
                 pipeline_results[key] = val.replace('\\', '/')
        
        return render_template('index.html', results=pipeline_results, model_status="Loaded" if clf_model else "Not Found")
        
    else:
        flash('Allowed image types are png, jpg, jpeg')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
