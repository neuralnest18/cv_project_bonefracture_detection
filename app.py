from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Provide the correct path to your model file
model_path = 'C:\\Users\\HP\\cv_lab_project\\resnet50_bone_fracture.keras'
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_fracture(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Rescale image

    prediction = model.predict(img)
    print(f"Raw prediction: {prediction}")  # Debug print to see raw prediction values
    return 'Fracture' if prediction[0][0] >= 0.5 else 'No Fracture'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-page')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded image
        result = predict_fracture(file_path)
        
        return jsonify({'filename': filename, 'result': result})
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on port 5001 to avoid conflict
