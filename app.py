from flask import Flask, request, jsonify, render_template
import os
from PIL import Image
import tensorflow as tf
import json
from datetime import datetime

app = Flask(__name__)


DATA_FILE = 'image_predictions.json'


def init_json_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)


model = tf.keras.models.load_model('medical_image_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

   
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    
    img = Image.open(filepath).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    predicted_class = "Normal" if prediction[0][0] > 0.5 else "Abnormal"

    try:
      
        init_json_file()
        with open(DATA_FILE, 'r') as f:
            predictions = json.load(f)
        
       
        prediction_data = {
            'filename': file.filename,
            'filepath': filepath,
            'prediction': predicted_class,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        predictions.append(prediction_data)
        
       
        with open(DATA_FILE, 'w') as f:
            json.dump(predictions, f, indent=4)
            
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        
        return jsonify({'prediction': predicted_class})

    return jsonify({'prediction': predicted_class})

@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        init_json_file()
        with open(DATA_FILE, 'r') as f:
            predictions = json.load(f)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_json_file()  
    app.run(debug=True)
