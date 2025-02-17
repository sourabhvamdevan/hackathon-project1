# Medical Image Analyzer

A web-based application for analyzing medical images using deep learning. The application classifies medical images into Normal or Abnormal categories using a trained TensorFlow model.

## Features

- 🖼️ Upload medical images through drag & drop or file selection
- 🤖 Real-time image classification using deep learning
- 📊 View prediction history
- 🌗 Dark/Light theme support
- 💾 Persistent storage of prediction history
- 📱 Responsive design for all devices

## Technology Stack

- **Backend**
  - Python 3.x
  - Flask (Web framework)
  - TensorFlow (Deep Learning)
  - Pillow (Image processing)

- **Frontend**
  - HTML5
  - TailwindCSS (Styling)
  - JavaScript (ES6+)
  - Fetch API (AJAX requests)

## Project Structure

```
Medical-Image-Analyzer/
├── app.py                     # Flask application
├── medical_image_model.h5     # TensorFlow model
├── image_predictions.json     # Prediction history
├── static/                    # Static files
├── templates/                 # HTML templates
│   └── index.html            # Main application page
├── uploads/                   # Uploaded images
└── dataset/                   # Training dataset
    ├── class1/               # Normal images
    └── class2/               # Abnormal images
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sourabhvamdevan/medical-image-analyzer.git
cd medical-image-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install flask tensorflow pillow
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload a medical image using the drag & drop interface or file selector

4. View the classification result and prediction history

## API Endpoints

### `GET /`
- Returns the main application page
- Response: HTML

### `POST /upload`
- Uploads and analyzes a medical image
- Request: Form data with 'file' field
- Response: JSON
```json
{
    "prediction": "Normal" | "Abnormal"
}
```

### `GET /predictions`
- Retrieves prediction history
- Response: JSON array of predictions
```json
[
    {
        "filename": "image.jpg",
        "filepath": "uploads/image.jpg",
        "prediction": "Normal",
        "timestamp": "2025-02-17 21:25:15"
    }
]
```

## Model Information

The application uses a pre-trained TensorFlow model (`medical_image_model.h5`) for binary classification of medical images. The model accepts images of size 224x224 pixels and outputs a probability score, where:
- Score > 0.5: Normal
- Score ≤ 0.5: Abnormal

## Development

- The application uses Flask's development server with debug mode enabled
- Image predictions are stored in `image_predictions.json`
- Uploaded images are saved in the `uploads/` directory
- The frontend uses TailwindCSS for styling with dark mode support

## Notes

- Maximum file upload size: 10MB
- Supported image formats: PNG, JPG, GIF
- The application maintains a history of the 5 most recent predictions
