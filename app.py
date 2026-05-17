import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

MODEL_PATH = 'brain_tumor_model.keras' 
CONFIDENCE_THRESHOLD = 0.50 

print("Loading Keras model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

if not os.path.exists('static'):
    os.makedirs('static')

def is_valid_mri(img_path):
   
    image_array = cv2.imread(img_path)
    
    if image_array is None:
        return False
        
 
    b, g, r = cv2.split(image_array)

    r, g, b = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))
    

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    dark_pixels_ratio = np.sum(gray < 20) / (gray.shape[0] * gray.shape[1])
    

    if color_diff > 15 or dark_pixels_ratio < 0.10:
        return False
        
    return True

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    confidence_value = None
    result_type = None  
    img_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)

        
            if not is_valid_mri(img_path):
                result_type = "error"
                prediction_text = " Error — This does not appear to be a valid brain MRI scan. Please upload a correct medical image."
                confidence_value = None
            else:
                
                processed_img = prepare_image(img_path)
                result = model.predict(processed_img)
                raw_score = float(result[0][0])

             
                if raw_score > 0.5:
                    confidence = raw_score
                    predicted_label = "tumor"
                else:
                    confidence = 1.0 - raw_score
                    predicted_label = "healthy"

                confidence_pct = round(confidence * 100, 2)

                
                if predicted_label == "tumor":
                    result_type = "tumor"
                    prediction_text = " Positive: Tumor Detected"
                    confidence_value = confidence_pct
                else:
                    result_type = "healthy"
                    prediction_text = " Negative: No Tumor Detected"
                    confidence_value = confidence_pct

    return render_template(
        'index.html',
        prediction=prediction_text,
        confidence=confidence_value,
        result_type=result_type,
        img_path=img_path
    )

if __name__ == '__main__':
    app.run(debug=True)
