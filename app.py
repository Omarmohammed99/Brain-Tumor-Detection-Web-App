import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)


MODEL_PATH = 'brain_tumor_model.h5'


print("Loading Keras model...")
model = load_model(MODEL_PATH)
print("Model loaded!")
if not os.path.exists('static'):
    os.makedirs('static')

def prepare_image(img_path):
    
    img = image.load_img(img_path, target_size=(128, 128))
    
 
    img_array = image.img_to_array(img)
    
 
    img_array = np.expand_dims(img_array, axis=0)
    
  
    img_array = img_array / 255.0
    
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    
    if request.method == 'POST':
       
        file = request.files['file']
        if file:
           
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
            
            
            processed_img = prepare_image(img_path)
            result = model.predict(processed_img)
            
           
            if result[0][0] > 0.5:
                confidence = round(result[0][0] * 100, 2)
                prediction = f"⚠️ Positive: Tumor Detected ({confidence}%)"
            else:
                confidence = round((1 - result[0][0]) * 100, 2)
                prediction = f" Negative: No Tumor Detected ({confidence}%)"

    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    
    app.run(debug=True)