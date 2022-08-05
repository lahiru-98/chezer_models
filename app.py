from flask import Flask , request
from tensorflow import keras
import joblib
import numpy as np
from PIL import Image , ImageOps


app = Flask(__name__)

arousal_model = keras.models.load_model('models_ar-270')
ar_scaler = joblib.load('arl-scaler.sav')


@app.route('/')
def index():
    return 'Your App is Working'


@app.route('/pred', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['file']
        #img = cv2.imread(uploaded_file)
        image_ready = np.array(
            Image.open(image).convert("L").resize((48, 48)) # image resizing
            )
        image_ready = image_ready/255 # normalize the image in 0 to 1 range
        img_array = np.expand_dims(np.expand_dims(image_ready, -1),0)
        ar_result = arousal_model.predict(img_array)
        arousal = ar_scaler.inverse_transform(ar_result)[0][0]
        return "Prediction - "+str(arousal)
    else:
        return "SOMETHING WENT WRONG"

if __name__ == "__main__":
    app.run()
