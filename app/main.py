import os
import warnings
import numpy as np
from PIL import Image

from flask import Flask, render_template, request
from keras.models import load_model


warnings.filterwarnings("ignore")

"""
Constants variables to use in this work.
"""
app = Flask(__name__)

model = load_model('final_model.h5', compile=False)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index_view():
    """
    Route to display the template where user uploads the image
    """
    return render_template('index.html')

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    """
    Function to restrict file names required for prediction
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(filename):
    """
    Function to process user input image for prediction.
    
    NB: In the same way the model was pre-processed for training, then, it should be the same
    way model will be pre-processed for prediction
    """
    img = Image.open(filename).resize((32, 32))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 
    return x

@app.route('/predict', methods=['POST'])
def predict():
    
    """
    Function to predict the class image belongs. The class value should be in (class_names) list
    """
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(os.getcwd(), 'static', 'images', filename)
            file.save(file_path)
            print(f"File saved at: {file_path}")

            if os.path.exists(file_path):
                img = read_image(file_path)
                class_prediction = model.predict(img)
                class_index = np.argmax(class_prediction)
                class_name = class_names[class_index]

                return render_template('predict.html', class_name=class_name, prob=class_prediction[0][class_index], user_image=file_path, filename=filename)
            else:
                return "File not found at the specified path"
        else:
            return "Unable to read the file. Please check the file extension"

if __name__ == '__main__':
    
    """
    Start the Flask application, running on port 5000
    """
    app.run(debug=True, use_reloader=False, port=5000)