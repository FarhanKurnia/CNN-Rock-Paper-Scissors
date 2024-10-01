from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from tensorflow import keras
# from keras.preprocessing import image
from keras_preprocessing import image
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)

# Load the .pkl model
with open('rps_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set upload folder
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded image
            img = image.load_img(filepath, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Predict the class using the .pkl model
            classes = model.predict(images)
            result = ''
            if classes[0][0] == 1:
                result = 'paper'
            elif classes[0][1] == 1:
                result = 'rock'
            else:
                result = 'scissors'

            # Render the result on the web page
            return render_template('predict.html', result=result, filepath=filepath, filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='images/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
