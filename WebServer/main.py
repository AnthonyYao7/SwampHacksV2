import os
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import sys
from PIL import Image
import Model.inference as inference

def predict(filename):
    return inference.inference_on_image(filename)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Uploads')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, template_folder='templates', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print('hi')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            flash('file {} saved'.format(file.filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            nutrients = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('nutritionFacts.html', calories=round(nutrients['Calories']), tFat=round(nutrients['Total Fat']), sFat=round(nutrients['Saturated Fat']), chol=round(nutrients['Cholesterol']), carb=round(nutrients['Carbohydrates']), fiber=round(nutrients['Fiber']), protein=round(nutrients['Protein']))  # put in the new page to redirect to in here
    return render_template('food.html')


if __name__ == '__main__':
    app.secret_key = 'the random string'
    app.run(debug=True)