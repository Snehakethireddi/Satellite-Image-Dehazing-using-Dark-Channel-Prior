from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils import process_dehazing
from PIL import Image
import io
import base64
import numpy as np
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            original_img, dehazed_img = process_dehazing(filepath)

            # Convert images to base64 for displaying in HTML
            buffered = io.BytesIO()
            dehazed_img.save(buffered, format="JPEG")
            dehazed_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Get original image for display
            original_img_pil = Image.fromarray((original_img * 255).astype(np.uint8))
            buffered_original = io.BytesIO()
            original_img_pil.save(buffered_original, format="JPEG")
            original_img_str = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

            return render_template('index.html',
                                   original_img=original_img_str,
                                   dehazed_img=dehazed_img_str)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)