import os
from flask import Flask, request, render_template

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = "/home/chengwei/Work/pytorch-flask-deploy/static"

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files.get('image')
        if image_file is None or image_file.filename == "":
            return render_template('index.html', prediction='Not allowed')
        elif not allowed_file(image_file.filename):
            return render_template('index.html', prediction='Not allowed')
        if image_file:
            # image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            # image_file.save(image_location)
            img_bytes = image_file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            return render_template('index.html', prediction=prediction.item())
    return render_template('index.html', prediction=0)
# def predict():
    # if request.method == 'POST':
    #     file = request.files.get('file')
    #     if file is None or file.filename == "":
    #         return jsonify({'error': 'no file'})
    #     if not allowed_file(file.filename):
    #         return jsonify({'error': 'format not supported'})

    #     try:
    #         img_bytes = file.read()
    #         tensor = transform_image(img_bytes)
    #         prediction = get_prediction(tensor)
    #         data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
    #         return jsonify(data)
    #     except:
    #         return jsonify({'error': 'error during prediction'})

if __name__ == "__main__":
    app.run(port=5000, debug=True)