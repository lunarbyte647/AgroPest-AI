from flask import Flask, render_template, request
from predict import predict_crop
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():

    # Create uploads folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Get uploaded image
    image = request.files['image']

    if image.filename == '':
        return "No image selected!"

    # Save image
    image_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        image.filename
    )

    image.save(image_path)

    
    result = predict_crop(image_path)
    

    return render_template(
        'result.html',
        image=image.filename,
        result=result
    )


if __name__ == '__main__':
    app.run(debug=True)