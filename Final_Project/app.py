from flask import Flask, render_template, request
import os

# Import your model
from bird_predictor import Bird_predictor

app = Flask(__name__)
UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Instantiate your models
bird = Bird_predictor()


# Base endpoint to perform prediction.
@app.route('/', methods=['POST'])
def make_prediction():
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    prediction = bird.predict(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return render_template('index.html', prediction=prediction, generated_text=None, tab_to_show='bird')

    


@app.route('/', methods=['GET'])
def load():
    return render_template('index.html', prediction=None, generated_text=None, tab_to_show='bird')


@app.route('/predict/image', methods=['POST'])
def make_image_prediction():
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    prediction = bird.predict(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    print(prediction)
    return str(prediction)




if __name__ == '__main__':
    app.run(debug=True)