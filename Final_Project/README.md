# Bird Call Classifier

Final project for McGill AI Society Intro to ML Bootcamp (Winter 2021).

Training data retrieved from [Kaggle](https://www.kaggle.com/c/birdsong-recognition/data).

## Project Description

This Bird Call Classifier project is a web app that classifies bird calls from an 
uploaded mp3 file. I built this model using Sci-kit Learn and built the web app using
Flask. I utilized Librosa's library to convert each mp3 file into a spectrogram then
passed them to Sci-kit's PCA and SVM methods to classify the bird calls from the data 
retrieved from Kaggle. 

## Running the app
To run the app, install all the packages in requirements.txt. Then change to the main directory ​
and run the following code
```bash
python ./model/model.py
python app.py
```


Then open a browser and go to [http://localhost:5000](http://localhost:5000)

## Repository organization
```
├── README.md
├── app.py                      # Main code to run the flask app
├── bird_predictior.py          # Contains the predictor class 
├── model
│   ├── model.py                # Contains the code for the model
│   ├── x.npy                   # Contains the preprocessed images
│   ├── y.npy                   # Contains the corresponding labels
│   └── results
│       └── clf.pkl             # Contains the pretrained classifier
├── requirements.txt            # File containing packages needed to run the code
├── static
│   ├── css
│   │   └── main.css            # Style sheet to make the front-end prettier
│   └── js
│       └── main.js             # Javascript file handling front-end actions
└── templates
    └── index.html              # HTML file that Flask renders
```
