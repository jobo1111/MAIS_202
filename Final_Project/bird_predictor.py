import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn
from PIL import Image
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import joblib
import os
warnings.filterwarnings("ignore")

bird_dic = {
    'aldfly':'Alder Flycatcher', 'amecro':'American Crow', 'amegfi':'American Goldfinch', 'amepip':'Buff-bellied Pipit', 
    'amered':'American Redstart', 'amerob':'American Robin', 'annhum':"Anna's Hummingbird", 'astfly':'Ash-throated Flycatcher',
    'balori':'Baltimore Oriole', 'banswa':'Sand Martin','barswa':'Barn Swallow', 'bewwre':"Bewick's Wren", 
    'bkbwar':'Blackburnian Warbler', 'bkcchi':'Black-capped Chickadee', 'bkhgro':'Black-headed Grosbeak','bkpwar':'Blackpoll Warbler', 
    'blkpho':'Black Phoebe', 'blujay':'Blue Jay', 'bnhcow':'Brown-headed Cowbird', 'boboli':'Bobolink'
}

def create_img(filename):
  n_fft = 2048
  hop_length=512
  
  y, sr = librosa.load(filename)
  ft = np.abs(librosa.stft(y, n_fft=n_fft,  hop_length=hop_length))
  fi = librosa.decompose.nn_filter(ft)
  return (fi, sr, n_fft, hop_length) # returns an np array

def fix_shape(image):
    img = Image.fromarray(image)
    img = img.resize((200,100))
    newImg = np.asarray(img).flatten()    
    return newImg
    

class Bird_predictor:
    def __init__(self):
        self.model = joblib.load("./backend/model/results/clf.pkl")
        self.pca = joblib.load("./backend/model/results/pca.pkl")

    def predict(self, filename):
        """
        This method reads the file uploaded from the Flask application POST request,
        and performs a prediction using the MNIST model.
        """

        
        image = create_img(filename)[0]
        image = fix_shape(image)
        image = self.pca.transform([image])
        model_output = self.model.predict(image)
        return bird_dic[model_output[0]]