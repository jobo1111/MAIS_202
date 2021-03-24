import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime
import sklearn.metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import torch
import torchvision
import torch.nn as nn
warnings.filterwarnings("ignore")


# create paths to csv and mp3 files
PATH = "."
TRAIN = PATH + "/train_audio"
test = TRAIN + "/aldfly/XC2628.mp3"

# list of bird species to learn
# birds selected from folders with 100 mp3 files
birds = ['aldfly', 'amecro', 'amegfi', 'amepip', 'amered',
         'amerob', 'annhum', 'astfly', 'balori', 'banswa',
         'barswa', 'bewwre', 'bkbwar', 'bkcchi', 'bkhgro',
         'bkpwar', 'blkpho', 'blujay', 'bnhcow', 'boboli']
         

# open csv file
df = pd.read_csv((PATH +'/train.csv'), usecols=['ebird_code','filename', 'duration'])

# add only listed bird species to dataframe
data_seq = []
cur_bird = 'aldfly'
count = 1

for row in df.itertuples():
  
  
  if row[1] in birds:
    """
    if row[2]<100 and row[2]>10:
      if cur_bird != row[1]:
        print(count)
        cur_bird = row[1]
        count = 1
      else:
        count += 1
        """
    data_seq.append([row[1],row[3],row[2]])

# turn the data into a dataframe
data = pd.DataFrame(data_seq, columns=['ebird_code','filename', 'duration'])


# open an mp3 file and create an np array holding the data
def create_img(code, filename):
  n_fft = 2048
  hop_length=512
  
  y, sr = librosa.load(TRAIN+"/"+code+"/"+filename)
  ft = np.abs(librosa.stft(y, n_fft=n_fft,  hop_length=hop_length))
  fi = librosa.decompose.nn_filter(ft)
  return (fi, sr, n_fft, hop_length, code) # returns an np array



# open all images
def create_imgs(data):
  imgs = []
  count = 0
  spec = 'aldfly'
  for bird in data.itertuples():
    try:
      new = create_img(str(bird[1]),str(bird[2]))
    except:
      print(bird[1], bird[2])
      continue
    
    print(new[0].shape)
    np.save(str(bird[2]), new)
    
  return imgs

# sets the size of each np array to 1025 x new_size
# and then flattens the array
def fix_shape(data):
  new_size = 2000
  
  for bird in data.itertuples():
    try:
      fpath = str(bird[2]) + ".npy"
      img = np.load(fpath, allow_pickle=True)[0]
      img = Image.fromarray(img)
      img = img.resize((200,100))
      newImg = np.asarray(img).flatten()    
      np.save(str(bird[2] + "_flat"), newImg)
    except:
      print(bird[1], bird[2])
  

# this creates a training set from the first 70 mp3 files from each bird
# and the remaining files are used for testing
def create_train(data):
  num = 70
  count = 0
  cur_bird = "aldfly"
  x_train = []
  y_train = []
  x_test = []
  y_test = []
  for bird in data.itertuples():
    try:
      fpath = str(bird[2]) + "_flat" + ".npy"
      img = np.load(fpath, allow_pickle=True)
      if img.shape[0] != 20000:
        continue
      
      if count >= num:
        if bird[1] != cur_bird:
          count = 0
          cur_bird = bird[1]
          x_train.append(img)
          y_train.append(bird[1])
        else:
          x_test.append(img)
          y_test.append(bird[1])
          
      else:
        count += 1
        x_train.append(img)
        y_train.append(bird[1])
    except:
      print(bird[1], bird[2])

  
  return (x_train, y_train, x_test, y_test)

# uncomment this when you want to convert all the mp3 files to npy files
"""
# create the images
#create_imgs(data)

# flatten the images
#fix_shape(data)

test_set = create_train(data)
np.save("x_train", test_set[0])
np.save("y_train", test_set[1])
np.save("x_test", test_set[2])
np.save("y_test", test_set[3])
"""

# uncomment this when you want to train and test the data
"""
x_train = np.load("x_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
x_test = np.load("x_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

pca = KernelPCA(n_components=150)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# choose one of the two following classifiers to train

# svm classifier
clf = svm.SVC(kernel="linear")

# knn classifier
#clf = KNeighborsClassifier(n_neighbors=5)


clf.fit(x_train, y_train)

y_train_pred = []
for vector in x_train:
  y_train_pred.append(clf.predict([vector]))
print(sklearn.metrics.accuracy_score(y_train,y_train_pred))

y_test_pred = []
for vector in x_test:
  y_test_pred.append(clf.predict([vector]))
print(sklearn.metrics.accuracy_score(y_test,y_test_pred))

"""

"""
# this is for showing a sample spectrogram
img = create_img('aldfly', 'XC317903.mp3')
librosa.display.specshow(img[0], sr=img[1], hop_length=img[3], x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB');
plt.show()
"""




