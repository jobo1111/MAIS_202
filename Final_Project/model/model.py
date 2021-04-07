import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn
import sklearn.metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import joblib

# librosa creates a warning for each mp3 file that it processes
# this silences them all
warnings.filterwarnings("ignore")


# create paths to csv and mp3 files
PATH = "D:\Mais_Deliverable"
TRAIN = PATH + "/train_audio"

# list of bird species to learn
# birds selected from folders with 100 mp3 files
birds = ['aldfly', 'amecro', 'amegfi', 'amepip', 'amered',
         'amerob', 'annhum', 'astfly', 'balori', 'banswa',
         'barswa', 'bewwre', 'bkbwar', 'bkcchi', 'bkhgro',
         'bkpwar', 'blkpho', 'blujay', 'bnhcow', 'boboli']
         

# open csv file
df = pd.read_csv((PATH +'/train.csv'), usecols=['ebird_code','filename', 'duration'])

# create a list of tuples containing the filename, the bird code and the duration of the bird call
data_seq = []
for row in df.itertuples():
  if row[1] in birds:
    data_seq.append([row[1],row[3],row[2]])

# turn the list into a dataframe
data = pd.DataFrame(data_seq, columns=['ebird_code','filename', 'duration'])


# open an mp3 file and create an np array holding the spectrogram
# as well as other information needed to graph the spectrogram
def create_img(code, filename):
  n_fft = 2048
  hop_length=512
  
  y, sr = librosa.load(TRAIN+"/"+code+"/"+filename)
  ft = np.abs(librosa.stft(y, n_fft=n_fft,  hop_length=hop_length))
  fi = librosa.decompose.nn_filter(ft)
  return (fi, sr, n_fft, hop_length, code) # returns an np array



# save each image as a .npy file
def create_imgs(data):
  for bird in data.itertuples():
    try:
      new = create_img(str(bird[1]),str(bird[2]))
    except:
      print(bird[1], bird[2])
      continue
    np.save(str(bird[2]), new)
    

# resizes the image then flattens it into a 1D array, 
# then saves the array again
def fix_shape(data):
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

# Saves the x array and y_label array to train and test the model 
def cre_train(data):
  x = []
  y = []
  for bird in data.itertuples():
    try:
      fpath = str(bird[2]) + "_flat" + ".npy"
      img = np.load(fpath, allow_pickle=True)
      if img.shape[0] != 20000:
        continue
      x.append(img)
      y.append(bird[1])
    except:
      print(bird[1], bird[2])
  
  np.save("x", x)
  np.save("y", y)




# train the data on the dataset
def train():
  x = np.load(PATH + "/x.npy", allow_pickle=True)
  y = np.load(PATH + "/y.npy", allow_pickle=True)

  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=0)

  pca = KernelPCA(n_components=150)
  pca.fit(x_train)

  joblib_pca = "./backend/model/results/pca.pkl"
  joblib.dump(pca, joblib_pca)

  x_train = pca.transform(x_train)
  x_test = pca.transform(x_test)

  # choose one of the two following classifiers to train
  # svm classifier
  clf = svm.SVC(kernel="linear")

  # knn classifier
  #clf = KNeighborsClassifier(n_neighbors=5)

  clf.fit(x_train, y_train)

  # Save to file in the current working directory
  joblib_file = "./backend/model/results/clf.pkl"
  joblib.dump(clf, joblib_file)

  # check training accuracy
  y_train_pred = clf.predict(x_train)
  print("Train Accuracy:", sklearn.metrics.accuracy_score(y_train,y_train_pred))
  print()

  # check test accuracy
  y_test_pred = clf.predict(x_test)
  print("Test Accuracy:", sklearn.metrics.accuracy_score(y_test,y_test_pred))


  np.set_printoptions(precision=2)
  # Plot non-normalized confusion matrix
  titles_options = [("Confusion matrix, without normalization", None)]
  for title, normalize in titles_options:
      disp = sklearn.metrics.plot_confusion_matrix(clf, x_test, y_test,
                                  display_labels=birds,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
      disp.ax_.set_title(title)
  plt.show()


def show_spec():
  # this is for showing a sample spectrogram
  img = create_img('aldfly', 'XC317903.mp3')
  librosa.display.specshow(img[0], sr=img[1], hop_length=img[3], x_axis='time', y_axis='log')
  plt.colorbar(format='%+2.0f dB')
  plt.show()

if __name__ == "__main__":
  # create the images
  #create_imgs(data)
  # flatten the images
  #fix_shape(data)
  
  cre_train(data)
  train()








