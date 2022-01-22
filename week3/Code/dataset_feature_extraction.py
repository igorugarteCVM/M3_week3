import numpy as np
from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from scipy.misc import imresize
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tqdm import tqdm

IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = './MIT_split'
MODEL_FNAME = './work/my_first_mlp.h5'

model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
model.add(Dense(units=2048, activation='relu',name='second'))
#model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=8, activation='softmax'))
model.load_weights(MODEL_FNAME)
model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)


train_images_filenames = pickle.load(open('train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat','rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames  = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('train_labels.dat','rb'))
test_labels = pickle.load(open('test_labels.dat','rb'))

print(str(len(train_images_filenames)) + ' training images')
print(str(len(test_images_filenames)) + ' test images')


#get the features from images
train_descriptors = []          # All the descriptors stacked together
train_label_per_descriptor = [] # Labels of the stacked descriptors (1 label per descriptor)

for filename,labels in tqdm(zip(train_images_filenames,train_labels)):
    ima=cv2.imread(filename)
    img = np.asarray(Image.open(filename))
    x = np.expand_dims(imresize(img, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
    features = model_layer.predict(x/255.0)
    train_descriptors.append(features[0])
    train_label_per_descriptor.append(labels)



test_descriptors = []          # All the descriptors stacked together
test_label_per_descriptor = [] # Labels of the stacked descriptors (1 label per descriptor)

for filename,labels in tqdm(zip(test_images_filenames,test_labels)):
    ima=cv2.imread(filename)
    img = np.asarray(Image.open(filename))
    x = np.expand_dims(imresize(img, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
    features = model_layer.predict(x/255.0)
    test_descriptors.append(features[0])
    test_label_per_descriptor.append(labels)


""" 
k = 128 # Number of visual words

# -- MLP K-MEANS --
MLP_codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
MLP_codebook.fit(features) """

scaler = StandardScaler()
scaler.fit(train_descriptors) 
train_descriptors = scaler.transform(train_descriptors)
scaler.fit(test_descriptors)
test_descriptors = scaler.transform(test_descriptors)

param_grid = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10]},
              {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]}]

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

grid.fit(train_descriptors, train_labels)

print("BEST PARAMS", grid.best_params_, grid.best_estimator_)

scaler.fit(test_descriptors)
test_descriptors = scaler.transform(test_descriptors)

grid_predictions = grid.predict(test_descriptors)

print("classification_report", classification_report(test_labels, grid_predictions))

