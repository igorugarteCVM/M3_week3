from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import optimizers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.misc import imresize
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import os
from sklearn.decomposition import PCA
from tqdm import tqdm

#user defined variables
f = open("env.txt", "r")
ENV = f.read().split('"')[1]

IMG_SIZE    = 32
BATCH_SIZE  = 16
if ENV == "local":
    DATASET_DIR = '../../../week2/MIT_split'
    MODEL_FNAME = './models/my_first_mlp.h5'

else:
    DATASET_DIR = '/home/mcv/datasets/MIT_split'
    MODEL_FNAME = './models/my_first_mlp.h5'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
else:
    print('The dataset directory exists!')

print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='Reshape'))
model.add(Dense(units=4096, activation='relu',name='first'))
model.add(Dense(units=2048, activation='relu',name='second'))
model.add(Dense(units=1024, activation='relu',name='third'))
model.add(Dense(units=512, activation='relu',name='fourth'))
model.add(Dense(units=8, activation='softmax'))


optimizer_name = 'Adagrad'
optimizer = optimizers.Adagrad(lr=1e-3)

model.compile(loss='categorical_crossentropy',
optimizer=optimizer,
metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

# Train or load the weights of the model
if not os.path.exists(MODEL_FNAME):
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
    DATASET_DIR+'/train',  # this is the target directory
    target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
    batch_size=BATCH_SIZE,
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
    class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
    DATASET_DIR+'/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
    class_mode='categorical')

    history = model.fit_generator(
    train_generator,
    steps_per_epoch=1881 // BATCH_SIZE,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=807 // BATCH_SIZE)

    print('Done!\n')
    print('Saving the model into '+MODEL_FNAME+' \n')
    model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
    print('Done!\n')

else:
    print("The model already exists. Loading the weights...")
    model.load_weights(MODEL_FNAME)

print('Reading filenames')
train_images_filenames = pickle.load(open(DATASET_DIR + '/train_images_filenames.dat', 'rb'))
test_images_filenames = pickle.load(open(DATASET_DIR + '/test_images_filenames.dat', 'rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('train_labels.dat','rb'))
test_labels = pickle.load(open('test_labels.dat','rb'))

# Dictionaty of features with the corresponding output layer as key
train_features = np.zeros((len(train_images_filenames),512),dtype=np.float32)
test_features = np.zeros((len(test_images_filenames),512),dtype=np.float32)

# Train features
print('Obtaining the train features...')
for idx in tqdm(range(len(train_images_filenames))):
    img = np.asarray(Image.open('../../../week2/' + train_images_filenames[idx][2:]))
    img = np.expand_dims(imresize(img, (IMG_SIZE, IMG_SIZE, 3)), axis=0)

    model_layer = Model(inputs=model.input, outputs=model.get_layer('fourth').output)
    train_features[idx,:] = model_layer.predict(img)
    

print('Obtaining the test features...')
for idx in tqdm(range(len(test_images_filenames))):
    img = np.asarray(Image.open('../../../week2/' + test_images_filenames[idx][2:]))
    img = np.expand_dims(imresize(img, (IMG_SIZE, IMG_SIZE, 3)), axis=0)

    model_layer = Model(inputs=model.input, outputs=model.get_layer('fourth').output)
    test_features[idx,:] = model_layer.predict(img)
    

param_grid = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10]},
              {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [1, 0.1, 0.01], 'C': [0.1, 1, 10]}  
             ]

#PCA

n_components = 2**np.arange(5,10)
print(n_components)

n_components_X_test = []  
for i in tqdm(range(len(n_components))):
    pca = PCA(n_components = n_components[i])
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)
    
    print(test_pca.shape)
    print(train_pca.shape)


    # SVM at first layer
    print('SVM at FOURTH layer:')
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid.fit(train_pca, train_labels)
    print("BEST PARAMS", grid.best_params_, grid.best_estimator_)
    grid_predictions = grid.predict(test_pca)
    print("classification_report\n", classification_report(test_labels, grid_predictions))
