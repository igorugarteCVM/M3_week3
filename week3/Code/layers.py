import os
import getpass

from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.misc import imresize
from tqdm import tqdm
from keras import optimizers

f = open("env.txt", "r")
ENV = f.read().split('"')[1]

# user defined variables
# hola
IMG_SIZE = 32
BATCH_SIZE = 16
if ENV == "local":
    DATASET_DIR = '../../../week2/MIT_split'
    MODEL_FNAME = './models/my_first_mlp.h5'

else:
    DATASET_DIR = '/home/mcv/datasets/MIT_split'
    MODEL_FNAME = './models/my_first_mlp.h5'



if not os.path.exists(DATASET_DIR):
    print(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
    quit()

print('Building MLP model...\n')

train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR + '/train',  # this is the target directory
    target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
    batch_size=BATCH_SIZE,
    classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    DATASET_DIR + '/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')

# Build the Multi Layer Perceptron model
acc = []
val_acc = []
loss = []
val_loss = []

init_unit = 4096


optimizers_names = ['Adagrad']
optimizer = optimizers.Adagrad(lr=1e-3)
""" optimizers_ = [optimizers.Adagrad(lr=1e-3),optimizers.RMSprop(lr=1e-4),optimizers.SGD(lr=1e-2),optimizers.Adam(lr=1e-3),optimizers.Adadelta(lr=1e-2),
               optimizers.Adamax(lr=1e-3),optimizers.Nadam(lr=1e-4)] """

model = Sequential()
   
model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='first'))
model.add(Dense(units=4096, activation='relu',name='second'))
model.add(Dense(units=2048, activation='relu',name='third'))
model.add(Dense(units=1024, activation='relu',name='fourth'))
model.add(Dense(units=512, activation='relu',name='fifth'))
model.add(Dense(units=8, activation='softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

print(model.summary())

print('Done!\n')

if os.path.exists(MODEL_FNAME):
    print('WARNING: model file ' + MODEL_FNAME + ' exists and will be overwritten!\n')

print('Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling

history = model.fit_generator(
    train_generator,
    steps_per_epoch=1881 // BATCH_SIZE,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=807 // BATCH_SIZE)

print('Done!\n')
print('Saving the model into ' + MODEL_FNAME + ' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
print('Done!\n')

# summarize history for accuracy
acc.append(history.history['acc'])
val_acc.append(history.history['val_acc'])

loss.append(history.history['loss'])s
val_loss.append(history.history['val_loss'])

np.save('history_' + optimizers_names[i] + '.npy', history.history)
    
""" optimizers_names = ['Adagrad','RMSprop','SGD','Adam','Adadelta','Adamax','Nadam']

acc = []
val_acc = []
loss = []
val_loss = []

for optimizer in optimizers_names:
    history = np.load('history_' + optimizer + '.npy')
    acc.append(history.item().get('acc'))
    val_acc.append(history.item().get('val_acc'))
    loss.append(history.item().get('loss'))
    val_loss.append(history.item().get('val_loss'))
    

plt.title('Best accuracies per optimizer')
plt.ylabel('Accuracy')
plt.xlabel('Optimizer')

max_acc = np.max(np.array(acc)[:,-5:],axis=1)
max_val_acc = np.max(np.array(val_acc)[:,-5:],axis=1)
X_axis = np.arange(len(optimizers_names))


plt.bar(X_axis - 0.2,max_acc,0.4, label='Train')
plt.bar(X_axis + 0.2,max_val_acc,0.4, label='Test')

for i in range(len(max_acc)):
    plt.text(X_axis[i] - 0.4, max_acc[i], str(np.round(max_acc[i],3)))
    plt.text(X_axis[i], max_val_acc[i], str(np.round(max_val_acc[i],3)))

  
  
plt.xticks(X_axis, optimizers_names)
plt.legend(loc='lower right')
plt.savefig('accuracy_per_optimizer.jpg')
plt.close() """
 
""" plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
legend = [] 
for i in range(len(acc)):
    print(i)
    plt.plot(acc[i])
    legend.append(optimizers_names[i])
    
plt.legend(legend, loc='upper left')
plt.savefig('accuracy_train.jpg')
plt.close()

plt.title('model accuracy validation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
legend = [] 
for i in range(len(val_acc)):
    plt.plot(val_acc[i])
    legend.append(optimizers_names[i])
    
plt.legend(legend, loc='upper left')
plt.savefig('accuracy_test.jpg')
plt.close()


plt.title('model losss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
legend = [] 
for i in range(len(loss)):
    plt.plot(loss[i])
    legend.append(optimizers_names[i])
    
plt.legend(legend, loc='upper left')
plt.savefig('loss_train.jpg')
plt.close()

plt.title('model loss validation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
legend = [] 
for i in range(len(val_loss)):
    plt.plot(val_loss[i])
    legend.append(optimizers_names[i])
    
plt.legend(legend, loc='upper left')
plt.savefig('loss_test.jpg')
plt.close() """

