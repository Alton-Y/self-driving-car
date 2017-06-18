import csv
import os

simdataList = os.listdir('simdata')

samples = []
for i in range(len(simdataList)):
    csvPath = 'simdata/'+simdataList[i]+'/driving_log.csv'
    with open(csvPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
# samples holds every line scanned from every csv


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.22
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                rndSeed = np.random.randint(3) # random select between 0,1,2
                # rndSeed = 0
                rndSeed2 = np.random.randint(2) # random flip or not
                folderName = batch_sample[rndSeed].split('/')[-3]
                fileName = batch_sample[rndSeed].split('/')[-1]
                name = 'simdata/'+folderName+'/IMG/'+fileName
                image = cv2.imread(name)
                angle = float(batch_sample[3])

                if rndSeed == 1:# left image
                    angle += correction
                elif rndSeed == 2: # right image
                    angle -= correction

                if rndSeed2 == 1: # flip the image
                    image = cv2.flip(image,1)
                    angle = angle*-1.0

                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from IPython.display import Image, display, SVG
from keras.utils.visualize_util import model_to_dot
# Save the model as png file
from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)

# model.load_weights('model_weights.h5')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)

model.save_weights('model_weights.h5')
model.save('model.h5')