import csv
import cv2
import numpy as np

lines = []
with open('/Users/altonyeung/Google Drive/Udacity/Git/self-driving-car/CarND-Behavioral-Cloning-P3/simdata/rev1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines: #loop through each line in the csv
    source_path = line[0] #get file name of center image
    filename = source_path.split('\\')[-1] #extract only the file name from file path
    current_path = '/Users/altonyeung/Google Drive/Udacity/Git/self-driving-car/CarND-Behavioral-Cloning-P3/simdata/rev1/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3]) # get steering angle 
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model_lambda.h5')

