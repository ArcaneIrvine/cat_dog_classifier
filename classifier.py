import tensorflow as tf
import numpy as np
import cv2
import imghdr
import os

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy

# avoid oom errors by setting gpu memory consumption growth
# grab all the gpus available in the machine
gpus = tf.config.experimental.list_physical_devices('GPU')
# for every gpu set memory growth (making tensorflow to keep the memory only to what it needs)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# remove faulty images
data_dir = 'data'
image_extensions = ['jpg', 'jpeg', 'png', 'bmp']

# loop through all our data
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            # load image
            img = cv2.imread(image_path)
            # check if the extension fits out list of allowed extensions
            tip = imghdr.what(image_path)
            # if not remove that image
            if tip not in image_extensions:
                print('image not in extensions list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('issue with image {}'.format(image_path))

# load data
# build a dataset on the fly using tf.keras which will automatically create a dataset from our images
data = tf.keras.utils.image_dataset_from_directory('data')

"""
# convert it to a numpy iterator (allows us to access our data pipeline)
data_iterator = data.as_numpy_iterator()
# get a batch from the iterator (allows us to access our data pipeline itself)
batch = data_iterator.next()

# visualize a batch
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
"""

# pre-process data
# scale data using map function (x is our image and y is the key)
data = data.map(lambda x, y: (x/255, y))
# convert it to a numpy iterator (allows us to access our data pipeline)
scaled_iterator = data.as_numpy_iterator()
# get a batch from the iterator (allows us to access our data pipeline itself)
batch = scaled_iterator.next()

# split data
train_size = int(len(data)*.7)
validation_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

# use take and skip methods (how many batches we want to allocate for each split we declared)
train = data.take(train_size)
validation = data.skip(train_size).take(validation_size)
test_size = data.skip(train_size+validation_size).take(test_size)

# Deep Model
# declare our model using the Sequential api
model = Sequential()
# add layers to the model

"""
add a convolution with 16 filters 3x3 in size and a stride of 1. Relu activation
that will convert any negative values to 0 and anything positive will remain unchanged.
Then also define what the input shape looks like so 256x256 pixels wired by 3 channels deep 
(basically scans over an image and tries to condense or extract the relevant information 
inside of that image to make an output classification)
"""
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
# apply a Max Pooling layer which is going to take the max value after the relu activation and return that value
model.add(MaxPooling2D())

# similarly add a convolution with 32 filters this time and relu activation
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

# similarly add a convolution with 16 filters again this time and relu activation
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

# flatten the data down
model.add(Flatten())

# add fully connected Dense layers with 256 neurons and relu activation
model.add(Dense(256, activation='relu'))
# add a final Dense layer with 1 neuron to get a single output that is going to represent 0 or 1 with a sigmoid activation (which will match our classes)
model.add(Dense(1, activation='sigmoid'))

# compile our model with adam optimizer and define what our losses are and an accuracy metric
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# model.summary()

# Train
logdir = 'logs'
# used for logging out the model training as its training, so we can come back and see how it performed at a particular time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# fit the model
hist = model.fit(train, epochs=20, validation_data=validation, callbacks=[tensorboard_callback])

# Plot our model performance using matplotlib
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# Plot our model accuracy using matplotlib
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# Evaluate performance
# define some instances
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# loop through each batch in our test data and make a prediction
for batch in test_size.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
