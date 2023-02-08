import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib
import cv2
import imghdr
import os

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


