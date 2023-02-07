import tensorflow as tf
import matplotlib
import cv2
import imghdr
import os

# avoid oom errors by setting gpu memory consumption growth
# grab all the gpus available in the machine
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
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


