import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot as plt

# load some image
img = cv2.imread('test/cattest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
resized_img = tf.image.resize(img, (256, 256))
plt.imshow(resized_img.numpy().astype(int))
plt.show()

# load the model and print result
model = load_model(os.path.join('models', 'cat_dog_model.h5'))
result = model.predict(np.expand_dims(resized_img/255, 0))
print(result)
if result > 0.5:
    print(f'Predicted class is Dog')
else:
    print(f'Predicted class is Cat')
