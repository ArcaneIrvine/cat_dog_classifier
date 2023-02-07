import tensorflow as tf
import os

# avoid oom errors by setting gpu memory consumption growth
# grab all the gpus available in the machine
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# for every gpu set memory growth (making tensorflow to keep the memory only to what it needs)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
