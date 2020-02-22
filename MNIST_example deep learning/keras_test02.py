# 1. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

#model = load_model('/home/bme/keras/keras_test/my_model_mnist.h5')
#model = load_model(r'd:\keras_test\my_model_mnist.h5')
#model = load_model('d:\\temp\\keras_test\\my_model_mnist.h5')
#model = load_model('d:/temp/keras_test/my_model_mnist.h5')
model = load_model(r'd:\model.hdf5')

img = cv2.imread(r'd:\number_2.jpg',0);
img2 = cv2.resize(img,(28,28))
#img = img2.reshape(1,1,28,28);
img = img2.reshape(1,28,28,1);
img = img.astype('float32');
img = 1.0 - img/255.0
b = img[0,:,:]
img3 = np.asarray(img);



img = cv2.imread(r'd:\number_5.jpg',0);
img2 = cv2.resize(img,(28,28))
#img = img2.reshape(1,1,28,28);
img = img2.reshape(1,28,28,1);
img = img.astype('float32');
img = 1.0 - img/255.0
b = img[0,:,:]
img3 = np.append(img3,img, axis=0);

img = cv2.imread(r'd:\number_9.jpg',0);
img2 = cv2.resize(img,(28,28))
#img = img2.reshape(1,1,28,28);
img = img2.reshape(1,28,28,1);
img = img.astype('float32');
img = 1.0 - img/255.0
b = img[0,:,:]
img3 = np.append(img3,img, axis=0);

print('Ready to classify')
out = model.predict(img3)
print(out)

plt.imshow(img[0,0,:,:])
#model.predict(img)


