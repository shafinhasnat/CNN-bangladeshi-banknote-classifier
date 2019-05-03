import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2


notes=['1000 front (shaheed minar)',
       '1000 back','2 front (bongobondhu)','2 back (shaheed minar)',
       '500 front (bongobondhu)','500 back','10 front (bongobondhu)',
       '10 back','20 front (bongobondhu)','20 back',
       '100 front','100 back','5 front','5 back',
       '50 front','50 back']
model=tf.keras.models.load_model('banknote_model.h5')
img=cv2.imread('D:/BanknoteClasiifier/trainingFiles/50b/50b_7.jpg',0)
cv2.imshow('img',img)
img=cv2.resize(img,(100,100))
##cv2.imshow('img',img)
img=img.reshape(1,100,100,1)
img=img.astype('float32')
img=img/255.0
print(img.dtype)
pred=model.predict(img)
print('argmax',np.argmax(pred[0]),'\n',pred[0][np.argmax(pred[0])],'\n',notes[np.argmax(pred[0])])
cv2.waitKey(0)

###1000f is 0
###1000b is 1
###2f is 2
###2b is 3
