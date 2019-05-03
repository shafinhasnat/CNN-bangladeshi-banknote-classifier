import glob
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

num=5
img_rows=100
img_cols=100
batch_size=32
nb_classes=16
#############
nb_epoch=5
nb_filters=32
nb_pool=2
nb_conv=3


def importImg():
    oneKF=[]
    oneKB=[]
    twoF=[]
    twoB=[]
    fiveHF=[]
    fiveHB=[]
    tenF=[]
    tenB=[]
    twentyF=[]
    twentyB=[]
    oneHF=[]
    oneHB=[]
    fiveF=[]
    fiveB=[]
    fiftyF=[]
    fiftyB=[]
#####################
    oneKF_path='D:/BanknoteClasiifier/trainingFiles/1000f\*.jpg'
    oneKB_path='D:/BanknoteClasiifier/trainingFiles/1000b\*.jpg'
    twoF_path='D:/BanknoteClasiifier/trainingFiles/2f\*.jpg'
    twoB_path='D:/BanknoteClasiifier/trainingFiles/2b\*.jpg'
    fiveHF_path='D:/BanknoteClasiifier/trainingFiles/500f\*.jpg'
    fiveHB_path='D:/BanknoteClasiifier/trainingFiles/500b\*.jpg'
    tenF_path='D:/BanknoteClasiifier/trainingFiles/10f\*.jpg'
    tenB_path='D:/BanknoteClasiifier/trainingFiles/10b\*.jpg'
    twentyF_path='D:/BanknoteClasiifier/trainingFiles/20f\*.jpg'
    twentyB_path='D:/BanknoteClasiifier/trainingFiles/20b\*.jpg'
    oneHF_path='D:/BanknoteClasiifier/trainingFiles/100f\*.jpg'
    oneHB_path='D:/BanknoteClasiifier/trainingFiles/100b\*.jpg'
    fiveF_path='D:/BanknoteClasiifier/trainingFiles/5f\*.jpg'
    fiveB_path='D:/BanknoteClasiifier/trainingFiles/5b\*.jpg'
    fiftyF_path='D:/BanknoteClasiifier/trainingFiles/50f\*.jpg'
    fiftyB_path='D:/BanknoteClasiifier/trainingFiles/50b\*.jpg'
#####################
    print('importing image intialized...')
    for file in glob.glob(oneKF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         oneKF.append(im)
    oneKF=np.array(oneKF)

    for file in glob.glob(oneKB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         oneKB.append(im)
    oneKB=np.array(oneKB)

    for file in glob.glob(twoF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         twoF.append(im)
    twoF=np.array(twoF)
    
    for file in glob.glob(twoB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         twoB.append(im)
    twoB=np.array(twoB)
    
    for file in glob.glob(fiveHF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         fiveHF.append(im)
    fiveHF=np.array(fiveHF)
    
    for file in glob.glob(fiveHB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         fiveHB.append(im)
    fiveHB=np.array(fiveHB)
    
    for file in glob.glob(tenF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         tenF.append(im)
    tenF=np.array(tenF)
    
    for file in glob.glob(tenB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         tenB.append(im)
    tenB=np.array(tenB)
    
    for file in glob.glob(twentyF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         twentyF.append(im)
    twentyF=np.array(twentyF)
    
    for file in glob.glob(twentyB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         twentyB.append(im)
    twentyB=np.array(twentyB)
    
    for file in glob.glob(oneHF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         oneHF.append(im)
    oneHF=np.array(oneHF)
    
    for file in glob.glob(oneHB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         oneHB.append(im)
    oneHB=np.array(oneHB)
    
    for file in glob.glob(fiveF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         fiveF.append(im)
    fiveF=np.array(fiveF)
    
    for file in glob.glob(fiveB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         fiveB.append(im)
    fiveB=np.array(fiveB)
    
    for file in glob.glob(fiftyF_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         fiftyF.append(im)
    fiftyF=np.array(fiftyF)
    
    for file in glob.glob(fiftyB_path):
         im=cv2.imread(file,0)
         im=cv2.resize(im,(img_rows,img_cols))
         im=np.array((im))
         fiftyB.append(im)
    fiftyB=np.array(fiftyB)
########################
    trainingImg=np.concatenate((oneKF,oneKB,twoF,twoB,fiveHF,fiveHB,tenF,tenB,twentyF,twentyB,oneHF,oneHB,fiveF,fiveB,fiftyF,fiftyB))
#######################
    
    
    print('importing done!')
    return oneKF,oneKB,twoF,twoB,fiveHF,fiveHB,tenF,tenB,twentyF,twentyB,oneHF,oneHB,fiveF,fiveB,fiftyF,fiftyB,trainingImg
#########################



def label():
    print('Labelling intialized...')
    oneKF_label=np.repeat(0,len(oneKF))
    oneKB_label=np.repeat(1,len(oneKB))
    twoF_label=np.repeat(2,len(twoF))
    twoB_label=np.repeat(3,len(twoB))
    fiveHF_label=np.repeat(4,len(fiveHF))
    fiveHB_label=np.repeat(5,len(fiveHB))
    tenF_label=np.repeat(6,len(tenF))
    tenB_label=np.repeat(7,len(tenB))
    twentyF_label=np.repeat(8,len(twentyF))
    twentyB_label=np.repeat(9,len(twentyB))
    oneHF_label=np.repeat(10,len(oneHF))
    oneHB_label=np.repeat(11,len(oneHB))
    fiveF_label=np.repeat(12,len(fiveF))
    fiveB_label=np.repeat(13,len(fiveB))
    fiftyF_label=np.repeat(14,len(fiftyF))
    fiftyB_label=np.repeat(15,len(fiftyB))
###############
    trainingLabel=np.concatenate((oneKF_label,oneKB_label,twoF_label,twoB_label,fiveHF_label,
                                  fiveHB_label,tenF_label,tenB_label,twentyF_label,twentyB_label,
                                  oneHF_label,oneHB_label,fiveF_label,fiveB_label,fiftyF_label,fiftyB_label))
###############
    print('Labelling done!')
    return trainingLabel

def trainingSet():
    trainingImg_set,trainingLabel_set=shuffle(trainingImg,trainingLabel,random_state=2)
    x_train,x_test,y_train,y_test=train_test_split(trainingImg_set,trainingLabel_set,test_size=.2,random_state=4)
    plt.imshow(x_test[num],cmap=plt.cm.binary)
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train=x_train/255.0
    x_test=x_test/255.0
    y_train=np_utils.to_categorical(y_train,nb_classes)
    return x_train,x_test,y_train,y_test

def training():
    print('training initialized...')
    model=Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid'))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=nb_epoch)
    print('training complete!!!')
    model.save('banknote_model.h5')
    print('model saved!!!')

    pred=model.predict(x_test)
    print('argmax',np.argmax(pred[num]),'\n',pred[num])
    pass


if __name__=='__main__':
    oneKF,oneKB,twoF,twoB,fiveHF,fiveHB,tenF,tenB,twentyF,twentyB,oneHF,oneHB,fiveF,fiveB,fiftyF,fiftyB,trainingImg=importImg()
################
    trainingLabel=label()
    x_train,x_test,y_train,y_test=trainingSet()
    print('\noneKF',len(oneKF),'\noneKB',len(oneKB),'\ntwoF',len(twoF),'\ntwoB',len(twoB),
          '\nfiveHF',len(fiveHF),'\nfiveHB',len(fiveHB),'\ntenF',len(tenF),'\ntenB',len(tenB),
          '\ntwentyF',len(twentyF),'\ntwentyB',len(twentyB),
          '\noneHF',len(oneHF),'\noneHB',len(oneHB),'\ntenF',len(fiveF),'\ntenB',len(fiveB),
          '\nfiftyF',len(fiftyF),'\nfiftyB',len(fiftyB),
          '\ntrainingImg',len(trainingImg),
          '\nx_train',len(x_train),'\nx_test',len(x_test),'\ny_train',len(y_train),'\ny_test',len(y_test))
################
    
    training()
