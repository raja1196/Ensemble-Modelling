import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np
#from main import load_data

baseMapNum = 32
weight_decay = 1e-4
num_classes=10
#x_train,x_test,y_train,y_test=load_data()

def TrainModel(x_train):
    CustomModel = Sequential()
    CustomModel.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    CustomModel.add(Activation('relu'))
    CustomModel.add(BatchNormalization())
    CustomModel.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    CustomModel.add(Activation('relu'))
    CustomModel.add(BatchNormalization())
    CustomModel.add(MaxPooling2D(pool_size=(2,2)))
    CustomModel.add(Dropout(0.2))
    
    CustomModel.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    CustomModel.add(Activation('relu'))
    CustomModel.add(BatchNormalization())
    CustomModel.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    CustomModel.add(Activation('relu'))
    CustomModel.add(BatchNormalization())
    CustomModel.add(MaxPooling2D(pool_size=(2,2)))
    CustomModel.add(Dropout(0.3))
    
    CustomModel.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    CustomModel.add(Activation('relu'))
    CustomModel.add(BatchNormalization())
    CustomModel.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    CustomModel.add(Activation('relu'))
    CustomModel.add(BatchNormalization())
    CustomModel.add(MaxPooling2D(pool_size=(2,2)))
    CustomModel.add(Dropout(0.4))
    
    CustomModel.add(Flatten())
    CustomModel.add(Dense(num_classes, activation='softmax'))
    
    CustomModel.summary()
    return(CustomModel)
