import keras
#import tensorflow
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input,Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np
from CustomCNN import TrainModel
from ResNet import residual_network
from keras.layers import *


#out1 = model1(commonInput)    
#out2 = model2(commonInput)  
#out3 = model3(commonInput)    
#
#mergedOut = Add()([out1,out2,out3])
## output layer
#mergedOut = Dense(10, activation='softmax')(mergedOut)

#from tensorflow.keras.models import Model

#commonInput = Input(input_shape)
#newModel = Model(commonInput, mergedOut)



#Check for GPU usage
from tensorflow.keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

img_rows, img_cols = 32, 32
img_channels       = 3
num_classes = 10
stack_n= 5
#Load dataset
def load_data():
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #z-score
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    

    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)
    return x_train,x_test,y_train,y_test


x_train,x_test,y_train,y_test = load_data()
optimizer= keras.optimizers.Adam(lr=0.001)
customModel=TrainModel(x_train)
customModel.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=None)
customModel.fit(x=x_train, y=y_train, batch_size=64, epochs=10, verbose=1)

img_input = Input(shape=(img_rows,img_cols,img_channels))
output    = residual_network(img_input,num_classes,stack_n)
