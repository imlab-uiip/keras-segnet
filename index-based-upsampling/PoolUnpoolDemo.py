import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

from custom_layers import MemoMaxPooling2D, MemoUpSampling2D
from custom_layers import MemoMaxPooling3D, MemoUpSampling3D

from keras import models
from keras import backend as K

import numpy as np

shape = (1, 8, 9, 17, 33)
#shape = (1, 2, 9, 17)

autoencoder = models.Sequential()

PoolingLayer = MemoMaxPooling2D if len(shape) == 4 else MemoMaxPooling3D
UpSamplingLayer = MemoUpSampling2D if len(shape) == 4 else MemoUpSampling3D

pool_1 = PoolingLayer(input_shape=shape[1:])
pool_2 = PoolingLayer()

autoencoder.add(pool_1)
autoencoder.add(pool_2)

autoencoder.add(UpSamplingLayer(pool_2))
autoencoder.add(UpSamplingLayer(pool_1))

autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=['accuracy'])

X = np.random.randint(0, 100, shape)

print X
test_output = K.function([autoencoder.layers[0].input],
                         [autoencoder.layers[3].output])

out = test_output([X])[0]
print out[0]
