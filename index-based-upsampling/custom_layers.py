from mpool import m_maxpool2d, m_maxpool3d
from keras.engine.topology import Layer as Layer_inh
from keras import backend as K
from theano import tensor as T
from keras.layers.pooling import _Pooling2D, _Pooling3D


class MemoMaxPooling2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering=K.image_dim_ordering(), **kwargs):
        super(MemoMaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output, ind = m_maxpool2d(inputs, pool_size, strides,
                                  border_mode, dim_ordering)
        self.ind = ind
        return output


class MemoUpSampling2D(Layer_inh):
    def __init__(self, pooling_layer, size=(2, 2), shape=None, dim_ordering=K.image_dim_ordering(), **kwargs):
        self.size = tuple(size)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        # self.input_spec = [InputSpec(ndim=4)]
        self.ind = pooling_layer.ind
        assert shape is None or len(shape) == 2, 'shape should be 2D vector'
        self.shape = shape or pooling_layer.ind.shape[-2:]
        super(MemoUpSampling2D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            self.shape = self.shape or self.size * input_shape[:-2]
            return (input_shape[0],
                    input_shape[1],
                    self.shape[0],
                    self.shape[1])
        elif self.dim_ordering == 'tf':
            self.shape = self.shape or self.size * input_shape[1:-1]
            return (input_shape[0],
                    self.shape[0],
                    self.shape[1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # TODO: implement for dim_ordering='tf'
        img = K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
        padded = T.zeros((img.shape[0], img.shape[1], self.shape[0], self.shape[1]))
        padded = T.set_subtensor(padded[:, :, :img.shape[2], :img.shape[3]], img)
        return T.switch(self.ind, padded, T.zeros_like(padded))

    def get_config(self):
        config = {'size': self.size}
        base_config = super(MemoUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#####################################################################################################

class MemoMaxPooling3D(_Pooling3D):
    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode='valid',
                 dim_ordering=K.image_dim_ordering(), **kwargs):
        super(MemoMaxPooling3D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output, ind = m_maxpool3d(inputs, pool_size, strides, border_mode, dim_ordering)
        self.ind = ind
        return output


class MemoUpSampling3D(Layer_inh):
    def __init__(self, pooling_layer, size=(2, 2, 2), shape=None, dim_ordering=K.image_dim_ordering(), **kwargs):
        self.size = tuple(size)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        # self.input_spec = [InputSpec(ndim=5)]
        self.ind = pooling_layer.ind
        assert shape is None or len(shape) == 3, 'shape should be 3D vector'
        self.shape = shape or pooling_layer.ind.shape[-3:]
        super(MemoUpSampling3D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    self.shape[0],
                    self.shape[1],
                    self.shape[2])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    self.shape[0],
                    self.shape[1],
                    self.shape[2],
                    input_shape[4])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # TODO: implement for dim_ordering='tf'
        img = K.resize_volumes(x, self.size[0], self.size[1], self.size[2], self.dim_ordering)
        padded = T.zeros((img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.shape[2]))
        padded = T.set_subtensor(padded[:, :, :img.shape[2], :img.shape[3], :img.shape[4]], img)
        return T.switch(self.ind, padded, T.zeros_like(padded))

    def get_config(self):
        config = {'size': self.size}
        base_config = super(MemoUpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
