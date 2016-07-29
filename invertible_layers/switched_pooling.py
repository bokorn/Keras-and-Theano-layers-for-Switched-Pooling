# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:04:56 2016

@author: bokorn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:58:14 2016

@author: bokorn
"""

from keras import backend as K
from keras.layers import convolutional as convolutional
from keras.layers.core import Lambda
import switched_pooling_backend as K_sp

def max_only_lambda(index_type='flattened', dim_ordering='th'):
    if index_type == 'flattened' or index_type == 'space_filling':
        if dim_ordering == 'th':            
            def max_only(X):
                split_idx = K.shape(X)[1]//2
                return X[:,:split_idx,:,:]
            def max_only_output_shape(X):
                return(X[0], X[1]//2, X[2], X[3])
        elif dim_ordering == 'tf':
            def max_only(X):
                split_idx = K.shape(X)[3]//2
                return X[:,:,:,:split_idx]
            def max_only_output_shape(X):
                return(X[0], X[1], X[2], X[3]//2)
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    elif index_type == 'array':
        if dim_ordering == 'th':            
            def max_only(X):
                split_idx = K.shape(X)[1]//3
                return X[:,:split_idx,:,:]
            def max_only_output_shape(X):
                return(X[0], X[1]//3, X[2], X[3])
        elif dim_ordering == 'tf':
            def max_only(X):
                split_idx = K.shape(X)[3]//3
                return X[:,:,:,:split_idx]
            def max_only_output_shape(X):
                return(X[0], X[1], X[2], X[3]//3)
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        raise Exception('Invalid index_type: ' + index_type)
    
    return Lambda(max_only, output_shape=max_only_output_shape)

def switch_only_lambda(index_type='flattened', dim_ordering='th'):
    if index_type == 'flattened' or index_type == 'space_filling':
        if dim_ordering == 'th':            
            def switch_only(X):
                split_idx = K.shape(X)[1]//2
                return X[:,split_idx:,:,:]
            def switch_only_output_shape(X):
                return(X[0], X[1]//2, X[2], X[3])
        elif dim_ordering == 'tf':
            def switch_only(X):
                split_idx = K.shape(X)[3]//2
                return X[:,:,:,split_idx:]
            def switch_only_output_shape(X):   
                return(X[0], X[1], X[2], X[3]//2)
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    elif index_type == 'array':
        if dim_ordering == 'th':            
            def switch_only(X):
                split_idx = K.shape(X)[1]//3
                return X[:,split_idx:,:,:]
            def switch_only_output_shape(X):
                return(X[0], (X[1]//3)*2, X[2], X[3])
        elif dim_ordering == 'tf':
            def switch_only(X):
                split_idx = K.shape(X)[3]//3
                return X[:,:,:,split_idx:]
            def switch_only_output_shape(X):
                return(X[0], X[1], X[2], (X[3]//3)*2)
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        raise Exception('Invalid index_type: ' + index_type)
    
    return Lambda(switch_only, output_shape=switch_only_output_shape)

    
class MaxPoolSwitch2D(convolutional._Pooling2D):
    '''Max pooling operation for spatial data.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    @property
    def output_shape(self):

        output_shape = list(super(MaxPoolSwitch2D, self).output_shape)
        
        if self.index_type == 'flattened' or self.index_type == 'space_filling':
            index_size = 2
        elif self.index_type == 'array':
            index_size = 3
        else:
            raise Exception('Invalid index_type: ' + self.index_type)


        if self.dim_ordering == 'th':
            output_shape[1] *= index_size
        elif self.dim_ordering == 'tf':
            output_shape[3] *= index_size
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        
        return tuple(output_shape)
            
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', index_type='flattened', index_scope='local', **kwargs):
        super(MaxPoolSwitch2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)
        self.index_type = index_type
        self.index_scope = index_scope

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K_sp.maxpoolswitch2d(inputs, pool_size, strides,
                          border_mode, dim_ordering)
        return output

    def get_config(self):
        config = {'index_type': self.index_type,
                  'index_scope': self.index_scope}
        base_config = super(MaxPoolSwitch2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UnPoolSwitch2D(convolutional.Layer):
    '''Unpools data using switchs

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, pool_size=(2, 2), strides=None, dim_ordering='th', 
                 index_type='flattened', index_scope='local',
                 original_input_shape=None, **kwargs):
        super(UnPoolSwitch2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.index_type = index_type
        self.index_scope = index_scope
        self.original_input_shape = original_input_shape
        
    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.original_input_shape is None:

            if self.dim_ordering == 'th':
                return (input_shape[0],
                        input_shape[1],
                        self.strides[0] * (input_shape[2] - 1) + self.pool_size[0],
                        self.strides[1] * (input_shape[3] - 1) + self.pool_size[1])
            elif self.dim_ordering == 'tf':
                return (input_shape[0],
                        self.strides[0] * (input_shape[1] - 1) + self.pool_size[0],
                        self.strides[1] * (input_shape[2] - 1) + self.pool_size[1],
                        input_shape[3])
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        else:
            return self.original_input_shape


    def get_output(self, train=False):
        X = self.get_input(train)
        output = K_sp.unpoolswitch2d(X, pool_size=self.pool_size, 
                                     strides=self.strides,
                                     dim_ordering=self.dim_ordering,
                                     index_type=self.index_type,
                                     index_scope=self.index_scope,
                                     original_input_shape=self.original_input_shape)
        return output
        
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_size': self.pool_size,
                  'strides': self.strides,
                  'index_type': self.index_type,
                  'index_scope': self.index_scope,
                  'original_input_shape': self.original_input_shape}
        base_config = super(UnPoolSwitch2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
