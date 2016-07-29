# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:11:39 2016

@author: bokorn
"""
from theano_switched_pooling import max_pool_switch_2d, unpool_switch_2d

def maxpoolswitch2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='th', index_type='flattened', index_scope='local'):
    if border_mode == 'same':
        # TODO: add implementation for border_mode="same"
        raise Exception('border_mode="same" not supported with Theano.')
    elif border_mode == 'valid':
        ignore_border = True
        padding = (0, 0)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))

    pool_out = max_pool_switch_2d(x, ds=pool_size, st=strides,
                                      ignore_border=ignore_border,
                                      padding=padding,
                                      index_type=index_type, 
                                      index_scope=index_scope)

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out
    
def unpoolswitch2d(x, pool_size, strides=(1, 1),
           dim_ordering='th', index_type='flattened', index_scope='local', original_input_shape=None):
    
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))
        if original_input_shape is not None:
            original_input_shape =  (original_input_shape[0], 
                                     original_input_shape[3], 
                                     original_input_shape[1], 
                                     original_input_shape[2])
            
    pool_out = unpool_switch_2d(x, ds=pool_size, st=strides,
                                      index_type=index_type, 
                                      index_scope=index_scope,
                                      original_input_shape=original_input_shape)

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out