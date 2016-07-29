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

import numpy as np
from six.moves import xrange
import six.moves.builtins as builtins

import theano
from theano import tensor as T
from theano import gof, Op, Variable, Apply
from theano.tensor.signal.pool import MaxPoolGrad
import warnings

def max_pool_switch_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
            index_type='flattened', index_scope='local'):

    if input.ndim < 2:
        raise NotImplementedError('max_pool_switched_2d requires a dimension >= 2')
    if ignore_border is None:
        ignore_border = False
    if input.ndim == 4:
        op = MaxPoolSwitch(ds, ignore_border, st=st, padding=padding,
                  index_type=index_type, index_scope=index_scope)
        output = op(input)
        return output

    # extract image dimensions
    img_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = T.cast(T.join(0, batch_size,
                                        T.as_tensor([1]),
                                        img_shape), 'int64')
    input_4D = T.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of images
    op = MaxPoolSwitch(ds, ignore_border, st=st, padding=padding,
              index_type=index_type, index_scope=index_scope)
    output = op(input_4D)

    # restore to original shape
    outshp = T.join(0, input.shape[:-2], output.shape[-2:])
    return T.reshape(output, outshp, ndim=input.ndim)

def unpool_switch_2d(input, ds, st=None,
            index_type='flattened', index_scope='local',
            original_input_shape=None):

    if input.ndim < 3:
        raise NotImplementedError('unpool_switched_2d requires a dimension >= 3')
    if input.ndim == 4:
        op = UnpoolSwitch(ds, st=st,
                  index_type=index_type, index_scope=index_scope,
                  original_input_shape=original_input_shape)
        output = op(input)
        return output

    # extract image dimensions
    img_shape = input.shape[-3:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input.shape[:-3])
    batch_size = T.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = T.cast(T.join(0, batch_size,
                                        img_shape), 'int64')
    input_4D = T.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of images
    op = UnpoolSwitch(ds, st=st,
              index_type=index_type, index_scope=index_scope,
              original_input_shape=original_input_shape)
    output = op(input_4D)

    # restore to original shape
    outshp = T.join(0, input.shape[:-2], output.shape[-2:])
    return T.reshape(output, outshp, ndim=input.ndim)

class MaxPoolSwitch(Op):

    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'index_type', 'index_scope')

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0), index_type='flattened'):
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if len(imgshape) < 4:
            input_channels = 1
        else:
            input_channels = imgshape[-3]

        if st is None:
            st = ds
        input_rows, input_cols = imgshape[-2:]
        input_rows += padding[0] * 2
        input_cols += padding[1] * 2

        if ignore_border:
            if ds[0] == st[0]:
                output_rows = input_rows // st[0]
            else:
                out_r = (input_rows - ds[0]) // st[0] + 1
                if isinstance(input_rows, theano.Variable):
                    output_rows = T.maximum(out_r, 0)
                else:
                    output_rows = np.maximum(out_r, 0)

            if ds[1] == st[1]:
                output_cols = input_cols // st[1]
            else:
                out_c = (input_cols - ds[1]) // st[1] + 1
                if isinstance(input_cols, theano.Variable):
                    output_cols = T.maximum(out_c, 0)
                else:
                    output_cols = np.maximum(out_c, 0)
        else:
            if isinstance(input_rows, theano.Variable):
                output_rows = T.switch(T.ge(st[0], ds[0]),
                                   (input_rows - 1) // st[0] + 1,
                                   T.maximum(0, (input_rows - 1 - ds[0]) //
                                                  st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                output_rows = (input_rows - 1) // st[0] + 1
            else:
                output_rows = max(0, (input_rows - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(input_cols, theano.Variable):
                output_cols = T.switch(T.ge(st[1], ds[1]),
                                   (input_cols - 1) // st[1] + 1,
                                   T.maximum(0, (input_cols - 1 - ds[1]) //
                                                  st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                output_cols = (input_cols - 1) // st[1] + 1
            else:
                output_cols = max(0, (input_cols - 1 - ds[1]) // st[1] + 1) + 1


        if index_type == 'flattened' or index_type == 'space_filling':
            index_size = 2
        elif index_type == 'array':
            index_size = 3
        else:
            raise Exception('Invalid index_type: ' + index_type)

        output_channels = input_channels * index_size
           
        rval = list(imgshape[:-3]) + [output_channels, output_rows, output_cols]
        return rval

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0),
                 index_type='flattened', index_scope='local'):
        self.ds = tuple(ds)
        if not all([isinstance(d, int) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        if(not isinstance(st, (tuple, list))):
            raise TypeError(
                "Pool stride parameters must be list or tuple."
                " Got %s" % type(st))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')
        if index_type not in ['flattened', 'space_filling', 'array']:
            raise ValueError(
                "MaxPoolSwitch index_type parameter only support 'flattened',"
                " 'space_filling' and 'array'. Got %s" % index_type)
        self.index_type = index_type
        if index_scope not in ['local', 'global']:
            raise ValueError(
                "MaxPoolSwitch index_scope parameter only support 'local'"
                " and 'global'. Got %s" % index_scope)
        self.index_scope = index_scope
                
        if self.index_type == 'flattened' or self.index_type == 'space_filling':
            self.index_size = 2
        elif self.index_type == 'array':
            self.index_size = 3
        
        self.index_type_int = ['flattened', 'space_filling', 'array'].index(self.index_type)

        if self.index_type == 'space_filling':
            raise ValueError(
                "MaxPoolSwitch index_type space_filling not implemented")

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError("MaxPoolSwitch Requires 4 Dimensions, " +  str(x.type.ndim) + " Given")
        x = T.as_tensor_variable(x)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = T.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding, self.index_type)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        # number of pooling output rows
        output_row = zz.shape[-2]
        # number of pooling output cols
        output_col = zz.shape[-1]
        row_downsample, col_downsample = self.ds
        row_stride, col_stride = self.st
        pad_height = self.padding[0]
        pad_width = self.padding[1]
        input_row = x.shape[-2] + 2 * pad_height
        input_col = x.shape[-1] + 2 * pad_width
        input_channels = x.shape[-3]
        
        is_global_scope = self.index_scope == 'global'
        
        if is_global_scope:
            scoped_size = (input_row, input_col)
        else:
            scoped_size = self.ds

        # pad the image
        if self.padding != (0, 0):
            y = np.zeros(
                (x.shape[0], x.shape[1], input_row, input_col),
                dtype=x.dtype)
            y[:, :, pad_height:(input_row - pad_height), pad_width:(input_col - pad_width)] = x
        else:
            y = x

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(output_row):
                    row_start = r * row_stride
                    row_end = builtins.min(row_start + row_downsample, input_row)

                    for c in xrange(output_col):
                        col_start = c * col_stride
                        col_end = builtins.min(col_start + col_downsample, input_col)

                        zz[n, k, r, c] = np.max(x[n, k, row_start:row_end, col_start:col_end])
                        max_idx = np.argmax(x[n, k, row_start:row_end, col_start:col_end])

                        local_size = (row_end-row_start, col_end-col_start)
                        max_row, max_col = np.unravel_index(max_idx, local_size)
                        
                        if is_global_scope:
                            max_row += row_start
                            max_col += col_start
                            
                        if self.index_type_int == 0:
                            max_idx = np.ravel_multi_index([max_row, max_col], scoped_size)
                            zz[n, k+input_channels, r, c] = max_idx
                        elif self.index_type_int == 2:
                            zz[n, k+input_channels, r, c] = max_row
                            zz[n, k+2*input_channels, r, c] = max_col
                            
    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(in_shapes[0], self.ds,
                             self.ignore_border, self.st, self.padding, self.index_type)
        return [shp]
    
    def grad(self, inputs, grads):
        x, = inputs
        gz, = grads
        maxswitchout = self(x)
        
        return [MaxPoolGrad(self.ds,
                            ignore_border=self.ignore_border,
                            st=self.st, padding=self.padding)(x, maxswitchout, gz)]

    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        row_downsample, col_downsample = self.ds
        row_stride, col_stride = self.st
        pad_height, pad_width = self.padding        
        index_type_int = self.index_type_int
        index_size = self.index_size
        is_global_scope = int(self.index_scope == 'global')
        
        ccode='''
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int output_row, output_col, output_channels; // shape of the output
        int input_row, input_col, input_channels; // shape of the padded_input
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        
        input_channels = PyArray_DIMS(%(x)s)[1];

        input_row = PyArray_DIMS(%(x)s)[2];
        input_col = PyArray_DIMS(%(x)s)[3];
        input_row += %(pad_height)s * 2;
        input_col += %(pad_width)s * 2;
        
        output_channels = input_channels * %(index_size)s;

        if (%(pad_height)s != 0 && %(pad_width)s != 0 && !%(ignore_border)s)
        {
          PyErr_SetString(PyExc_ValueError,
            "padding must be (0,0) when ignore border is False");
          %(fail)s;
        }
        if (%(ignore_border)s)
        {
            // '/' in C is different from '/' in python
            if (input_row - %(row_downsample)s < 0)
            {
              output_row = 0;
            }
            else
            {
              output_row = (input_row - %(row_downsample)s) / %(row_stride)s + 1;
            }
            if (input_col - %(col_downsample)s < 0)
            {
              output_col = 0;
            }
            else
            {
              output_col = (input_col - %(col_downsample)s) / %(col_stride)s + 1;
            }
        }
        else
        {
            // decide how many rows the output has
            if (%(row_stride)s >= %(row_downsample)s)
            {
                output_row = (input_row - 1) / %(row_stride)s + 1;
            }
            else
            {
                output_row = std::max(0, (input_row - 1 - %(row_downsample)s + %(row_stride)s) / %(row_stride)s) + 1;
            }
            // decide how many columns the output has
            if (%(col_stride)s >= %(col_downsample)s)
            {
                output_col = (input_col - 1) / %(col_stride)s + 1;
            }
            else
            {
                output_col = std::max(0, (input_col - 1 - %(col_downsample)s + %(row_stride)s) / %(col_stride)s) + 1;
            }
        }

        int scoped_col;
        if(%(is_global_scope)s)
        {
            scoped_col = input_col;
        }
        else
        {
            scoped_col = %(col_downsample)s;
        }
        
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != output_channels)
          ||(PyArray_DIMS(%(z)s)[2] != output_row)
          ||(PyArray_DIMS(%(z)s)[3] != output_col)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=output_channels;
          dims[2]=output_row;
          dims[3]=output_col;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        // used for indexing a pool region inside the input
        int row_start, row_end, col_start, col_end;
        dtype_%(x)s collector; // temp var for the value in a region

        if (output_row && output_col)
        {
            for(int n=0; n<PyArray_DIMS(%(x)s)[0]; n++){
                for(int k=0; k<input_channels; k++){
                    for(int r=0; r< output_row; r++){
                        row_start = r * %(row_stride)s;
                        row_end = row_start + %(row_downsample)s;
                        // skip the padding
                        row_start = row_start < %(pad_height)s ? %(pad_height)s : row_start;
                        row_end = row_end > (input_row - %(pad_height)s) ? input_row - %(pad_height)s : row_end;
                        // from padded_img space to img space
                        row_start -= %(pad_height)s;
                        row_end -= %(pad_height)s;
                        // handle the case where no padding, ignore border is True
                        if (%(ignore_border)s)
                        {
                            row_end = row_end > input_row ? input_row : row_end;
                        }
                        for(int c=0; c<output_col; c++){
                            col_start = c * %(col_stride)s;
                            col_end = col_start + %(col_downsample)s;
                            // skip the padding
                            col_start = col_start < %(pad_width)s ? %(pad_width)s : col_start;
                            col_end = col_end > (input_col - %(pad_width)s) ? input_col - %(pad_width)s : col_end;
                            dtype_%(z)s * z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k, r, c)));
                            // change coordinates from padding_img space into img space
                            col_start -= %(pad_width)s;
                            col_end -= %(pad_width)s;
                            // handle the case where no padding, ignore border is True
                            if (%(ignore_border)s)
                            {
                                col_end = col_end > input_col ? input_col : col_end;
                            }
                            // use the first element as the initial value of collector
                            collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k,row_start,col_start)))[0];
                            int max_row = row_start;
                            int max_col = col_start;
                            // go through the pooled region in the unpadded input
                            for(int lr=row_start; lr<row_end; lr++){
                                for(int lc=col_start; lc<col_end; lc++){
                                    dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k,lr,lc)))[0];
                                    //collector = (a > collector) ? a : collector;
                                    if (a > collector) {
                                        collector = a;
                                        max_row = lr;
                                        max_col = lc;
                                    }
                                }
                            }
                            z[0] = collector;
                            
                            if(!%(is_global_scope)s)
                            {
                                max_row -= row_start;
                                max_col -= col_start;
                            }
                            
                            if(%(index_type_int)s == 0)
                            {
                                z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k+input_channels, r, c)));
                                z[0] = max_row * scoped_col + max_col;
                            }
                            if(%(index_type_int)s == 2)
                            {
                                z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k+input_channels, r, c)));
                                z[0] = max_row;
                                z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k+2*input_channels, r, c)));
                                z[0] = max_col;
                            }
                        }
                    }
                }
            }
        }
        '''
        return ccode % locals()

class UnpoolSwitch(Op):
    __props__ = ('ds', 'st', 'index_type', 'index_scope', 'original_input_shape')

    @staticmethod
    def out_shape(ds, imgshape, st, index_type='flattened', original_input_shape=None):
        if len(imgshape) < 3:
            raise TypeError('imgshape must have at least three elements '
                            '(channels, rows, cols)')

        input_channels, input_rows, input_cols = imgshape[-3:]

        if index_type == 'flattened' or index_type == 'space_filling':
            index_size = 2
        elif index_type == 'array':
            index_size = 3
        else:
            raise Exception('Invalid index_type: ' + index_type)

#        if input_channels % index_size != 0:
#            raise Exception('Number of channels is not multiple of index size: '
#                            '%s, %s' % (input_channels, index_size))


        output_channels = input_channels // index_size
        if(original_input_shape is not None):
            output_rows = original_input_shape[2]
            output_cols = original_input_shape[3]
        else:
            output_rows = (input_rows - 1) * st[0] + ds[0]
            output_cols = (input_cols - 1) * st[1] + ds[1]

        rval = list(imgshape[:-3]) + [output_channels, output_rows, output_cols]
        return rval

    def __init__(self, ds, st=None, index_type='flattened', index_scope='local', original_input_shape = None):
        self.ds = tuple(ds)
        if not all([isinstance(d, int) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        if index_type not in ['flattened', 'space_filling', 'array']:
            raise ValueError(
                "MaxPoolSwitch index_type parameter only support 'flattened',"
                " 'space_filling' and 'array'. Got %s" % index_type)
        self.index_type = index_type
        if index_scope not in ['local', 'global']:
            raise ValueError(
                "MaxPoolSwitch index_scope parameter only support 'local'"
                " and 'global'. Got %s" % index_scope)
        self.index_scope = index_scope
        
        if self.index_type == 'flattened' or self.index_type == 'space_filling':
            self.index_size = 2
        elif self.index_type == 'array':
            self.index_size = 3
        
        self.index_type_int = ['flattened', 'space_filling', 'array'].index(self.index_type)

        self.original_input_shape = original_input_shape
        if(self.index_size == 2 and self.index_scope == 'global' and self.original_input_shape is None):
            warnings.warn("Flattened Index and Global Scope may lead loss of index information due to size change. Original input size is recommended as input")
            
        if self.index_type == 'space_filling':
            raise ValueError(
                "MaxPoolSwitch index_type space_filling not implemented")
        

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError("UnpoolSwitch Requires 4 Dimensions, " +  str(x.type.ndim) + " Given")
        x = T.as_tensor_variable(x)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = T.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])        
        
    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = self.out_shape(self.ds, x.shape, self.st, self.index_type, self.original_input_shape)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.zeros(z_shape, dtype=x.dtype)
        else:
            z[0].fill(0)
        
        zz = z[0]
        # number of pooling output rows
        output_row = zz.shape[-2]
        # number of pooling output cols
        output_col = zz.shape[-1]
        output_channels = zz.shape[-3]

        row_downsample, col_downsample = self.ds
        row_stride, col_stride = self.st

        input_row = x.shape[-2]
        input_col = x.shape[-1]
        input_channels = x.shape[-3]

        if input_channels % self.index_size != 0:
            raise Exception('Number of channels is not multiple of index size: '
                            '%s, %s' % (input_channels, self.index_size))
        
        is_global_scope = self.index_scope == 'global'

        if is_global_scope:
            scoped_size = (output_row, output_col)
        else:
            scoped_size = self.ds
            
        for n in xrange(x.shape[0]):
            for k in xrange(output_channels):
                for r in xrange(input_row):
                    row_start = r * row_stride
                    for c in xrange(input_col):
                        col_start = c * col_stride
                        max_val = x[n, k, r, c]
                        
                        if self.index_type_int == 0:
                            max_idx = int(x[n, k+output_channels, r, c])
                            max_row, max_col = np.unravel_index(max_idx, scoped_size)
                        
                        elif self.index_type_int == 2:
                            max_row = int(x[n, k+output_channels, r, c])
                            max_col = int(x[n, k+2*output_channels, r, c])
                        
                        if not is_global_scope:
                            max_row += row_start
                            max_col += col_start
                            
                        zz[n, k, max_row, max_col] = max_val

    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        row_downsample, col_downsample = self.ds
        row_stride, col_stride = self.st
        index_type_int = self.index_type_int
        index_size = self.index_size
        is_global_scope = int(self.index_scope == 'global')
        if(self.original_input_shape is not None):
            use_original_size = 1
            _,_, original_input_rows, original_input_cols = self.original_input_shape
        else:
            use_original_size = 0
            original_input_rows = 0
            original_input_cols = 0
        
        ccode='''
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int output_row, output_col, output_channels;
        int input_row, input_col, input_channels;
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }

        input_channels = PyArray_DIMS(%(x)s)[1];        
        input_row = PyArray_DIMS(%(x)s)[2];
        input_col = PyArray_DIMS(%(x)s)[3];
        
        output_channels = input_channels / %(index_size)s;
        if(%(use_original_size)s)
        {
            output_row = %(original_input_rows)s;
            output_col = %(original_input_cols)s;
        }
        else
        {
            output_row = (input_row - 1) * %(row_stride)s + %(row_downsample)s;
            output_col = (input_col - 1) * %(col_stride)s + %(col_downsample)s;
        }
        
        int scoped_col;
        if(%(is_global_scope)s)
        {
            scoped_col = output_col;
        }
        else
        {
            scoped_col = %(col_downsample)s;
        }

        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != output_channels)
          ||(PyArray_DIMS(%(z)s)[2] != output_row)
          ||(PyArray_DIMS(%(z)s)[3] != output_col)
          )
        {
            if (%(z)s)
            {
                Py_XDECREF(%(z)s);
            }
            npy_intp dims[4] = {0,0,0,0};
            dims[0]=PyArray_DIMS(%(x)s)[0];
            dims[1]=output_channels;
            dims[2]=output_row;
            dims[3]=output_col;
            %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }

        int row_start, col_start;
        int max_idx, max_row, max_col;
        
        for(int n=0; n<PyArray_DIMS(%(x)s)[0]; n++)
        {
            for(int k=0; k<output_channels; k++)
            {
                for(int r=0; r< input_row; r++)
                {
                    row_start = r * %(row_stride)s;
                    
                    for(int c=0; c< input_col; c++)
                    {
                        col_start = c * %(col_stride)s;

                        if(%(index_type_int)s == 0)
                        {
                            max_idx = (int)(((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k+output_channels,r,c)))[0]);
                            max_row = max_idx/scoped_col;
                            max_col = max_idx-max_row*scoped_col;
                        }
                        
                        if(%(index_type_int)s == 2)
                        {
                            max_row = (int)(((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k+output_channels,r,c)))[0]);
                            max_col = (int)(((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k+2*output_channels,r,c)))[0]);
                        }
                        
                        if(!%(is_global_scope)s)
                        {
                            max_row = max_row + row_start;
                            max_col = max_col + col_start;
                        }

                        if(max_row < 0 || max_row >= output_row || max_col < 0 || max_col >= output_col)
                        {
                            PyErr_Format(PyExc_ValueError,
                                            "Max Index %%d, %%d Outside of Image (%%d, %%d)", max_row, max_col, output_row, output_col);
                            %(fail)s;
                        }

                        ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k, max_row, max_col)))[0] = (double)((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k,r,c)))[0];
                    }
                }
            }
        }
        '''
        return ccode % locals()


    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(self.ds, in_shapes[0], self.st, self.index_type, self.original_input_shape)
        return [shp]
        
    def grad(self, inputs, grads):
        x, = inputs
        gz, = grads

        return [UnpoolSwitchGrad(self.ds,
                                 st=self.st, index_type=self.index_type, 
                                 index_scope=self.index_scope)(x, gz)]
        
class UnpoolSwitchGrad(Op):
    __props__ = ('ds', 'st', 'index_type', 'index_scope')

    def __init__(self, ds, st=None, index_type='flattened', index_scope='local'):
        self.ds = tuple(ds)
        if not all([isinstance(d, int) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        
        if index_type not in ['flattened', 'space_filling', 'array']:
            raise ValueError(
                "MaxPoolSwitch index_type parameter only support 'flattened',"
                " 'space_filling' and 'array'. Got %s" % index_type)
        self.index_type = index_type
        
        if index_scope not in ['local', 'global']:
            raise ValueError(
                "MaxPoolSwitch index_scope parameter only support 'local'"
                " and 'global'. Got %s" % index_scope)
        self.index_scope = index_scope
        
        if self.index_type == 'flattened' or self.index_type == 'space_filling':
            self.index_size = 2
        elif self.index_type == 'array':
            self.index_size = 3
        
        self.index_type_int = ['flattened', 'space_filling', 'array'].index(self.index_type)

        if self.index_type == 'space_filling':
            raise ValueError(
                "MaxPoolSwitch index_type space_filling not implemented")

    def make_node(self, x, gz):
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        x = T.as_tensor_variable(x)
        gz = T.as_tensor_variable(gz)

        return Apply(self, [x, gz], [x.type()])
        
    def perform(self, node, inp, out):
        x, gz = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = x.shape
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.zeros(z_shape, dtype=x.dtype)
        else:
            z[0].fill(0)
        
        zz = z[0]
        input_gradient_channels = gz.shape[-3]

        row_downsample, col_downsample = self.ds
        row_stride, col_stride = self.st

        input_row = x.shape[-2]
        input_col = x.shape[-1]
        input_channels = x.shape[-3]

        if input_channels % self.index_size != 0:
            raise Exception('Number of channels is not multiple of index size: '
                            '%s, %s' % (input_channels, self.index_size))
        
        is_global_scope = self.index_scope == 'global'

        if is_global_scope:
            scoped_size = (input_row, input_col)
        else:
            scoped_size = self.ds
            
        for n in xrange(x.shape[0]):
            for k in xrange(input_gradient_channels):
                for r in xrange(input_row):
                    row_start = r * row_stride
                    for c in xrange(input_col):
                        col_start = c * col_stride
                        
                        if self.index_type_int == 0:
                            max_idx = int(x[n, k+input_gradient_channels, r, c])
                            max_row, max_col = np.unravel_index(max_idx, scoped_size)
                        
                        elif self.index_type_int == 2:
                            max_row = x[n, k+input_gradient_channels, r, c]
                            max_col = x[n, k+2*input_gradient_channels, r, c]
                        
                        if not is_global_scope:
                            max_row += row_start
                            max_col += col_start
                            
                        zz[n, k, r, c] = gz[n, k, max_row, max_col]
                        
            
    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]
        
    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        
        x, gz = inp
        z, = out
        fail = sub['fail']
        row_downsample, col_downsample = self.ds
        row_stride, col_stride = self.st
        index_type_int = self.index_type_int
        index_size = self.index_size
        is_global_scope = int(self.index_scope == 'global')

        ccode='''
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int input_row, input_col, input_channels;
        int input_gradient_row, input_gradient_col, input_gradient_channels;
        bool z_zeroed = false;

        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        
        input_channels = PyArray_DIMS(%(x)s)[1];
        input_row = PyArray_DIMS(%(x)s)[2];
        input_col = PyArray_DIMS(%(x)s)[3];

        if(input_channels %% %(index_size)s != 0)
        {
            PyErr_Format(PyExc_ValueError, "Number of channels is not multiple of index size: %%d,  %%d", input_channels, %(index_size)s);
            %(fail)s;
        }

        input_gradient_channels = PyArray_DIMS(%(gz)s)[1];        
        input_gradient_row = PyArray_DIMS(%(gz)s)[2];
        input_gradient_col = PyArray_DIMS(%(gz)s)[3];

        int scoped_col;
        if(%(is_global_scope)s)
        {
            scoped_col = input_col;
        }
        else
        {
            scoped_col = %(col_downsample)s;
        }
        
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != input_channels)
          ||(PyArray_DIMS(%(z)s)[2] != input_row)
          ||(PyArray_DIMS(%(z)s)[3] != input_col)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=input_channels;
          dims[2]=input_row;
          dims[3]=input_col;
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
          z_zeroed = true;
        }

        int row_start, col_start;
        int max_idx, max_row, max_col;
        
        for(int n=0; n < PyArray_DIMS(%(x)s)[0]; n++)
        {
            for(int k=0; k<input_gradient_channels; k++)
            {
                for(int r=0; r < input_row; r++)
                {
                    row_start = r * %(row_stride)s;
                    for(int c=0; c < input_col; c++)
                    {
                        col_start = c * %(col_stride)s;
                        
                        if(%(index_type_int)s == 0)
                        {
                            max_idx = (int)((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k+input_gradient_channels,r,c)))[0];
                            max_row = max_idx/scoped_col;
                            max_col = max_idx-max_row*scoped_col;
                        }
                        else if(%(index_type_int)s == 2)
                        {
                            max_row = (int)(((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k+input_gradient_channels,r,c)))[0]);
                            max_col = (int)(((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,n,k+2*input_gradient_channels,r,c)))[0]);
                        }
                        
                        if(!%(is_global_scope)s)
                        {
                            max_row = max_row + row_start;
                            max_col = max_col + col_start;
                        }

                        if(max_row < 0 || max_row >= input_gradient_row || max_col < 0 || max_col >= input_gradient_col)
                        {
                            PyErr_Format(PyExc_ValueError,
                                         "Max Index %%d, %%d Outside of Image (%%d, %%d)", max_row, max_col, input_gradient_row, input_gradient_col);
                            %(fail)s;
                        }
                        
                        ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k, r, c)))[0] = (double)((dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s,n,k,max_row,max_col)))[0];
                    }
                }
            }
            if(!z_zeroed)
            {
                for(int k=input_gradient_channels; k<input_channels; k++)
                {
                    for(int r=0; r < input_row; r++)
                    {
                        for(int c=0; c < input_col; c++)
                        {
                            ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, n, k, r, c)))[0] = 0;
                        }
                    }
                }
            }
        }
        '''
        return ccode % locals()
