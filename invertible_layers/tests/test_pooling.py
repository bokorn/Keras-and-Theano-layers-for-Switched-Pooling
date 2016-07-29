from __future__ import absolute_import, print_function, division

from itertools import product
import unittest
import six.moves.builtins as builtins

import numpy

import theano
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.signal.pool import (Pool, pool_2d)

from theano import function


class TestDownsampleFactorMax(utt.InferShapeTester):

    @staticmethod
    def numpy_max_pool_2d(input, ds, ignore_border=False, mode='max'):
        '''Helper function, implementing pool_2d in pure numpy'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))
        xi = 0
        yi = 0
        if not ignore_border:
            if input.shape[-2] % ds[0]:
                xi += 1
            if input.shape[-1] % ds[1]:
                yi += 1
        out_shp = list(input.shape[:-2])
        out_shp.append(input.shape[-2] / ds[0] + xi)
        out_shp.append(input.shape[-1] / ds[1] + yi)
        output_val = numpy.zeros(out_shp)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii = i * ds[0]
                for j in range(output_val.shape[-1]):
                    jj = j * ds[1]
                    patch = input[k][ii:ii + ds[0], jj:jj + ds[1]]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride_padding(
            x, ds, ignore_border=True, st=None, padding=(0, 0), mode='max'):
        assert ignore_border
        pad_h = padding[0]
        pad_w = padding[1]
        h = x.shape[-2]
        w = x.shape[-1]
        assert ds[0] > pad_h
        assert ds[1] > pad_w

        def pad_img(x):
            y = numpy.zeros(
                (x.shape[0], x.shape[1],
                 x.shape[2] + pad_h * 2, x.shape[3] + pad_w * 2),
                dtype=x.dtype)
            y[:, :, pad_h:(x.shape[2] + pad_h), pad_w:(x.shape[3] + pad_w)] = x

            return y
        img_rows = h + 2 * pad_h
        img_cols = w + 2 * pad_w
        out_r = (img_rows - ds[0]) // st[0] + 1
        out_c = (img_cols - ds[1]) // st[1] + 1
        out_shp = list(x.shape[:-2])
        out_shp.append(out_r)
        out_shp.append(out_c)
        ds0, ds1 = ds
        st0, st1 = st
        output_val = numpy.zeros(out_shp)
        y = pad_img(x)
        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average
        inc_pad = mode == 'average_inc_pad'

        for k in numpy.ndindex(*x.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii_st = i * st[0]
                ii_end = builtins.min(ii_st + ds[0], img_rows)
                if not inc_pad:
                    ii_st = builtins.max(ii_st, pad_h)
                    ii_end = builtins.min(ii_end, h + pad_h)
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    jj_end = builtins.min(jj_st + ds[1], img_cols)
                    if not inc_pad:
                        jj_st = builtins.max(jj_st, pad_w)
                        jj_end = builtins.min(jj_end, w + pad_w)
                    patch = y[k][ii_st:ii_end, jj_st:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    @staticmethod
    def numpy_max_pool_2d_stride(input, ds, ignore_border=False, st=None,
                                 mode='max'):
        '''Helper function, implementing pool_2d in pure numpy
           this function provides st input to indicate the stide size
           for the pooling regions. if not indicated, st == sd.'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim,'
                                      ' shape is %s'
                                      % str(input.shape))

        if st is None:
            st = ds
        img_rows = input.shape[-2]
        img_cols = input.shape[-1]

        out_r = 0
        out_c = 0
        if img_rows - ds[0] >= 0:
            out_r = (img_rows - ds[0]) // st[0] + 1
        if img_cols - ds[1] >= 0:
            out_c = (img_cols - ds[1]) // st[1] + 1

        if not ignore_border:
            if out_r > 0:
                if img_rows - ((out_r - 1) * st[0] + ds[0]) > 0:
                    rr = img_rows - out_r * st[0]
                    if rr > 0:
                        out_r += 1
            else:
                if img_rows > 0:
                        out_r += 1
            if out_c > 0:
                if img_cols - ((out_c - 1) * st[1] + ds[1]) > 0:
                    cr = img_cols - out_c * st[1]
                    if cr > 0:
                        out_c += 1
            else:
                if img_cols > 0:
                        out_c += 1

        out_shp = list(input.shape[:-2])
        out_shp.append(out_r)
        out_shp.append(out_c)

        func = numpy.max
        if mode == 'sum':
            func = numpy.sum
        elif mode != 'max':
            func = numpy.average

        output_val = numpy.zeros(out_shp)
        for k in numpy.ndindex(*input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii_st = i * st[0]
                ii_end = builtins.min(ii_st + ds[0], img_rows)
                for j in range(output_val.shape[-1]):
                    jj_st = j * st[1]
                    jj_end = builtins.min(jj_st + ds[1], img_cols)
                    patch = input[k][ii_st:ii_end, jj_st:jj_end]
                    output_val[k][i, j] = func(patch)
        return output_val

    def test_DownsampleFactorMax(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        # generate random images
        maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
        imval = rng.rand(4, 2, 16, 16)
        images = tensor.dtensor4()
        for maxpoolshp, ignore_border, mode in product(maxpoolshps,
                                                       [True, False],
                                                       ['max',
                                                        'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad']):
                # print 'maxpoolshp =', maxpoolshp
                # print 'ignore_border =', ignore_border

                # Pure Numpy computation
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp,
                                                          ignore_border,
                                                          mode=mode)
                output = pool_2d(images, maxpoolshp, ignore_border,
                                 mode=mode)
                f = function([images, ], [output, ])
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

                # Pool op
                maxpool_op = Pool(maxpoolshp,
                                  ignore_border=ignore_border,
                                  mode=mode)(images)

                output_shape = Pool.out_shape(imval.shape, maxpoolshp,
                                              ignore_border=ignore_border)
                utt.assert_allclose(numpy.asarray(output_shape), numpy_output_val.shape)
                f = function([images], maxpool_op)
                output_val = f(imval)
                utt.assert_allclose(output_val, numpy_output_val)

    def test_DownsampleFactorMaxStride(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1, 1), (3, 3), (5, 3), (16, 16))
        stridesizes = ((1, 1), (3, 3), (5, 7),)
        # generate random images
        imval = rng.rand(4, 10, 16, 16)
        # The same for each mode
        outputshps = (
            (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
            (4, 10, 16, 16), (4, 10, 6, 6), (4, 10, 4, 3),
            (4, 10, 14, 14), (4, 10, 5, 5), (4, 10, 3, 2),
            (4, 10, 14, 14), (4, 10, 6, 6), (4, 10, 4, 3),
            (4, 10, 12, 14), (4, 10, 4, 5), (4, 10, 3, 2),
            (4, 10, 12, 14), (4, 10, 5, 6), (4, 10, 4, 3),
            (4, 10, 1, 1), (4, 10, 1, 1), (4, 10, 1, 1),
            (4, 10, 1, 1), (4, 10, 1, 1), (4, 10, 1, 1),)

        images = tensor.dtensor4()
        indx = 0
        for mode, maxpoolshp, ignore_border in product(['max',
                                                        'sum',
                                                        'average_inc_pad',
                                                        'average_exc_pad'],
                                                       maxpoolshps,
                                                       [True, False]):
                for stride in stridesizes:
                    outputshp = outputshps[indx % len(outputshps)]
                    indx += 1
                    # Pool op
                    numpy_output_val = \
                        self.numpy_max_pool_2d_stride(imval, maxpoolshp,
                                                      ignore_border, stride,
                                                      mode)
                    assert numpy_output_val.shape == outputshp, (
                        "outshape is %s, calculated shape is %s"
                        % (outputshp, numpy_output_val.shape))
                    maxpool_op = \
                        Pool(maxpoolshp,
                             ignore_border=ignore_border,
                             st=stride, mode=mode)(images)
                    f = function([images], maxpool_op)
                    output_val = f(imval)
                    utt.assert_allclose(output_val, numpy_output_val)

if __name__ == '__main__':
    unittest.main()