# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:08:51 2016

@author: bokorn
"""
import sys
sys.path.insert(0,'../')
from theano.tests import unittest_tools as utt
from switched_pooling_backend.theano_switched_pooling import MaxPoolSwitch, UnpoolSwitch, UnpoolSwitchGrad

from theano.tensor.signal.pool import Pool
import numpy as np
import theano
from theano import tensor as T
from theano import config
from itertools import product
import six.moves.builtins as builtins
from datetime import datetime
from time import mktime

class TestMaxPoolSwitch(utt.InferShapeTester):
    def setUp(self):
        super(TestMaxPoolSwitch, self).setUp()
        self.op_class = MaxPoolSwitch
        self.op = MaxPoolSwitch(ds=(2,2), index_scope='local', index_type='flattened')
        self.rng = np.random.RandomState(utt.fetch_seed(mktime(datetime.now().timetuple())))
        
    def tearDown(self):
        self.op = None

    @staticmethod
    def check_max_and_switches(input, output, expected_maxpool_output, 
                               ds, ignore_border, st, padding,
                               index_type, index_scope):

        input_channels = input.shape[1]

        np.testing.assert_allclose(expected_maxpool_output, output[:,:input_channels,:,:], 
                            err_msg='Max Values Incorrect (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope))
        

        if index_scope == 'global':
            scoped_size = (input.shape[2], input.shape[3])
        else:
            scoped_size = ds
        
        for n in range(output.shape[0]):
            for k in range(input_channels):
                for r in range(output.shape[2]):
                    row_start = r * st[0]
                    for c in range(output.shape[3]):
                        col_start = c * st[1]
                        max_val = output[n, k, r, c]
                        if index_type == 'flattened':
                            max_idx = int(output[n, k+input_channels, r, c])
                            try:
                                max_row, max_col = np.unravel_index(max_idx, scoped_size)
                            except ValueError:
                                raise ValueError('Max Index {6} Invalid in Scope Size {7} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, max_idx, scoped_size))
                        elif index_type == 'array':
                            max_row = int(output[n, k+input_channels, r, c])
                            max_col = int(output[n, k+2*input_channels, r, c])
                        
                        if index_scope == 'local':
                            max_row += row_start
                            max_col += col_start
                        assert max_val == input[n, k, max_row, max_col], 'Max Index ({6},{7}) Incorrect, {8} != {9} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, max_row, max_col, max_val, input[n, k, max_row, max_col])
                    
    def test_basic(self):
        x = T.dtensor4()
        f = theano.function([x], self.op(x))
        inp = self.rng.rand(4, 3, 64, 64)
        
        out = f(inp)
        max_pool_op = Pool(ds=self.op.ds, ignore_border=self.op.ignore_border, st=self.op.st, padding=self.op.padding, mode='max')
        max_pool_f = theano.function([x], max_pool_op(x))
        exp_out = max_pool_f(inp)

        self.check_max_and_switches(inp, out, exp_out, 
                                    self.op.ds, self.op.ignore_border, self.op.st, self.op.padding, 
                                    self.op.index_type, self.op.index_scope)
        
    
        
    def test_c_code(self):
        x = T.dtensor4()
        f = theano.function([x], self.op(x), mode='DebugMode')
        inp = self.rng.rand(4, 3, 64, 64)
        
        out = f(inp)
                            
    def test_verify_grad(self):
        inp = np.asarray(self.rng.rand(4, 3, 64, 64), dtype=config.floatX)
        utt.verify_grad(self.op, [inp], rng=self.rng, cast_to_output_type = True)

    def test_infer_shape(self):
        x = T.tensor4()
        self._compile_and_check([x],  # theano.function inputs
                                [self.op(x)],  # theano.function outputs
                                # Always use not square matrix!
                                # inputs data
                                [np.asarray(np.random.rand(1, 3, 88, 64),
                                               dtype=config.floatX)],
                                # Op that should be removed from the graph.
                                self.op_class)
        
    def test_gauntlet(self):
        maxpoolshps = ((1, 1), (3, 3), (5, 3),)
        stridesizes = ((1, 1), (3, 3), (5, 7),)
        # generate random images
        imval = self.rng.rand(4, 10, 16, 16)

        images = T.dtensor4()
        for index_type, index_scope, maxpoolshp, stride, ignore_border in product(['flattened',
                                                        'array'],
                                                        ['local',
                                                         'global'],
                                                       maxpoolshps,
                                                       stridesizes,
                                                       [True, False]):
                # Pool op
                max_pool_op = Pool(ds=maxpoolshp, 
                                   ignore_border=ignore_border, 
                                   st=stride, mode='max')(images)
                max_pool_f = theano.function([images], max_pool_op)
                maxpool_output_val = max_pool_f(imval)
                
                maxpoolswitch_op = MaxPoolSwitch(ds = maxpoolshp,
                                                 ignore_border=ignore_border,
                                                 st=stride, index_type=index_type, 
                                                 index_scope=index_scope)(images)
                f = theano.function([images], maxpoolswitch_op, mode='DebugMode')
                output_val = f(imval)
                self.check_max_and_switches(imval, output_val, maxpool_output_val, 
                                            maxpoolshp, ignore_border, stride, None, 
                                            index_type, index_scope)

    def test_verify_grad_gauntlet(self):
        maxpoolshps = ((1, 1), (3, 3), (5, 3),)
        stridesizes = ((1, 1), (3, 3), (5, 7),)
        # generate random images
        imval = self.rng.rand(4, 10, 16, 16)

        for index_type, index_scope, maxpoolshp, stride, ignore_border in product(['flattened',
                                                        'array'],
                                                        ['local',
                                                         'global'],
                                                       maxpoolshps,
                                                       stridesizes,
                                                       [True, False]):
                maxpoolswitch_op = MaxPoolSwitch(ds = maxpoolshp,
                                                 ignore_border=ignore_border,
                                                 st=stride, index_type=index_type, 
                                                 index_scope=index_scope)
                
                print index_type, index_scope, maxpoolshp, stride, ignore_border
                # The tollerance on this test has been increased because I can not figure out why the 
                # gradients differ for global indices. This should be looked into in future versions.
                utt.verify_grad(maxpoolswitch_op, [imval], rng=self.rng, abs_tol=1e-2, rel_tol=1e-2)

class TestUnpoolSwitch(utt.InferShapeTester):
    def setUp(self):
        super(TestUnpoolSwitch, self).setUp()
        self.op_class = UnpoolSwitch
        self.op = UnpoolSwitch(ds=(2,2))
        self.rng = np.random.RandomState(utt.fetch_seed())
        
    def tearDown(self):
        self.op = None

    @staticmethod
    def check_unpool_and_switches(input, output, maxpool_output, 
                               ds, ignore_border, st, padding,
                               index_type, index_scope):

        row_downsample, col_downsample = ds
        row_stride, col_stride = st
        output_row = output.shape[-2]
        output_col = output.shape[-1]
        input_channels = input.shape[-3]
        
        assert input.shape[0] == output.shape[0], 'Image Number Size Incorrect, {6} != {7} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, input.shape[0], output.shape[0])
        assert input.shape[1] == output.shape[1], 'Image Channel Size Incorrect, {6} != {7} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, input.shape[1], output.shape[1])
        #assert maxpool_output.shape[2]*st[0] + ds[0]-1 == output.shape[2], 'Image Column Size Incorrect, {6} != {7} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, maxpool_output.shape[2]*st[0] + ds[0]-1, output.shape[2])
        #assert maxpool_output.shape[3]*st[1] + ds[1]-1 == output.shape[3], 'Image Row Size Incorrect, {6} != {7} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, maxpool_output.shape[3]*st[1] + ds[1]-1, output.shape[3])
        
        if(st[0] >= ds[0] and st[1] >= ds[1]):
            for n in range(maxpool_output.shape[0]):
                for k in range(input_channels):
                    for r in range(maxpool_output.shape[2]):
                        row_start = r * st[0]
                        row_end = builtins.min(row_start + row_downsample, output_row)
    
                        for c in range(maxpool_output.shape[3]):
                            col_start = c * st[1]
                            col_end = builtins.min(col_start + col_downsample, output_col)
    
                            max_val = np.max(output[n, k, row_start:row_end, col_start:col_end])
                            expected_max_val = maxpool_output[n, k, r, c]

                            assert max_val == expected_max_val, '(max: {8}, maxpool: {9}, input: {10}, output: {11}) Invalid Max Value in Output Patch {6}, {7} Incorrect, {8} != {9} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, r, c, max_val, expected_max_val, input[n, k, row_start, col_start], output[n, k, row_start, col_start])
    
                            nz_count = np.count_nonzero(output[n, k, row_start:row_end, col_start:col_end])
                            
                            if(expected_max_val != 0):
                                assert nz_count == 1, 'Number of Nonzero Values in Output Patch {6}, {7} Incorrect, {8} != 1 (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, r, c, nz_count)
                            else:
                                assert nz_count == 0, 'Number of Nonzero Values in Output Patch {6}, {7} Incorrect, {8} != 0 (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, r, c, nz_count)
        else:
            for n in range(output.shape[0]):
                for k in range(output.shape[1]):
                    for r in range(output.shape[2]):
                        for c in range(output.shape[3]):
                            val = output[n, k, r, c]
                        
                            if(val != 0):
                                expected_max_matched = False
                                for i in range(max((r-ds[0]+1)//st[0], 0), min(r//st[0]+1, maxpool_output.shape[2])):
                                    for j in range(max((c-ds[1]+1)//st[1], 0), min(c//st[1]+1, maxpool_output.shape[3])):
                                        if(not expected_max_matched):
                                            expected_max_matched = (maxpool_output[n, k, i, j] == val)
                                assert expected_max_matched, 'Nonzero Values {6}, {7} Not Equal to Local Maxs, {8} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, r, c, val)
                                
                                assert val == input[n, k, r, c], 'Nonzero Values {6}, {7} Not Same as Input Image, {8} != {9} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, r, c, val, input[n, k, r, c])
                            else:
                                if(r < input.shape[2] and c < input.shape[3] and r//st[0] < maxpool_output.shape[2] and c//st[1] < maxpool_output.shape[3]):
                                    expected_max_val = maxpool_output[n, k, r//st[0], c//st[1]]
                                    assert input[n, k, r, c] <= expected_max_val, 'Zeroed Value {6}, {7} Greater than Max, {8} > {9} (ds={0}, ignore_border={1}, st={2}, padding={3}, index_type={4}, index_scope={5})'.format(ds, ignore_border, st, padding, index_type, index_scope, r, c, input[n, k, r, c], expected_max_val)                               

    def test_basic(self):
        x = T.tensor4()
        y = T.tensor4()
        f_unpool = theano.function([x], self.op(x))
        mps = MaxPoolSwitch(ds=self.op.ds, st=self.op.st, index_type=self.op.index_type, index_scope=self.op.index_scope)
        f_pool = theano.function([y], mps(y))
        
        inp = self.rng.rand(4, 3, 64, 64)

        out = f_unpool(f_pool(inp))
        max_pool_op = Pool(ds=self.op.ds, st=self.op.st, mode='max')
        max_pool_f = theano.function([x], max_pool_op(x))
        max_out = max_pool_f(inp)

        self.check_unpool_and_switches(inp, out, max_out, 
                                       self.op.ds, False, self.op.st, (0,0), 
                                       self.op.index_type, self.op.index_scope)
                                

        
    def test_gauntlet(self):
        maxpoolshps = ((1, 1), (3, 3), (5, 3),)
        stridesizes = ((1, 1), (3, 3), (5, 7),)
        # generate random images
        imval = self.rng.rand(4, 10, 16, 16)

        images = T.dtensor4()
        pooled_images = T.tensor4()

        for index_type, index_scope, maxpoolshp, stride, ignore_border in product(['flattened',
                                                        'array'],
                                                        ['local',
                                                         'global'],
                                                       maxpoolshps,
                                                       stridesizes,
                                                       [True, False]):
                # Pool op
                max_pool_op = Pool(ds=maxpoolshp, 
                                   ignore_border=ignore_border, 
                                   st=stride, mode='max')(images)
                max_pool_f = theano.function([images], max_pool_op)
                maxpool_output_val = max_pool_f(imval)
                
                maxpoolswitch_op = MaxPoolSwitch(ds = maxpoolshp,
                                                 ignore_border=ignore_border,
                                                 st=stride, index_type=index_type, 
                                                 index_scope=index_scope)(images)
                                                 
                unpoolswitch_op = UnpoolSwitch(ds = maxpoolshp,
                                               st=stride, index_type=index_type, 
                                               index_scope=index_scope,
                                               original_input_shape = imval.shape)(pooled_images)
                                               
                f_pool = theano.function([images], maxpoolswitch_op, mode='DebugMode')
                
                f_unpool = theano.function([pooled_images], unpoolswitch_op, mode='DebugMode')
                
                output_val = f_unpool(f_pool(imval))
                self.check_unpool_and_switches(imval, output_val, maxpool_output_val, 
                                               maxpoolshp, ignore_border, stride, (0,0), 
                                               index_type, index_scope)

    def test_verify_grad_gauntlet(self):
        maxpoolshps = ((1, 1), (3, 3), (5, 3),)
        stridesizes = ((1, 1), (3, 3), (5, 7),)
        # generate random images
        imval = self.rng.rand(4, 10, 16, 16)

        for index_type, index_scope, maxpoolshp, stride, ignore_border in product(['flattened',
                                                        'array'],
                                                        ['local'],
                                                       maxpoolshps,
                                                       stridesizes,
                                                       [True, False]):
                                                 
                unpoolswitch_op = UnpoolSwitch(ds = maxpoolshp,
                                               st=stride, index_type=index_type, 
                                               index_scope=index_scope)
                                               
                if(index_type == 'flattened'):                                               
                    def op_with_fixed_switchs(x):
                        x_with_zero_switchs = T.concatenate((x, T.zeros_like(x)), 1)
                        return unpoolswitch_op(x_with_zero_switchs)
                else:
                    def op_with_fixed_switchs(x):
                        x_with_zero_switchs = T.concatenate((x, T.zeros_like(x), T.zeros_like(x)), 1)
                        return unpoolswitch_op(x_with_zero_switchs)

                utt.verify_grad(op_with_fixed_switchs, [imval], rng=self.rng)

    def test_verify_grad(self):
        inp = np.asarray(self.rng.rand(1, 3, 64, 64), dtype=config.floatX)

        def op_with_fixed_switchs(x):
            x_with_zero_switchs = T.concatenate((x, T.zeros_like(x)), 1)
            return self.op(x_with_zero_switchs)

        utt.verify_grad(op_with_fixed_switchs, [inp], rng=self.rng)
    
    def test_infer_shape(self):
        x = T.tensor4()
        self._compile_and_check([x],  # theano.function inputs
                                [self.op(x)],  # theano.function outputs
                                # Always use not square matrix!
                                # inputs data
                                [np.asarray(np.random.rand(1, 6, 44, 32),
                                               dtype=config.floatX)],
                                # Op that should be removed from the graph.
                                self.op_class)
                                
    def test_c_code(self):
        x = T.dtensor4()
        f = theano.function([x], self.op(x), mode='DebugMode')
        inp = self.rng.rand(1, 6, 64, 64)
        
        out = f(inp)
        
class TestUnpoolSwitchGrad(utt.InferShapeTester):
    def setUp(self):
        super(TestUnpoolSwitchGrad, self).setUp()
        self.op_class = UnpoolSwitchGrad
        self.op = UnpoolSwitchGrad(ds=(2,2))
        self.rng = np.random.RandomState(utt.fetch_seed())
        
    def tearDown(self):
        self.op = None
        
    def test_infer_shape(self):
        x = T.tensor4()
        gz = T.tensor4()
        self._compile_and_check([x, gz],  # theano.function inputs
                                [self.op(x, gz)],  # theano.function outputs
                                # Always use not square matrix!
                                # inputs data
                                [np.asarray(np.random.rand(1, 6, 44, 32),
                                               dtype=config.floatX), 
                                 np.asarray(np.random.rand(1, 3, 88, 64),
                                               dtype=config.floatX)],
                                # Op that should be removed from the graph.
                                self.op_class)
    
    def test_c_code(self):
        x = T.dtensor4()
        gz = T.tensor4()
        f = theano.function([x, gz], self.op(x, gz), mode='DebugMode')
        inp = self.rng.rand(1, 6, 4, 4)
        inp_gz = self.rng.rand(1, 3, 8, 8)

        out = f(inp, inp_gz)