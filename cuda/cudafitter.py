#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:58:52 2020

@author: mints
"""
from .kernels import get_p_zyt, get_mhat, balance, update_params, fill_2and3
from .gpu import meminfo, get_fit_gpu_dims
from numba import cuda
import numpy as np
import logging


SQRT_TWOPI = 2.506628274631


class CudaFitter():
    MIN_SIGMA = 1e-3
    PARAM_PRECISION = 1e-5

    def __init__(self, magnitudes, errs, sed_count=2):
        self.logger = logging.getLogger('Fit_internals')
        self.logger.setLevel(logging.INFO)

        self.mags = magnitudes
        self.errs = errs
        self.max_size, _, _ = get_fit_gpu_dims(sed_count, self.mags.shape[1])
        if len(self.mags) > self.max_size - 1:
            self.logger.info('Too many objects, cutting to %d', self.max_size)
            choice = np.arange(self.max_size - 1, dtype=int)
            self.mags = self.mags[choice]
            self.errs = self.errs[choice]
        self.logger.info('Running SED fit on %d objects', self.mags.shape[0])
        self.points = magnitudes.shape[0]
        self.bands = magnitudes.shape[1]
        self.sed_count = sed_count
        self.stream = cuda.stream()
        self.cu_mags = cuda.to_device(np.array(self.mags),
                                      stream=self.stream)
        self.cu_errs = cuda.to_device(np.array(self.errs),
                                      stream=self.stream)
        self.cu_mhat = cuda.to_device(np.zeros((len(self.cu_mags),
                                                self.sed_count)),
                                      stream=self.stream)
        self.stream.synchronize()
        self.params = []

    def init_params(self):
        """
        Initiate fit parameters.
        Last template is built from median magnitude values per band,
        all others are distributed evenly between a flat template
        and the last one.
        """
        w = np.ones(self.sed_count) / self.sed_count
        sed = np.ones((self.sed_count, self.bands))
        template = np.nanmedian(self.mags, axis=0)
        template = template - template.mean()
        for i in range(self.sed_count):
            sed[i] = template * float(i) / (self.sed_count * (self.bands - 1))
        sigma = np.ones((self.sed_count, self.bands)) * 0.4
        self.params = [w, sed, sigma]

    def is_converged(self, params):
        """Test if iterations converged. Difference in
        at least one parameter should be no less than PARAM_PRECISION.

        Args:
            params : New parameter value.

        """
        for i in range(3):
            if np.any(np.abs(self.params[i] - params[i]) >
                      self.PARAM_PRECISION):
                return False
        return True

    def get_chi2(self):
        threadsperblock = 256
        blockspergrid_x = 256
        blockspergrid_y = 16
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        cu_w = cuda.to_device(self.params[0], stream=self.stream)
        cu_sed = cuda.to_device(self.params[1], stream=self.stream)
        cu_sederr = cuda.to_device(self.params[2], stream=self.stream)
        p_zyt = np.empty((len(self.cu_mags), self.sed_count))
        cu_p_zyt = cuda.to_device(p_zyt, stream=self.stream)
        get_p_zyt[blockspergrid, threadsperblock, self.stream](
            self.cu_mags, self.cu_errs,
            self.cu_mhat,
            cu_w, cu_sed,
            cu_sederr, cu_p_zyt)
        return cu_p_zyt.copy_to_host(stream=self.stream)

    def fit(self, maxiter=40, reinit_params=True):
        if reinit_params:
            self.init_params()
        # Initialize to "infinite" values, to get at least one iteration
        params = [np.ones(self.sed_count),
                  np.ones_like(self.params[1]) * 1e10,
                  np.ones_like(self.params[1]) * 1e10]
        iterations = 0
        # Configure CUDA arrays
        cu_w = cuda.to_device(self.params[0], stream=self.stream)
        cu_sed = cuda.to_device(self.params[1], stream=self.stream)
        cu_sederr = cuda.to_device(self.params[2], stream=self.stream)
        p_zyt = np.empty((len(self.cu_mags), self.sed_count))
        cu_p_zyt = cuda.to_device(p_zyt, stream=self.stream)
        p2 = cuda.device_array((p_zyt.shape[0], self.sed_count,
                                self.bands), stream=self.stream)
        p3 = cuda.device_array((p_zyt.shape[0], self.sed_count,
                                self.bands), stream=self.stream)
        cu_tmu = cuda.device_array((self.sed_count, self.bands),
                                   stream=self.stream)
        cu_tsigma = cuda.device_array((self.sed_count, self.bands),
                                      stream=self.stream)
        meminfo(self.logger)

        # Configure the blocks
        _, grid_size, block_size = get_fit_gpu_dims(self.sed_count,
                                                    self.bands)
        threadsperblock = block_size
        blockspergrid_x = grid_size
        blockspergrid_y = grid_size
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        output = []
        while iterations < maxiter and not self.is_converged(params):
            self.stream.synchronize()
            self.logger.debug('Get mhat')
            get_mhat[blockspergrid, threadsperblock, self.stream](
                self.cu_mags, self.cu_errs,
                cu_sed, cu_sederr,
                self.cu_mhat)
            self.logger.debug('Get p_zyt')
            get_p_zyt[blockspergrid, threadsperblock, self.stream](
                self.cu_mags, self.cu_errs,
                self.cu_mhat,
                cu_w, cu_sed,
                cu_sederr, cu_p_zyt)
            l_value = cu_p_zyt.copy_to_host(stream=self.stream)
            l_value = np.sum(np.log10(l_value.sum(axis=1)))
            self.logger.debug('Balance')
            balance[blockspergrid_x, threadsperblock, self.stream](cu_p_zyt)
            self.logger.debug('Fill p2 & p3')
            fill_2and3[blockspergrid, threadsperblock, self.stream](
                cu_p_zyt, p2, p3,
                cu_sederr, self.cu_errs)
            p2_host = p2.copy_to_host(stream=self.stream)
            sum_p2_host = np.nansum(p2_host, axis=0)
            sum_p2_host[~(sum_p2_host > 0)] = np.inf
            self.stream.synchronize()
            self.logger.debug('Update params')
            update_params[blockspergrid, threadsperblock, self.stream](
                p2, p3,
                self.cu_mags, self.cu_mhat,
                cu_sed, cu_tmu, cu_tsigma)
            self.stream.synchronize()
            self.logger.debug('ALmost there')
            p_zyt = cu_p_zyt.copy_to_host(stream=self.stream)
            tw = np.nanmean(p_zyt, axis=0)
            #tw = cu_w.copy_to_host(stream=self.stream)
            tw = tw / tw.sum()
            tmu = cu_tmu.copy_to_host(stream=self.stream)
            tmu = tmu / sum_p2_host
            tmu -= np.nanmean(tmu, axis=1)[:, np.newaxis]
            tsigma = cu_tsigma.copy_to_host(stream=self.stream)
            tsigma = np.sqrt(tsigma / sum_p2_host)
            tsigma[tsigma < self.MIN_SIGMA] = self.MIN_SIGMA
            params = [np.copy(p) for p in self.params]
            self.params = [np.copy(tw),
                           np.copy(tmu),
                           np.copy(tsigma)]
            iterations += 1
            cu_w = cuda.to_device(tw, stream=self.stream)
            cu_sed = cuda.to_device(tmu, stream=self.stream)
            cu_sederr = cuda.to_device(tsigma, stream=self.stream)
            output.append([tw, tmu, tsigma, l_value])
            self.logger.debug('Ready for next iteration')

        self.logger.info('Finished after %d iterations', iterations)
        return self.params, iterations, l_value, output
