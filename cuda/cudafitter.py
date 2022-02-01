#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:58:52 2020

@author: mints
"""
import math
from numba import cuda
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from semiphore_public.utils.cosmology import unnormed_schechter, flat_schechter


SQRT_TWOPI = 2.506628274631


@cuda.jit
def get_mstar(mag, err, sed, sed_err, output):
    pos, col = cuda.grid(2)
    if col >= len(sed) or pos >= len(mag):
        return
    part1 = 0
    part2 = 0
    for i in range(sed.shape[1]):
        if not math.isnan(mag[pos, i]) and not math.isnan(err[pos, i]):
            denom = 1. / (err[pos, i]**2 + sed_err[col, i]**2)
            part1 += (mag[pos, i] - sed[col, i]) * denom
            part2 += denom
    output[pos, col] = part1 / part2


@cuda.jit
def get_p_zyt(m, err, m0, w, mu, sigma, p_zyt_out):
    """
    Calculate (A4).
    """
    row, col = cuda.grid(2)
    if col >= len(w) or row >= len(m):
        return
    p = w[col]
    for b in range(m.shape[1]):
        if not math.isnan(m[row, b]) and not math.isnan(err[row, b]):
            delta2 = err[row, b]**2 + sigma[col, b]**2
            p *= math.exp(-0.5 *
                          (m[row, b] - m0[row, col] - mu[col, b]) *
                          (m[row, b] - m0[row, col] - mu[col, b])
                          / delta2) / (SQRT_TWOPI * math.sqrt(delta2))
    p_zyt_out[row, col] = p


@cuda.jit
def get_chy2(m, err, m0,
             w, mu, sigma, p_zyt_out):
    row, col = cuda.grid(2)
    if col >= len(w) or row >= len(m):
        return
    p = 0
    for b in range(m.shape[1]):
        if not math.isnan(m[row, b]) and not math.isnan(err[row, b]):
            delta2 = err[row, b]**2 + sigma[col, b]**2
            p += (m[row, b] - m0[row, col] - mu[col, b]) * \
                 (m[row, b] - m0[row, col] - mu[col, b]) / delta2
    p_zyt_out[row, col] = p


@cuda.jit
def balance(p_zyt_out):
    """
    Calculate (A5).
    """
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < p_zyt_out.shape[0]:
        summ = 0
        for i in range(p_zyt_out.shape[1]):
            summ += p_zyt_out[pos, i]
        if summ > 0:
            for i in range(p_zyt_out.shape[1]):
                p_zyt_out[pos, i] = p_zyt_out[pos, i] / summ


@cuda.jit
def fill_2and3(p1, p2, p3, sigma, delta):
    """
    Calculate (1+delta^2/sigma^2)^n for n=1 (p2) and n=2 (p3).
    """
    row, col = cuda.grid(2)
    if col < p1.shape[1] and row < p1.shape[0]:
        for b in range(p2.shape[2]):
            # Update to speed-up (A9) and (A11)
            p2[row, col, b] = p1[row, col] / \
                (1.0 + (delta[row, b]/sigma[col, b])**2)
            p3[row, col, b] = p1[row, col] / \
                (1.0 + (delta[row, b]/sigma[col, b])**2)**2


@cuda.jit
def update_params(p2, p3, mags, m0, mu, tmu, tsigma):
    """
    Updating mu_t and sigma_t.
    """
    component, band = cuda.grid(2)
    if component >= p2.shape[1] or band >= p2.shape[2]:
        return
    tmu[component, band] = 0
    tsigma[component, band] = 0
    for i in range(p2.shape[0]):
        if not math.isnan(mags[i, band]) and \
                not math.isnan(p2[i, component, band]) and \
                not math.isnan(p3[i, component, band]):
            tmu[component, band] += (mags[i, band] - m0[i, component]) * \
                p2[i, component, band]
            tsigma[component, band] += \
                (mags[i, band] - m0[i, component] - mu[component, band])**2 * \
                p3[i, component, band]


class CudaFitter():
    MIN_SIGMA = 1e-3
    PARAM_PRECISION = 1e-5

    def __init__(self, magnitudes, delta, sed_count=2):
        self.mags = magnitudes
        self.delta = delta
        self.points = magnitudes.shape[0]
        self.bands = magnitudes.shape[1]
        self.sed_count = sed_count
        self.stream = cuda.stream()
        self.cu_mags = cuda.to_device(np.array(magnitudes[:256**2 - 1]),
                                      stream=self.stream)
        self.cu_errs = cuda.to_device(np.array(delta[:256**2 - 1]),
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
        p2 = cuda.to_device(np.zeros((p_zyt.shape[0], self.sed_count,
                                      self.bands)), stream=self.stream)
        p3 = cuda.to_device(np.zeros((p_zyt.shape[0], self.sed_count,
                                      self.bands)), stream=self.stream)
        cu_tmu = cuda.to_device(np.zeros_like(self.params[1]),
                                stream=self.stream)
        cu_tsigma = cuda.to_device(np.zeros_like(self.params[1]),
                                   stream=self.stream)

        # Configure the blocks
        threadsperblock = 256
        blockspergrid_x = 256
        blockspergrid_y = 16
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        output = []
        while iterations < maxiter and not self.is_converged(params):
            self.stream.synchronize()
            get_mstar[blockspergrid, threadsperblock, self.stream](
                self.cu_mags, self.cu_errs,
                cu_sed, cu_sederr,
                self.cu_mhat)
            get_p_zyt[blockspergrid, threadsperblock, self.stream](
                self.cu_mags, self.cu_errs,
                self.cu_mhat,
                cu_w, cu_sed,
                cu_sederr, cu_p_zyt)
            l_value = cu_p_zyt.copy_to_host(stream=self.stream)
            l_value = np.sum(np.log10(l_value.sum(axis=1)))
            balance[blockspergrid_x, threadsperblock, self.stream](cu_p_zyt)
            fill_2and3[blockspergrid, threadsperblock, self.stream](
                cu_p_zyt, p2, p3,
                cu_sederr, self.cu_errs)
            p2x = p2.copy_to_host(stream=self.stream)
            sum1x = np.nansum(p2x, axis=0)
            sum1x[~(sum1x > 0)] = np.inf
            update_params[blockspergrid, threadsperblock, self.stream](
                p2, p3,
                self.cu_mags, self.cu_mhat,
                cu_sed, cu_tmu, cu_tsigma)
            p_zyt = cu_p_zyt.copy_to_host(stream=self.stream)
            tw = np.nanmean(p_zyt, axis=0)
            tw = tw / tw.sum()
            tmu = cu_tmu.copy_to_host(stream=self.stream)
            tmu = tmu / sum1x
            tmu -= np.nanmean(tmu, axis=1)[:, np.newaxis]
            tsigma = cu_tsigma.copy_to_host(stream=self.stream)
            tsigma = np.sqrt(tsigma / sum1x)
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
        mstar = None
        return self.params, iterations, l_value, mstar, output

    def get_mstar(self, p_zyt):
        best_model = np.argmax(p_zyt, axis=1)
        mstar = np.empty((self.sed_count, self.bands, 3))
        for i in range(self.sed_count):
            mstar_model_group = self.mags[best_model == i]
            for band in range(self.bands):
                mstar_band_group = mstar_model_group[:, band]
                if np.all(np.isnan(mstar_band_group)):
                    mstar[i, band] = [-1, 0, 0]
                    continue
                y, x = np.histogram(mstar_band_group, bins=30,
                                    range=(np.nanmin(mstar_band_group),
                                           np.nanmax(mstar_band_group)))
                x = 0.5 * (x[1:] + x[:-1])
                if np.argmax(y) > 2:
                    try:
                        out = curve_fit(flat_schechter,
                                        x[:np.argmax(y)-1],
                                        y[:np.argmax(y)-1],
                                        p0=(x[np.argmax(y)], np.max(y)),
                                        bounds=((x[0] - 2, 0),
                                                (30, np.inf)))

                        mstar[i, band, 0] = -1
                        mstar[i, band, 1] = out[0][0]
                    except RuntimeError:
                        mstar[i, band] = [-1, x[np.argmax(y)-1], 0]
                else:
                    mstar[i, band] = [-1, x[np.argmax(y)], 0]
                norm = quad(lambda x: unnormed_schechter(x,
                                                         mstar[i, band, 0],
                                                         mstar[i, band, 1], 1),
                            10, np.nanmax(self.mags[:, band]))[0]
                mstar[i, band, 2] = 1./norm
        return mstar
