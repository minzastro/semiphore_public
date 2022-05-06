#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:05:50 2020

@author: mints
"""
import time
import joblib
import numpy as np
from semiphore_public.utils.params import BANDS, LIMITS, EXTINCTIONS
from semiphore_public.cuda import gpu
import pandas as pd
from numba import cuda
import math
import warnings
import argparse
from semiphore_public.mstar.mstarmanager import MStarManager
import logging


# create logger with 'spam_application'
logger = logging.getLogger('semiphore_run_estimate')
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s | %(module)s | %(levelname)s | %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)


MIN_WIDTH = 0.02


@cuda.jit
def get_mhat(mag, err, sed, sed_err, output):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    z = ty * cuda.blockDim.x + tx
    pos = by * cuda.gridDim.x + bx
    if pos >= mag.shape[0] or z >= sed.shape[0]:
        return
    for comp in range(sed.shape[1]):
        part1 = 0
        part2 = 0
        for i in range(sed.shape[2]):
            if not math.isnan(mag[pos, i]) and not math.isnan(err[pos, i]):
                denom = 1. / (err[pos, i]**2 + sed_err[z, comp, i]**2)
                part1 += (mag[pos, i] - sed[z, comp, i]) * denom
                part2 += denom
        output[pos, z, comp] = part1 / part2


@cuda.jit
def get_p_zyt(mag, err, mhat, w, sed, sed_err, output):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    blockPos = ty * cuda.blockDim.x + tx
    z = blockPos // sed.shape[1]
    comp = blockPos % sed.shape[1]
    pos = by * cuda.gridDim.x + bx
    if pos >= mag.shape[0] or z >= sed.shape[0] or comp >= sed.shape[1]:
        return
    p = math.log(w[z, comp])
    for b in range(sed.shape[2]):
        if not math.isnan(mag[pos, b]) and not math.isnan(err[pos, b]):
            delta2 = err[pos, b]**2 + sed_err[z, comp, b]**2
            p += -0.5 * \
                (mag[pos, b] - mhat[pos, z, comp] - sed[z, comp, b]) * \
                (mag[pos, b] - mhat[pos, z, comp] - sed[z, comp, b]) \
                / delta2 - 0.5 * math.log(delta2)
    output[pos, z, comp] = p


@cuda.jit
def cuda_prior_direct(mag, sed, mhat, m_spaces, m_direct, prior):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    i_z = ty * cuda.blockDim.x + tx
    pos = by * cuda.gridDim.x + bx
    if pos >= mag.shape[0] or i_z >= sed.shape[0]:
        return
    band = sed.shape[2] // 2
    for ised in range(sed.shape[1]):
        if math.isnan(mag[pos, band]):
            band_mag = mhat[pos, i_z, ised] + sed[i_z, ised, band]
        else:
            band_mag = mag[pos, band]
        mpos = int(
            (band_mag - m_spaces[band][0]) * (len(m_spaces[band]) - 1)
            / (m_spaces[band][-1] - m_spaces[band][0]))
        if mpos < 0:
            mpos = 0
        elif mpos >= m_spaces.shape[1]:
            mpos = m_spaces.shape[1] - 1
        prior[pos, ised, i_z] = math.log(m_direct[band, mpos, i_z])


@cuda.jit
def find_photo_z(p, prior, p_values, z, z_est, sed_est, p_est, p_est_noprior):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos >= len(p):
        return
    pp = p[pos]
    best_sed_index = -1
    best_p_index = -1
    best_p_value = -np.inf
    for i in range(len(pp)):
        sed_value = -np.inf
        sed_index = -1
        for j in range(pp.shape[1]):
            if pp[i, j] > sed_value:
                sed_index = j
                sed_value = pp[i, j] + prior[pos, j, i]
        p_values[pos, i] = sed_value
        if sed_value >= best_p_value:
            best_p_value = sed_value
            best_p_index = i
            best_sed_index = sed_index
    # Smoothing:
    for i in range(len(pp) - 2):
        p_values[pos][i] = (p_values[pos, i] + p_values[pos, i + 1]) / 2
    for i in range(len(pp) - 2):
        p_values[pos][-1-i] = (p_values[pos, -1-i] + p_values[pos, -2 - i]) / 2
    if best_p_index == 0:
        w = p_values[pos, :3]
        z0 = z[1]
    elif best_p_index == (p.shape[1] - 1):
        w = p_values[pos, -3:]
        z0 = z[-2]
    else:
        w = p_values[pos, best_p_index-1:best_p_index+2]
        z0 = z[best_p_index]
    est = z0 - 0.5 * (z[1] - z[0]) * (w[2] - w[0]) / (w[0] + w[2] - 2 * w[1])
    if est > 1 or est < 0 or math.isnan(est):
        est = z[best_p_index]
    z_est[pos] = est
    sed_est[pos] = best_sed_index
    p_est[pos] = best_p_value
    p_est_noprior[pos] = best_p_value - prior[pos, best_sed_index, best_p_index]


parser = argparse.ArgumentParser(description="""
Run photo-z estimation.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-c', '--catalogs', type=str, default=None, required=True,
                    help='Input catalogs to use (comma separated)')
parser.add_argument('--calibrate', type=str, default=None,
                    help='Calibration filename')
parser.add_argument('-i', '--input', type=str, default=None,
                    help='Input filename')
parser.add_argument('-o', '--output', type=str, default=None,
                    help='Output filename')
parser.add_argument('-n', '--nsed', type=int, default=1,
                    help='Number of SEDs to fit')
parser.add_argument('-V', '--verbose', action="store_true",
                    default=False,
                    help='Be verbose')
parser.add_argument('--quick', action="store_true",
                    default=False,
                    help='Quick run - only 1 subset')
parser.add_argument('--csv', action="store_true",
                    default=False,
                    help='Use CVS output')

args = parser.parse_args()
time_start_all = time.process_time()

logger.info("Starting Semiphore")
columns = []
ecolumns = []
names = args.catalogs.split(',')
nseds = args.nsed
if args.calibrate is None:
    cfilename = '../calibrations/seds/%s_%sseds.joblib' % ('_'.join(names),
                                                           nseds)
else:
    cfilename = args.calibrate

logger.info("Reading calibration data")
d = joblib.load(cfilename)
for t in names:
    columns += ['%s_%s' % (t.lower(), b.lower()) for b in BANDS[t]]
    ecolumns += ['e_%s_%s' % (t.lower(), b.lower()) for b in BANDS[t]]

z = d['z']

msm = MStarManager()
for catalog in names:
    msm.load(catalog)

logger.info("Reading input catalogue")

name = args.input
if name is None:
    name = '../inputs/%s.parquet' % ('_'.join(names))
    data = pd.read_parquet(name)
elif name.endswith('csv'):
    data = pd.read_csv(name)
else:
    data = pd.read_parquet(name)

logger.info("Read %s data points from the catalogue %s" % (len(data), name))

ids = data['id']
rowids = np.arange(len(data))
zs = data['z']
mags = data[columns].copy()
errs = data[ecolumns].copy()
icol = 0
for name in names:
    for column, limit, extinction in zip(BANDS[name], LIMITS[name],
                                         EXTINCTIONS[name]):
        column = '%s_%s' % (name.lower(), column.lower())
        # Mask below-limit measurements, large and zero uncertainties.
        mask = (mags[column] > limit) | \
               (errs[f'e_{column}'] > 0.5) | \
               (errs[f'e_{column}'] < 1e-6)
        mags.loc[mask, column] = np.nan
        if 'extinction' in data.columns:
            mags[column] -= data['extinction'] * extinction
        icol += 1

mask = mags.count(axis='columns') >= 3
all_mags = np.array(mags[mask])
all_errs = np.array(errs[mask])
zs = np.array(zs[mask])
ids = np.array(ids[mask])
rowids = np.array(rowids[mask])
results = []

logger.info("%s data points in the catalogue after filtering" % len(all_mags))

weights = d['weights']
seds = d['sed']
sed_errs = d['err']
cu_w = cuda.to_device(weights)
cu_sed = cuda.to_device(seds)
cu_sederr = cuda.to_device(sed_errs)

sed_count = seds.shape[1]
mag_count = seds.shape[2]
redshift_count = seds.shape[0]
# Calculate batch and grid sizes depending on GPU memory
batch_size, grid_size, block = gpu.get_photoz_gpu_dims(sed_count,
                                                       mag_count,
                                                       redshift_count)
logger.info("GPU configuration: batch size = %s / block size = %s "
            "/ grid size = %s", batch_size, block, grid_size)
if len(all_mags) > batch_size and args.quick:
    # For a quick run we use only 1 randomly composed batch.
    choice = np.random.choice(np.arange(len(all_mags)), batch_size,
                              replace=False)
    zs = zs[choice]
    ids = ids[choice]
    rowids = rowids[choice]
    all_mags = all_mags[choice]
    all_errs = all_errs[choice]

logger.info("%s data points in the catalogue selected for processing" %
            len(all_mags))

time_start_processing = time.process_time()
total_batches = int(np.ceil(len(all_mags) / batch_size))
df_result = None

cu_m_star_m_spaces = cuda.to_device(msm.get_m_spaces_as_array())
cu_m_star_pre = cuda.to_device(msm.get_precomputed(z))
cu_p_values = cuda.device_array((batch_size, redshift_count))
cu_mhat = cuda.device_array((batch_size, redshift_count, sed_count))
cu_p = cuda.device_array((batch_size, redshift_count, sed_count))
cu_z = cuda.to_device(z)
cu_m_prior = cuda.device_array((batch_size, sed_count, redshift_count))
cu_z_estimate = cuda.device_array(batch_size)
cu_p_est = cuda.device_array(batch_size)
cu_p_est_noprior = cuda.device_array(batch_size)
cu_sed_estimate = cuda.device_array(batch_size, dtype=int)
# Configure the blocks
threadsperblock = block
blockspergrid_x = grid_size
blockspergrid_y = grid_size
blockspergrid = (blockspergrid_x, blockspergrid_y)

for i in range(total_batches):
    if total_batches > 1:
        logger.info("Running batch %s of %s" % (i + 1, total_batches))
    batch_mags = all_mags[i*batch_size:(i+1)*batch_size]
    batch_errs = all_errs[i*batch_size:(i+1)*batch_size]
    logger.info("Sending data to GPU")
    cu_mags = cuda.to_device(np.ascontiguousarray(batch_mags))
    cu_errs = cuda.to_device(np.ascontiguousarray(batch_errs))

    cuda.synchronize()

    logger.info("Running mhat")
    get_mhat[blockspergrid, threadsperblock](cu_mags, cu_errs,
                                             cu_sed, cu_sederr,
                                             cu_mhat)

    cuda.synchronize()
    logger.info("Running p_zyt")
    get_p_zyt[blockspergrid, threadsperblock](cu_mags, cu_errs, cu_mhat,
                                              cu_w, cu_sed, cu_sederr,
                                              cu_p)
    cuda.synchronize()
    #gpu.meminfo(logger)
    output = []
    warnings.simplefilter('ignore', np.RankWarning)

    logger.info("Calculating priors")
    cuda_prior_direct[blockspergrid, threadsperblock](cu_mags, cu_sed, cu_mhat,
                                                      cu_m_star_m_spaces,
                                                      cu_m_star_pre,
                                                      cu_m_prior)
    cuda.synchronize()

    logger.info("Finding photo zs")  # Why is it slow?
    find_photo_z[blockspergrid_x, threadsperblock](cu_p, cu_m_prior,
                                                   cu_p_values, cu_z,
                                                   cu_z_estimate,
                                                   cu_sed_estimate,
                                                   cu_p_est, cu_p_est_noprior)
    cuda.synchronize()

    current_size = len(ids[i*batch_size:(i+1)*batch_size])
    df = pd.DataFrame.from_dict({
        'id': ids[i*batch_size:(i+1)*batch_size],
        'rowid': rowids[i*batch_size:(i+1)*batch_size],
        'z': zs[i*batch_size:(i+1)*batch_size],
        'z_est': cu_z_estimate.copy_to_host()[:current_size],
        'model': cu_sed_estimate.copy_to_host()[:current_size],
        'p': cu_p_est.copy_to_host()[:current_size],
        'p0': cu_p_est_noprior.copy_to_host()[:current_size],
        'mags': np.isnan(batch_mags).sum(axis=1)
    })
    if df_result is None:
        df_result = df
    else:
        df_result = df_result.append(df, ignore_index=True)
time_processing = time.process_time() - time_start_processing
logger.info(
    "Processing runtime %.2f second, performance %.1f objects per second" % (
        time_processing, len(all_mags) / time_processing))
logger.info("Saving result")
if args.output is None:
    if args.csv:
        ext = 'csv'
    else:
        ext = 'parquet'
    outname = 'result-%s-%s.%s' % ('_'.join(names), nseds, ext)
else:
    outname = args.output

if args.csv:
    df_result.to_csv(outname)
else:
    df_result.to_parquet(outname)
time_all = time.process_time() - time_start_all
logger.info(
    "Complete runtime %.2f second, performance %.1f objects per second" % (
        time_all, len(all_mags) / time_all))
