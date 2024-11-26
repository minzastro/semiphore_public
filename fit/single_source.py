#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:49:49 2020

@author: mints
"""
import argparse
import logging

import joblib
import numpy as np
import pandas as pd
import pylab as plt
from astropy.cosmology import Planck18
from semiphore_public.mstar.mstarmanager import MStarManager
from semiphore_public.utils import params
from semiphore_public.utils.params import BANDS
from semiphore_public.utils.params import EXTINCTIONS
from semiphore_public.utils.params import LIMITS

logging.basicConfig(
    format='%(asctime)s | %(module)s | %(levelname)s | %(message)s',
    level=logging.INFO)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (5, 3.5)
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.size'] = 10.


def get_mhat1(row, sed, err, sed_err):
    return np.nansum((row - sed) / (err + sed_err**2)) / \
        np.nansum(1 / (err + sed_err**2))


parser = argparse.ArgumentParser(description="""
Run photo-z estimation.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-c', '--catalogs', type=str, default=None, required=True,
                    help='Input catalogs to use (comma separated)')
parser.add_argument('--calibrate', type=str, default=None,
                    help='Calibration filename')
parser.add_argument('-i', '--input', type=str, default=None,
                    help='Input filename')
parser.add_argument('-id', type=str, default=None,
                    help='Input filename')
parser.add_argument('-o', '--output', type=str, default=None,
                    help='Output filename')
parser.add_argument('-n', '--nsed', type=int, default=1,
                    help='Number of SEDs to fit')
parser.add_argument('-V', '--verbose', action="store_true",
                    default=False,
                    help='Be verbose')

args = parser.parse_args()

names = args.catalogs.split(',')

if args.calibrate is None:
    cfilename = '../calibrations/seds/%s_%sseds.joblib' % ('_'.join(names), args.nsed)
else:
    cfilename = args.calibrate
logging.info("Loading calibration data...")
d = joblib.load(cfilename)

columns, ecolumns, limits = params.collect(names)

logging.info("Loading input data...")
name = args.input
if name.endswith('csv'):
    data = pd.read_csv(name)
else:
    data = pd.read_parquet(name)
z = d['z']
dm0 = Planck18.distmod(z).value

weights = d['weights']
seds = d['sed']
sed_errs = np.sqrt(d['err']**2 + 0e-2)

ID = args.id
v = np.zeros_like(weights)
vv = np.zeros_like(weights)
mags = data[columns].copy()
errs = data[ecolumns]

icol = 0
pos = np.where(np.asarray(data['id'], dtype=str) == ID)[0][0]
for name in names:
    for column, limit, extinction in zip(BANDS[name], LIMITS[name],
                                         EXTINCTIONS[name]):
        column = '%s_%s' % (name.lower(), column.lower())
        mask = mags[column] > limit
        mags.loc[mask, column] = np.nan
        if 'extinction' in data.columns:
            mags.loc[:, column] -= data['extinction'] * extinction
        icol += 1

row = np.array(mags)[pos]
erow = np.array(errs)[pos]**2  #+ 1e-8
magnames = []
row[erow > 0.25] = np.nan
erow[erow > 0.25] = np.nan
erow[row > limits] = np.nan
row[row > limits] = np.nan
row[np.isnan(erow)] = np.nan
erow[np.isnan(row)] = np.nan
print(row, erow)
mm = np.zeros_like(weights)
sed_w = np.log(np.prod(sed_errs, axis=2))


logging.info("Loading mstar data...")
msm = MStarManager()
for catalog in names:
    msm.load(catalog)
m_prior = msm.get_direct(row, z, columns)
m_prior_per_band = msm.get_single(row, z, columns)
#import ipdb; ipdb.set_trace()
dm = Planck18.distmod(z).value
logging.info("Finding p values...")

trace_L = np.empty((v.shape[0], v.shape[1], len(columns)))
for iz in range(len(z)):
    sed = seds[iz]
    sed_err = sed_errs[iz]
    for g in range(v.shape[1]):
        m0 = get_mhat1(row, sed[g], erow, sed_err[g])
        v[iz, g] = np.log(weights[iz, g]) - 0.5 * np.nansum(
            (row - sed[g] - m0)**2 / (sed_err[g]**2 + erow)
            + np.log(sed_err[g]**2 + erow))
        mm[iz, g] = m0
        trace_L[iz, g] = (row - sed[g] - m0)**2 / (sed_err[g]**2 + erow) + np.log(sed_err[g]**2 + erow)

np.save('%s_trace_L.npy' % ID, trace_L)
np.save('%s_trace_P.npy' % ID, v)
np.save('%s_mhat.npy' % ID, mm)
np.save('%s_mprior.npy' % ID, m_prior_per_band)
abs_mag = np.nanmin(row) - dm
p_with_prior = (v +
                m_prior[:, np.newaxis] +
                0.)

print(m_prior)
print('='*50)
print(v)
print(p_with_prior)

ztrue = np.searchsorted(z, data['z'][pos])
m0true = np.zeros(v.shape[1])
for g in range(v.shape[1]):
    m0true[g] = get_mhat1(row, seds[ztrue][g], erow, sed_errs[ztrue][g])
ddd = np.argmax(p_with_prior, axis=1)
p_best = np.choose(ddd, p_with_prior.T)
zest = np.argmax(p_best)
print(ddd, z[zest])
print(m_prior[zest])

logging.info("Plotting...")
m0est = mm[zest]
fig, ax = plt.subplots(1, 3)
ax[0].plot(z, np.log10(v.max() - v + 1), '--')
ax[0].plot(z, np.log10(p_with_prior.max() - p_with_prior + 1))
ax[1].plot(z, m_prior_per_band.T)
ax[0].axvline(data['z'][pos])
ax[0].set_xlabel(data['id'][pos])
ax[2].plot((seds[ztrue] + m0true[:, np.newaxis]).T, '--', label='True z')
ax[2].plot((seds[zest] + m0est[:, np.newaxis]).T, label='Estimated z')
ax[2].plot(row, '-', color='black', label='SED')
ax[2].legend()
x_row = np.arange(len(row))
ax[2].errorbar(x_row, row, yerr=np.sqrt(erow), color='black')
ax[2].set_xticks(x_row)
ax[2].set_xticklabels(magnames, rotation=45)
plt.show()
