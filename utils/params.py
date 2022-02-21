#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:16:00 2020

@author: mints
"""

A_K = 0.306


BANDS = {
    'AllWISE': ['W1mag', 'W2mag'],
    'ATLAS': ['%sap3' % s for s in list('UGRIZ')],
    'DES': ['mag_auto_%s' % s for s in list('grizy')],
    'KIDS': ['%smag' % s for s in list('ugri')],
    'LAS': ['p%smag' % s for s in ['y', 'j', 'h', 'k']],
    'LS8': ['dered_mag_%s' % s for s in list('grz')],
    'NSC': ['%smag' % s for s in list('ugrizy') + ['vr']],
    'PS1': ['%skmag' % s for s in list('grizy')],
    'SDSS': ['%smag' % s for s in list('ugriz')],
    'unWISE': ['W1mag', 'W2mag'],
    'VHS': ['%spmag' % s for s in list('YJH') + ['Ks']],
    'VIKING': ['%spmag' % s for s in list('ZYJH') + ['Ks']],
    #--- Simulated:
    'Happy': ['%smag' % s for s in list('ugriz')],
    'Teddy': ['%smag' % s for s in list('ugriz')],
}

EXTINCTIONS = {
    'AllWISE': [0, 0],
    'ATLAS': [0, 0, 0, 0, 0],
    'DES': [3.237, 2.176, 1.595, 1.217, 1.058],
    'KIDS': [4.239, 3.303, 2.285, 1.698],
    'LAS': [1.194957, 0.895497, 0.568943, 0.356779],
    'LS8': [0, 0, 0],
    'NSC': [5.10826797, 3.9170915, 2.73640523, 2.07503268, 1.51035948,
            1.30611111, 2.816129032],
    'PS1': [3.612, 2.691, 2.097, 1.604, 1.336],
    'SDSS': [0, 0, 0, 0, 0],
    'unWISE': [0, 0],
    'VHS': [1.213, 0.891, 0.564, 0.373],
    'VIKING': [1.578, 1.213, 0.891, 0.564, 0.373],
    # -- Simulated:
    'Happy': [0, 0, 0, 0, 0],
    'Teddy': [0, 0, 0, 0, 0],
}


LIMITS = {
    'AllWISE': [17.1, 15.7],
    'ATLAS': [21.78, 22.71, 22.17, 21.40, 20.23],
    'DES': [24.33, 24.08, 23.44, 22.69, 21.44],
    'KIDS': [24.3, 25.4, 25.2, 24.2],
    'LAS': [20.5, 20.0, 18.8, 18.4],
    'LS8': [24.5, 23.9, 22.9],
    'NSC': [22.6, 23.6, 23.2, 22.8, 22.3, 21.0, 23.3],
    'PS1': [23.3, 23.2, 23.1, 22.3, 21.3],
    'SDSS': [22.0, 22.2, 22.2, 21.3, 20.5],
    'unWISE': [17.93, 16.72],
    'VHS': [23., 21.6, 21.0, 20.2],
    'VIKING': [23.1, 22.3, 22.1, 21.5, 21.2],
    # -- Simulated:
    #'Happy': [22.0, 22.2, 22.2, 21.3, 20.5],
    'Happy': [99, 99, 99, 99, 99],
    'Teddy': [99, 99, 99, 99, 99],
}

def collect(names):
    columns = []
    ecolumns = []
    limits = []
    for t in names:
        columns += ['%s_%s' % (t.lower(), b.lower()) for b in BANDS[t]]
        ecolumns += ['e_%s_%s' % (t.lower(), b.lower()) for b in BANDS[t]]
        limits.extend(LIMITS[t])
    return columns, ecolumns, limits
