#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:56:03 2020

@author: mints
"""
import logging
import itertools
import joblib
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from pandas.core.common import SettingWithCopyWarning
from semiphore_public.cuda.cudaprocessor import CudaProcessor
from semiphore_public.utils import interpolate
warnings.filterwarnings('ignore', category=AstropyUserWarning, append=True)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning, append=True)


def distance(w1, w2, sed1, sed2, err1, err2):
    """Calculate distance between two SED templates

    Args:
        w1 (float): weight of the first SED
        w2 (float): weight of the second SED
        sed1 (float[]): magnitudes of the first SED
        sed2 (float[]): magnitudes of the second SED
        err1 (float[]): width of the first SED
        err2 (float[]): width of the second SED

    Returns:
        "Distance"
    """
    d = (w1 * (sed1 - sed2)**2 / (err1**2 + 1e-2),
         w2 * (sed1 - sed2)**2 / (err2**2 + 1e-2))
    return np.sum(np.sqrt(d))


def get_order(w1, w2, sed1, sed2, err1, err2):
    """Reorder SEDs. Here all parameters are arrays along the redshift.

    Args:
        w1 (float[]): weight of the first SED
        w2 (float[]): weight of the second SED
        sed1 (float[][]): magnitudes of the first SED
        sed2 (float[][]): magnitudes of the second SED
        err1 (float[][]): width of the first SED
        err2 (float[][]): width of the second SED


    Returns:
        [type]: [description]
    """
    nn = len(w1)
    d = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            d[i, j] = distance(w1[i], w2[j],
                               sed1[i], sed2[j],
                               err1[i], err2[j])
    smin = np.inf
    tOk = None
    for t in itertools.permutations(np.arange(nn, dtype=int), nn):
        s = 0
        for i in range(nn):
            s += d[i, t[i]]
        if s < smin:
            smin = s
            tOk = t
    return tOk


if __name__ == '__main__':
    import argparse
    import logging
    logger = logging.getLogger("FIT")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="""
    Perform a full CUDA-based SED-PhotoZ fit.
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input filename')
    parser.add_argument('-c', '--catalogs', type=str,
                        default=None, required=True,
                        help='Input catalogs to use (comma separated)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output filename')
    parser.add_argument('-n', '--nsed', type=int, default=1,
                        help='Number of SEDs to fit')
    parser.add_argument('-V', '--verbose', action="store_true",
                        default=False,
                        help='Be verbose')

    args = parser.parse_args()

    processor = CudaProcessor(args.catalogs.split(','), args.nsed)
    results = []
    sizes = []
    logger.info("Load data from %s", args.catalogs)
    processor.load_data(filename=args.input)
    z_len = len(processor.z)
    is_ok = []
    izs = []
    # Forward run
    for z0, mags, errs in processor.iterate_data(size=1000):
        logger.info("Forward run, redshift=%.2f", processor.z[int(z0)])
        if len(results) > 0:
            output = processor.run_on_data(mags, errs,
                                           custom_params=results[-1][0])
        else:
            output = processor.run_on_data(mags, errs)
        if output is not None:
            res, size, _ = output
            if res[1] >= processor.MAX_ITERATIONS * args.nsed:
                logger.warn(f'Iteration count exceeded for z nr {z0}')
                is_ok.append(False)
            else:
                is_ok.append(True)
            results.append(res)
            sizes.append(size)
            izs.append(z0)

    # Backward run:
    for ii in range(len(izs)-2, 0, -1):
        if not is_ok[ii + 1] or not is_ok[ii]:
            continue
        old_norm = results[ii][2] / sizes[ii]
        if results[ii + 1][2] / sizes[ii + 1] > old_norm:
            logger.info("Backward run, redshift=%.2f",
                        processor.z[int(izs[ii])])
            mags, errs = processor.get_data_for_zs(izs[ii])
            output = processor.run_on_data(mags, errs,
                                           custom_params=results[ii+1][0])
            if output is not None:
                res, size, _ = output
                if res[2] / size >= results[ii][2] / sizes[ii]:
                    logger.debug(f'...new l_norm={res[2] / size} is better')
                    results[ii] = res
                    sizes[ii] = size
                else:
                    logger.debug(f'...new l_norm={res[2] / size} is lower, rejecting')
    iz_min = int(np.ceil(np.min(izs)))
    iz_max = int(np.ceil(np.max(izs)))
    izs = processor.z[0] + np.array(izs) * 0.02

    sed_shape = (z_len, processor.n_seds, len(processor.columns))
    output = {'z': processor.z,
              'names': processor.names,
              'weights': np.zeros((z_len, processor.n_seds)),
              'sed': np.zeros(sed_shape),
              'err': np.zeros(sed_shape),
              'l_values': np.zeros(len(izs)),
              'iterations': np.zeros(len(izs)),
              'sizes': sizes,
              }
    w = np.array([results[ii][0][0] for ii in range(len(results))])
    sed = np.array([results[ii][0][1] for ii in range(len(results))])
    err = np.array([results[ii][0][2] for ii in range(len(results))])
    output['iterations'] = np.array([results[ii][1]
                                     for ii in range(len(results))])
    output['l_values'] = np.array([results[ii][2]
                                   for ii in range(len(results))])
    ind = np.argsort(w)

    logger.info("Reordering...")
    # Reordering
    output['weights00'] = w
    output['sed00'] = sed
    output['err00'] = err
    w_order = [w[0]]
    sed_order = [sed[0]]
    err_order = [err[0]]
    for i in range(0, len(w)-1):
        new_order = list(get_order(w_order[i], w[i+1],
                                   sed_order[i], sed[i+1],
                                   err_order[i], err[i+1]))
        w_order.append(w[i + 1][new_order])
        sed_order.append(sed[i + 1][new_order])
        err_order.append(err[i + 1][new_order])

    logger.info("Interpolating...")
    # Interpolation
    output['weights0'] = w_order
    output['sed0'] = sed_order
    output['err0'] = err_order
    output['weights'] = interpolate.curve_processor(izs, np.array(w_order),
                                                    processor.z, is_log=True)
    output['sed'] = interpolate.curve_processor(izs, np.array(sed_order),
                                                processor.z, is_log=False)
    output['err'] = interpolate.curve_processor(izs, np.array(err_order),
                                                processor.z,
                                                is_log=True, bounded=True)

    output['weights'] = output['weights'] / \
        output['weights'].sum(axis=1)[:, np.newaxis]
    output['z_base'] = izs
    output['input_file'] = args.input

    if args.output is None:
        names = '_'.join(processor.names)
        outname = f'../calibrations/seds/{names}_{processor.n_seds}seds.joblib'
    else:
        outname = args.output
    logger.info('Saving calibration to %s', outname)
    joblib.dump(output, outname)
    logger.info("Finished")
