from glob import glob

import numpy as np
import pandas as pd
from astropy import table
from scipy.stats import binned_statistic


def get_outlier_fraction(tbl, suffix='', bins=20):
    diff = np.array(np.abs(tbl['z_est'] - tbl['z']) > 0.15 * (1 + tbl['z']),
                    dtype=float)
    stat = binned_statistic(tbl['z%s' % suffix], diff, statistic='mean',
                            bins=bins)
    return stat.statistic


def get_diagnostics(z1, z2):
    diff = np.array(z1 - z2) / (1 + np.array((z1)))
    outlier_mask = np.abs(diff) < 0.15  # * (1 + z1)
    med = np.median(diff)
    mad = np.median(np.abs(diff - med))
    return 100*np.array((np.mean(diff[outlier_mask]),
                         np.std(diff[outlier_mask]),
                         med, mad, 1-outlier_mask.mean()))


def run_for_table_old(name, min=None):
    t = table.Table.from_pandas(pd.read_csv(name))
    tmax = t['mags'].max()
    t = t[t['z_est'] > 0]
    if min is None:
        max_mag = 2
        while max_mag <= max(max_mag, tmax):
            t_ = t[t['mags'] <= max_mag]
            if len(t_) > 0.9 * len(t):
                break
            max_mag += 1
        diag_old = get_diagnostics(t_['z'], t_['z_est'])
        max_outlier_rate = diag_old[-1]
        used_fraction = len(t_)*100 / len(t)
        i = 2
        for i in range(max_mag, tmax + 1):
            t_ = t[t['mags'] <= i]
            x = t_['z']
            y = t_['z_est']
            if len(t_) == 0:
                break
            diag = get_diagnostics(x, y)
            print(name, i, '%.3f' % diag[-1], len(t_), i,
                  '%.3f' % max_outlier_rate)
            if diag[-1] > max_outlier_rate:
                break
            diag_old = diag
            used_fraction = len(t_)*100 / len(t)
    else:
        i = min + 1
        t_ = t[t['mags'] <= int(min)]
        diag_old = get_diagnostics(t_['z'], t_['z_est'])
        used_fraction = len(t_)*100 / len(t)
    return len(t_['z']), diag_old, i - 1, used_fraction


def run_for_table(name):
    if name.endswith('csv'):
        df = pd.read_csv(name)
    elif name.endswith('parquet'):
        df = pd.read_parquet(name, columns=['mags', 'z', 'z_est'])
    else:
        return [0, [0]*5]
    x = df['z']
    y = df['z_est']
    diag = get_diagnostics(x, y)
    return len(df['z']), diag


def name_to_caption(name):
    output = name.split('/')[-1].replace('.csv', '').replace('.parquet', '')
    if '-' in output:
        output_parts = output.split('-')[1:-1]
        output = ' '.join([s.replace('_', ' ').replace('+', '')
                           for s in output_parts])
    output = output.replace('  ', ' ').replace(' ', ', ')
    return output


def get_stats_for_file(name, **kwargs):
    output = table.Table(names=['Name', 'Mean', 'Std', 'Median',
                                'MAD', 'Outliers',
                                'Count'],
                         dtype=[str, float, float, float, float, float,
                                int])
    row = run_for_table(name, **kwargs)
    output.add_row([name_to_caption(name), *row[1], row[0]])
    return output


def get_stats_for_folder(folder, **kwargs):
    output = table.Table(names=['Name', 'Mean', 'Std', 'Median',
                                'MAD', 'Outliers',
                                'Count'],
                         dtype=[str, float, float, float, float, float,
                                int])

    names = glob('%s/*.csv' % folder) + glob('%s/*.parquet' % folder)
    names.sort()
    for f in names:
        row = run_for_table(f, **kwargs)
        output.add_row([name_to_caption(f), *row[1], row[0]])
    return output
