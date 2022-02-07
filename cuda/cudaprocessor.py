import os
import pandas as pd
import numpy as np
from .cudafitter import CudaFitter
from semiphore_public.utils.filters import filter_data
from semiphore_public.utils.params import BANDS, LIMITS, EXTINCTIONS


class CudaProcessor():

    MAX_ITERATIONS = 300

    z = np.arange(0.02, 1.01, 0.02)

    def __init__(self, names, n_seds=3):
        self.names = names
        self.n_seds = n_seds
        self.columns = []
        self.ecolumns = []
        for t in self.names:
            self.columns += ['%s_%s' % (t.lower(), b.lower())
                             for b in BANDS[t]]
            self.ecolumns += ['e_%s_%s' % (t.lower(), b.lower())
                              for b in BANDS[t]]

    def load_data(self, filename=None, min_ok_mags=3):
        if filename is None:
            fname = '../inputs/%s.parquet' % ('_'.join(self.names))
        else:
            fname = filename
        if os.path.exists(fname):
            if fname.endswith('csv'):
                self.data = pd.read_csv(fname)
            else:
                self.data = pd.read_parquet(fname)
        else:
            RuntimeError("No input file provided")
        self.zs = self.data['z']
        self.mags = self.data[self.columns]
        self.errs = self.data[self.ecolumns]
        for name in self.names:
            for column, limit, extinction in zip(BANDS[name], LIMITS[name],
                                                 EXTINCTIONS[name]):
                column = '%s_%s' % (name.lower(), column.lower())
                self.mags[column][self.mags[column] > limit] = np.nan
                if 'extinction' in self.data.columns:
                    self.mags[column] -= self.data['extinction'] * extinction
        max_mag = len(self.columns) - min_ok_mags + 1
        mask = np.isnan(np.array(self.mags)).sum(axis=1) < max_mag
        self.mags = np.array(self.mags[mask])
        self.errs = np.array(self.errs[mask])
        self.zs = np.array(self.zs[mask])
        self.mags[np.where(np.isnan(self.errs))] = np.nan
        self.errs[np.where(np.isnan(self.mags))] = np.nan
        mask = np.isnan(self.mags).sum(axis=1) < max_mag
        self.mags = np.array(self.mags[mask])
        self.errs = np.array(self.errs[mask])
        self.zs = np.array(self.zs[mask])
        self.ind = np.digitize(self.zs, self.z - 0.01) - 1
        self.counts = np.unique(self.ind, return_counts=True)

    def iterate_data(self, size):
        i0 = 0
        count = 0
        i_weighted = 0
        i = 0
        while i <= self.counts[0].max():
            if i in self.counts[0]:
                cpos = self.counts[0].searchsorted(i)
                #print('...%s : %s' % (i, self.counts[1][cpos]))
                count += self.counts[1][cpos]
                i_weighted += i * self.counts[1][cpos]
                if count > size:
                    yield i_weighted / count, *self.get_data_for_zs(
                        np.arange(i0, i + 1, dtype=int))
                    i0 = i + 1
                    count = 0
                    i_weighted = 0
            i += 1
        if count > 0:
            while count < size:
                i0 -= 1
                if i0 in self.counts[0]:
                    #print('...%s : %s' % (i0, self.counts[1][cpos]))
                    cpos = self.counts[0].searchsorted(i0)
                    count += self.counts[1][cpos]
                    i_weighted += i0 * self.counts[1][cpos]
            yield i_weighted / count, *self.get_data_for_zs(
                np.arange(i0, i + 1, dtype=int))

    def get_data_for_zs(self, ii_arr):
        mask = np.in1d(self.ind, ii_arr)
        mags = self.mags[mask]
        errs = self.errs[mask]
        #print('='*20, ii_arr)
        #print('Objects %s' % mags.shape[0], end='->')
        if len(mags) > 256**2 - 1:
            choice = np.arange(256 ** 2 - 1, dtype=int)
            mags = mags[choice]
            errs = errs[choice]
        elif len(mags) < 100:
            return mags, errs
        # TODO: remove 2nd parameter:
        mask = filter_data(mags, mags.shape[1])
        mags = mags[mask]
        errs = errs[mask]
        #print(mags.shape[0])
        return mags, errs

    def get_data_for_z(self, ii):
        return self.get_data_for_zs([ii])

    def run_on_data(self, mags, errs, n_seds=None, full_output=False,
                    custom_params=None):
        if len(mags) < 100:
            return None
        if n_seds is None:
            n_seds = self.n_seds
        fitter = CudaFitter(mags, errs, n_seds)
        if custom_params is None:
            result = fitter.fit(self.MAX_ITERATIONS * n_seds)
        else:
            fitter.params = custom_params
            result = fitter.fit(self.MAX_ITERATIONS * n_seds,
                                reinit_params=False)
        if not full_output:
            result = result[:4]
        return result, len(mags), fitter

    def run_single_z(self, ii, n_seds=None, full_output=False,
                     custom_params=None):
        mags, errs = self.get_data_for_z(ii)
        return self.run_on_data(mags, errs, n_seds, full_output, custom_params)
