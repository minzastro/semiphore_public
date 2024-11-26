import joblib
import numpy as np
from scipy.integrate import quad

from .mstar_func import fun2


class MStarManager():
    def __init__(self):
        # ranges covered by m values
        self.m_spaces = {}
        # Parameter values
        self.params = {}

    def load(self, catalog):
        """Load mstar calibration data for a given catalog.

        Args:
            catalog (str): Catalog key.
        """
        data = joblib.load('../calibrations/mstar/%s.joblib' % catalog)
        for band in data.keys():
            m_space, out, _, _ = data[band]
            self.m_spaces[band] = m_space
            norm = np.empty(len(out))
            for i in range(len(norm)):
                norm[i] = quad(lambda x: fun2(x, *out[i]), 0, 1)[0]
            out[:, 2:] = out[:, 2:] / norm[:, np.newaxis]
            self.params[band] = out

    def get_direct(self, mags, z, bands):
        """Calculate mstar value for given magnitude values
        at a given redshift.

        Args:
            mags (float[:,:]): magnitude values
            z (float): redshift
            bands (str[]): band names

        Returns:
            float: mstar value
        """
        u = 0
        n = 0
        for i, m in enumerate(mags):
            if np.isnan(m):
                continue
            pos = self.m_spaces[bands[i]].searchsorted(m) - 1
            value = fun2(z, *self.params[bands[i]][pos])
            u += np.log(len(z)*value / np.nansum(value))
            n += 1
        return u / n

    def get_precomputed(self, zs):
        """Pre-compute prior values for future use.

        Args:
            zs (float[]): grid of redshifts

        Returns:
            float[][][]: prior value per band, magnitude bin and redshift
        """
        bands = list(self.m_spaces.keys())
        m_size = len(self.m_spaces[bands[0]])
        p = np.zeros((len(bands), m_size, len(zs))) * np.nan
        for b, band in enumerate(bands):
            for pos in range(m_size):
                p[b, pos] = fun2(zs, *self.params[band][pos])
        return p

    def get_single(self, mags, z, bands):
        """Calculate per-band mstar value for given magnitude values
        at a given redshift.

        Args:
            mags (float[:,:]): magnitude values
            z (float): redshift
            bands (str[]): band names

        Returns:
            float[]: mstar values
        """
        p = np.zeros((len(mags), len(z))) * np.nan
        for i, m in enumerate(mags):
            if np.isnan(m):
                continue
            pos = self.m_spaces[bands[i]].searchsorted(m) - 1
            value = fun2(z, *self.params[bands[i]][pos])
            p[i] = np.log(value / np.nansum(value))
        return p

    def get_params_as_array(self):
        """
        Returns params as a numpy array
        """
        params = np.zeros((len(self.params), 19, 5))
        for i, b in enumerate(self.params.keys()):
            params[i] = self.params[b]
        return params

    def get_m_spaces_as_array(self):
        """
        Returns m_spaces as a numpy array
        """
        m_spaces = np.zeros((len(self.m_spaces), 19))
        for i, b in enumerate(self.m_spaces.keys()):
            m_spaces[i] = self.m_spaces[b]
        return m_spaces
