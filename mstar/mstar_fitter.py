#!/usr/bin/env python3
import sys
import pylab as plt
import numpy as np
from scipy.optimize import curve_fit
from semiphore_public.cuda.cudaprocessor import CudaProcessor
from semiphore_public.utils.params import LIMITS
from semiphore_public.mstar.mstar_func import fun, fun2
import joblib

np.set_printoptions(linewidth=200)


def get_all_fits(processor, band):
    m = processor.mags[:, band]
    me = processor.errs[:, band]
    mask = np.isnan(m) | np.isnan(me) | (me > 0.5) | (me < 1e-6)
    z = processor.zs[~mask]
    m = m[~mask]
    m_space = np.linspace(np.quantile(m, 2e3 / len(m)), lim[band], 20)
    m_dig = np.digitize(m, m_space)
    m_out_space = []
    out = []
    x = []
    y = []
    for i in range(1, 20):
        z_i = z[m_dig == i]
        z_i = z_i[z_i < 1]
        hh = np.histogram(z_i, bins=100)
        z_pos = 0.5 * (hh[1][1:] + hh[1][:-1])
        try:
            b0 = z_pos[np.argmax(hh[0])]
            a0 = 3
            h0 = hh[0].min()
            fit = curve_fit(fun, z_pos, hh[0],
                            p0=(a0, b0, (hh[0] - h0).sum(), h0),
                            sigma=np.sqrt(hh[0]+1))
            if fit[0][1] > 0.1 and not np.any(np.isinf(np.diag(fit[1]))):
                try:
                    fit1 = curve_fit(fun2, z_pos, hh[0],
                                     p0=(*fit[0], 0),
                                     sigma=np.sqrt(hh[0]+1))
                    print('+', i, len(z_i), fit1[0], np.sqrt(np.diag(fit1[1])))
                except RuntimeError:
                    fit1 = (list(fit[0]) + [0.], fit[1])
                    print('-', i, len(z_i), fit1[0], np.sqrt(np.diag(fit1[1])))
                fit_par = fit1[0]
            else:
                print(i, len(z_i), fit[0], np.sqrt(np.diag(fit[1])))
                fit_par = list(fit[0]) + [0.]
            if np.any(np.isinf(np.diag(fit[1]))) and i > 0:
                fit_par = out[-1]
            out.append(fit_par)
            m_out_space.append(0.5 * (m_space[i - 1] + m_space[i]))

        except RuntimeError:
            import ipdb; ipdb.set_trace()
            pass
        x.append(z_pos)
        y.append(hh[0])
    out = np.array(out)
    print('='*20)
    return np.array(m_out_space), out, x, y


if __name__ == '__main__':
    survey = sys.argv[1]
    processor = CudaProcessor([survey], 4)
    processor.load_data(min_ok_mags=1)
    lim = LIMITS[survey]
    store = {}
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()

    for i in range(len(processor.columns)):
        m_space, out, x, y = get_all_fits(processor, i)
        store[processor.columns[i]] = (m_space, out, x, y)
        X = m_space
        Y = np.abs(out[:, 0])
        l = ax[0].plot(X, Y, label=processor.columns[i])

        Y = np.abs(out[:, 1])
        ax[1].plot(X, Y, color=l[0].get_color())

        Y = np.abs(out[:, 3] / out[:, 2])
        ax[2].plot(X, Y, color=l[0].get_color())

        Y = np.abs(out[:, 4])
        ax[3].plot(X, Y, color=l[0].get_color())
    ax[0].legend()
    ax[2].set_yscale('log')
    joblib.dump(store, '../calibrations/mstar/%s.joblib' % survey)

    plt.savefig('%s.png' % survey)
