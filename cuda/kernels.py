import math
from numba import cuda

SQRT_TWOPI = 2.506628274631


@cuda.jit
def get_mhat(mag, err, sed, sed_err, output):
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
def get_p_zyt(m, err, mhat, w, mu, sigma, p_zyt_out):
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
                          (m[row, b] - mhat[row, col] - mu[col, b]) *
                          (m[row, b] - mhat[row, col] - mu[col, b])
                          / delta2) / (SQRT_TWOPI * math.sqrt(delta2))
    p_zyt_out[row, col] = p


@cuda.jit
def get_chy2(m, err, mhat, w, mu, sigma, chy2_out):
    row, col = cuda.grid(2)
    if col >= len(w) or row >= len(m):
        return
    p = 0
    for b in range(m.shape[1]):
        if not math.isnan(m[row, b]) and not math.isnan(err[row, b]):
            delta2 = err[row, b]**2 + sigma[col, b]**2
            p += (m[row, b] - mhat[row, col] - mu[col, b]) * \
                 (m[row, b] - mhat[row, col] - mu[col, b]) / delta2
    chy2_out[row, col] = p


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
def update_params(p2, p3, mags, mhat, mu, tmu, tsigma):
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
            tmu[component, band] += (mags[i, band] - mhat[i, component]) * \
                p2[i, component, band]
            tsigma[component, band] += \
                (mags[i, band] - mhat[i, component] -
                 mu[component, band])**2 * \
                p3[i, component, band]

