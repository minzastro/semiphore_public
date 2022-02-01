import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def extrapolate(_x, _values, _x0, order=1):
    poly = np.polyfit(_x, _values, deg=order)
    return np.polyval(poly, _x0)


def reorder(array, ind):
    output = np.empty_like(array)
    for ii in range(len(array)):
        output[ii] = array[ii][ind[ii][::-1]]
    return output


def curve_processor(_x, _values, _x0, order=1, is_log=False,
                    window=7, bounded=False):
    input_y = _values.copy()
    result = np.empty(shape=(_x0.shape[0], *(_values.shape[1:])))
    if len(_values.shape) > 1:
        for sub in range(_values.shape[-1]):
            result[..., sub] = curve_processor(_x, _values[..., sub], _x0,
                                               order, is_log, window, bounded)
        return result
    else:
        fill_value = 0
        low = _x0.searchsorted(_x.min())
        high = _x0.searchsorted(_x.max())
        if is_log:
            input_y = np.log10(input_y)
            fill_value = -np.inf
        result = interp1d(_x, input_y, fill_value=fill_value,
                          bounds_error=False)(_x0)
        result[:low] = extrapolate(_x[:window], input_y[:window], _x0[:low],
                                   order=order)
        result[high:] = extrapolate(_x[-window:], input_y[-window:],
                                    _x0[high:], order=order)
        result = savgol_filter(result, window, 3)
        if is_log:
            result = np.power(10, result)
        if bounded:
            result[result < _values.min()] = _values.min()
            result[result > _values.max()] = _values.max()
        return result
