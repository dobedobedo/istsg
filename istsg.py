"""
This is the modular Python implementation of the improved spatial-temporal Savitzky–Golay filter
Original code can be found here: https://github.com/wyWang365/iSTSG/blob/main/PYcode
Reference of this study:
Wang, W., Cao, R., Liu, L., Zhou, J., Shen, M., Zhu, X., Chen, J., 2025. 
    An Improved Spatiotemporal Savitzky–Golay (iSTSG) Method to Improve the Quality of Vegetation Index 
    Time-Series Data on the Google Earth Engine. 
    IEEE Transactions on Geoscience and Remote Sensing 63, 1–17. https://doi.org/10.1109/TGRS.2025.3528988
"""

import numpy as np
from scipy.signal import savgol_coeffs
import numba


@numba.njit
def _linear_least_squares_fit(x, y, w):
    # Ensure all arrays are of dtype float64
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    w = w.astype(np.float64)
    
    # Calculate the weighted least squares fit
    X = np.zeros((len(x), 2), dtype=np.float64)
    X[:, 0] = x
    X[:, 1] = 1
    beta = np.linalg.lstsq(X*w[:, None], y*w)[0]
    return beta


@numba.njit
def linear_interp(arr):
    arr = arr.copy()
    non_nan_indices = np.nonzero(~np.isnan(arr))[0]

    if len(non_nan_indices) == 0:
        return arr
    
    nan_indices = np.nonzero(np.isnan(arr))[0]
    
    first_non_nan_index = non_nan_indices[0]
    arr[:first_non_nan_index] = arr[first_non_nan_index]
    last_non_nan_index = non_nan_indices[-1]
    arr[last_non_nan_index + 1:] = arr[last_non_nan_index]
    arr[nan_indices] = np.interp(nan_indices, non_nan_indices, arr[non_nan_indices])
    return arr


@numba.njit
def check_sign(reg, data):
    sign_reg = np.sign(reg)
    sign_data = np.sign(data)
    mask = (sign_reg != sign_data) & (~np.isnan(reg)) & (~np.isnan(data))
    reg[mask] = np.nan
    return reg


@numba.njit
def get_window(data, row, col, win):
    rmin = max(row - win, 0)
    rmax = min(row + win + 1, data.shape[0])
    cmin = max(col - win, 0)
    cmax = min(col + win + 1, data.shape[1])
    return np.ascontiguousarray(data[rmin:rmax, cmin:cmax, :])


@numba.njit
def auto_filter(curve, coeffs):
    winsize = int(len(coeffs))
    halfwin = (winsize - 1) // 2
    extended = np.concatenate((curve[:halfwin], curve, curve[-halfwin:]))
    result = np.zeros_like(curve)
    for i in range(curve.size):
        window = extended[i:i + winsize]
        result[i] = np.sum(window * coeffs)
    return result


@numba.njit
def three_stage_filter(curve, coeffs):
    winsize = int(len(coeffs))
    size = int(len(curve))
    halfwin = int((winsize - 1) / 2)

    f1 = auto_filter(curve, coeffs)
    f2 = auto_filter(np.maximum(curve, f1), coeffs)
    f3 = auto_filter(np.maximum(curve, f2), coeffs)
    
    W1 = np.full(curve.shape, np.nan)
    contant = np.ones(curve.shape)
    for i in range(size):
        start_index = max(i - halfwin, 0)
        end_index = min(i + halfwin + 1, size)
        eachslice = curve[start_index:end_index]

        sliceNocenter = np.copy(eachslice)
        sliceNocenter[int(min(i, halfwin))] = np.nan

        center = eachslice[int(min(i, halfwin))]
        mean = np.nanmean(sliceNocenter)
        denom = max(np.abs(np.nanmax(eachslice) - mean), np.abs(mean - np.nanmin(eachslice)))
        if denom == 0:
            continue

        W1[i] = np.abs(center - mean) / denom

    W2 = contant - W1
    blend = W1 * np.maximum(curve, f1) + W2 * f3

    return blend


@numba.njit
def corrcoef_numba(center, others):
    n = others.shape[0]
    corrs = np.zeros(n)
    for i in range(n):
        valid = (~np.isnan(center)) & (~np.isnan(others[i])) & (center != 0) & (others[i] != 0)
        if np.sum(valid) < 2:
            corrs[i] = np.nan
        else:
            x, y = center[valid], others[i][valid]
            corrs[i] = np.corrcoef(x, y)[0, 1]
    return corrs


@numba.njit
def linear_fit_neighbors(array, target_index, Thalfwin):
    m, n, s = array.shape
    k = np.zeros(s)
    b = np.zeros(s)
    target_id = min(target_index, Thalfwin)
    target = array[:, :, target_id].flatten()

    for j in range(s):
        if j == target_id:
            k[j] = np.nan
            b[j] = np.nan
            continue
        sim = array[:, :, j].flatten()
        mask = ~np.isnan(target) & ~np.isnan(sim)
        if np.any(mask):
            valid_indices = np.nonzero(mask)[0]
            x = sim[valid_indices]
            y = target[valid_indices]
            weights = np.ones_like(x, dtype=np.float64)
            # k[j], b[j] = np.polyfit(x, y, 1)
            # Solve the linear regression using linear algebra using numba
            popt = _linear_least_squares_fit(x, y, weights)
            k[j], b[j] = popt
        else:
            k[j] = np.nan
            b[j] = np.nan
    return k, b


@numba.njit(parallel=True)
def process_pixel(data, COEFFS_TREND, lower_thred=-1, upper_thred=1, halfwin=5, Thalfwin=4, threshold_R=0.85, threshold_n=10):
    height, width, Tsize = data.shape
    output = np.full(data.shape, np.nan)

    for row in numba.prange(height):
        for col in numba.prange(width):
            data_win = get_window(data, row, col, halfwin)
            center = data_win[halfwin, halfwin, :]
            
            not_nan = np.sum(~np.isnan(data_win), axis=2)
            R = corrcoef_numba(center, data_win.reshape(-1, Tsize))
            # R = R.reshape(2 * halfwin + 1, 2 * halfwin + 1)
            R = R.reshape(data_win.shape[:2])
            mask = (not_nan >= threshold_n) & (R >= threshold_R)

            sim = np.where(mask[..., None], data_win, np.nan)
            sim[halfwin, halfwin, :] = center

            reg = np.zeros(Tsize)
            for t in numba.prange(Tsize):
                start, end = max(0, t - Thalfwin), min(Tsize, t + Thalfwin + 1)
                win = sim[:, :, start:end]
                tgt = win[halfwin, halfwin, :]
                win_nocenter = win.copy()
                win_nocenter[halfwin, halfwin, :] = np.nan
                k, b = linear_fit_neighbors(win_nocenter, t, Thalfwin)
                regression = tgt * k + b
                # # Cut off data out of normal range
                regression[(regression > upper_thred) | (regression < lower_thred)] = np.nan
                reg[t] = np.nanmedian(regression)

            data_pix = data[row, col, :]
            reg = check_sign(reg, data_pix)
            reg_interp = linear_interp(reg)

            # Replace nans in data with regression_interp, otherwise keep data
            # syn = np.where(np.isnan(data_pix), reg_interp, np.maximum(data_pix, reg_interp))
            syn = np.where(np.isnan(data_pix), reg_interp, np.fmax(data_pix, reg_interp))
            output[row, col, :] = three_stage_filter(linear_interp(syn), COEFFS_TREND)

    return output


def run_istsg(data, lower_thred=-1, upper_thred=1, halfwin=5, Thalfwin=4, threshold_R=0.85, threshold_n=10, sg_filterlength=7, sg_polyorder=4):
    """
    data: numpy array in the shape of (y, x, time)
    """
    COEFFS_TREND = savgol_coeffs(sg_filterlength, sg_polyorder)
    output = process_pixel(data, COEFFS_TREND, lower_thred, upper_thred, halfwin, Thalfwin, threshold_R, threshold_n)

    # Copy the edge of the original data to the output result
    output[:halfwin, :halfwin, :] = np.where(np.isnan(data[:halfwin, :halfwin, :]), 
                                             output[:halfwin, :halfwin, :], 
                                             data[:halfwin, :halfwin, :])
    output[-halfwin:, -halfwin:, :] = np.where(np.isnan(data[-halfwin:, -halfwin:, :]), 
                                               output[-halfwin:, -halfwin:, :], 
                                               data[-halfwin:, -halfwin:, :])
    output[..., :Thalfwin] = np.where(np.isnan(data[..., :Thalfwin]), 
                                       output[..., :Thalfwin], 
                                       data[..., :Thalfwin])
    output[..., -Thalfwin:] = np.where(np.isnan(data[..., -Thalfwin:]), 
                                       output[..., -Thalfwin:], 
                                       data[..., -Thalfwin:])
    
    return output