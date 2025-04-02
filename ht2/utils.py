import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import multiprocessing as mp
from joblib import Parallel, delayed
from typing import Optional, Union
import torch
import torch.nn as nn

#import jax.numpy as np
"""Author(s): Ike H Zhang

References: HiTIPS"""

def bias_corr():
    """
    extract nucleus, remove nucleolus, fit brightness to bessel function with offset
    divide by bessel function
    """
    pass

def exp_decay(t, lambd, a, b):
    "lambd for exponential factor, a for scaling, b for offset"
    return a * np.exp(-lambd * t) + b

def biexp_decay(t, lambd1, a1, lambd2, a2, b):
    return a1 * np.exp(-lambd1 * t) + a2 * np.exp(-lambd2 * t) + b

def fit_exp(y, t, lambd0=0.000128):
    popt, pcov = curve_fit(exp_decay, t, y, p0=[lambd0, np.max(y) - np.min(y), np.min(y)], bounds=([0, 0.1 * (np.max(y) - np.min(y)), 0], [np.inf, 2*np.max(y), np.inf]), method='trf')
    return popt, exp_decay(t, popt[0], popt[1], popt[2])

def fit_biexp(y, t):
    popt0, _ = curve_fit(exp_decay, t, y)
    popt, pcov = curve_fit(biexp_decay, t, y, p0=[popt0[0], popt0[1], 0, 0, popt0[2]], bounds=([0, -np.inf, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]), method="trf")
    return popt, biexp_decay(t, popt[0], popt[1], popt[2], popt[3], popt[4])

def validate_dims(dim_order: str):
    if len(dim_order) > 5:
        raise ValueError("dimensions should be a combination of (T)ime, Z, (C)hannel, Y, X")
    for letter in dim_order:
        if letter not in "TZCYX":
            raise ValueError("dimensions should be a combination of (T)ime, Z, (C)hannel, Y, X")
    return True

def get_transpose_ind(in_dim_order: str, out_dim_order:str="TZCYX"):
    validate_dims(in_dim_order)
    validate_dims(out_dim_order)
    if len(in_dim_order) != len(out_dim_order):
        raise ValueError("dimension lengths do not match")
    dim_dict = {letter : i for i, letter in enumerate(out_dim_order)}
    return np.argsort(np.array([dim_dict[letter] for letter in in_dim_order]))

def get_dim_ind(in_dim: str, dim_order:str="TZCYX"):
    validate_dims(dim_order)
    validate_dims(in_dim)
    dim_dict = {letter : i for i, letter in enumerate(dim_order)}
    return np.argsort(np.array([dim_dict[letter] for letter in in_dim]))

def reorder_img(img, in_dim_order, out_dim_order):
    if (len(img.shape) != len(in_dim_order)):
        raise ValueError("image dimensions and provided dimensions do not match")
    transpose_ind = get_transpose_ind(in_dim_order, out_dim_order)
    tp_img = np.transpose(img, transpose_ind)
    return tp_img

def plot_hist(src, depth=12):
    bins = np.arange(0, 2 ** depth + 1, step=1)
    #pmf = np.bincount(src.flatten(), minlength=2 ** depth)
    pmf, bins = np.histogram(src.flatten(), bins=2**depth, range=(0, 2 ** depth))
    pmf = pmf
    #plt.bar(bins[:-1], np.log(pmf + 1), )
    plt.stairs(np.log(pmf + 1)[np.min(np.nonzero(pmf)[0]):np.max(np.nonzero(pmf)[0])], bins[np.min(np.nonzero(pmf)[0]):np.max(np.nonzero(pmf)[0]) + 1], fill=True)
    plt.title("Intensity log distribution")
    plt.xlabel("Intensity")
    plt.ylabel("log(count + 1)")
    plt.show()

def get_cdf(img:np.ndarray, depth: int=12):
    pmf = np.bincount(img.flatten(), minlength=2 ** depth)
    cdf = np.cumsum(pmf) / img.size
    return pmf, cdf

def histogram_eq(src:np.ndarray, depth: int=12):
    im = src.copy()
    bins = np.arange(0, 2 ** depth + 1, step=1)
    pmf, cdf = get_cdf(src, depth=depth)
    for i in range(2 ** depth):
        im[im == i] = (2 ** depth - 1) * cdf[i]
    return im.astype(np.int16)

def histogram_match(src:np.ndarray, ref:np.ndarray, depth:int =12): 
    # Pretty implementation, gets rid of wasted loops by selecting the points that matter
    dst = src.copy()
    pmf_r, cdf_r = get_cdf(ref, depth=depth)
    pmf_s, cdf_s = get_cdf(src, depth=depth)

    Is = np.nonzero(pmf_s)[0] # Saves important intensities 
    Ps = cdf_s[Is] # Probabilities 
    Js = np.searchsorted(cdf_r, Ps, side='left') # for every point in the cdf, find the point in the ref_cdf with the closest cumulative prob

    for I, J in zip(Is, Js): # This indexing is horrendous
        dst[src == I] = J # Map old intensities to new intensities
    return dst

def histogram_match_cdf(src:np.ndarray, cdf_r:np.ndarray, depth:int =12): 
    # match to preexisting cdf
    dst = np.zeros_like(src)
    pmf_s, cdf_s = get_cdf(src, depth=depth)

    Is = np.nonzero(pmf_s)[0] # Saves important intensities 
    Ps = cdf_s[Is] # Probabilities 
    Js = np.searchsorted(cdf_r, Ps, side='left') # for every point in the cdf, find the point in the ref_cdf with the closest cumulative prob

    for I, J in zip(Is, Js): # This indexing is horrendous
        dst[src == I] = J # Map old intensities to new intensities
    return dst

def _histogram_match_cdf(src:np.ndarray, cdf_r:np.ndarray, depth:int =12): 
    # match to preexisting cdf
    dst = np.zeros_like(src)
    pmf_s, cdf_s = get_cdf(src, depth=depth)

    Js = np.searchsorted(cdf_r, cdf_s, side='left') # for every point in the cdf, find the point in the ref_cdf with the closest cumulative prob
    for I, J in enumerate(Js): 
        dst[src == I] = J
    return dst

def pb_corr(src:np.ndarray, ref:np.ndarray, depth: int=12):
    dst = np.zeros_like(src, )
    pmf_r, cdf_r = get_cdf(ref, depth=depth) # get cdf of reference image

    for t in tqdm(range(src.shape[0])): # For every frame
        pmf_s, cdf_s = get_cdf(src[t, ...], depth=depth) 
        
        # All this BS is to "save time" on wasted computations where I is zero
        Is = np.nonzero(pmf_s)[0] # Saves important intensities 
        Ps = cdf_s[Is] # Probabilities 
        Js = np.searchsorted(cdf_r, Ps, side='left') # Find new intensities based on closest cumulative prob in each CDF
        
        for I, J in zip(Is, Js): # This indexing is horrendous
            dst[t, ...][src[t, ...] == I] = J # Map old intensities to new intensities
    return dst

def pb_corr_cdf(src:np.ndarray, cdf_r:np.ndarray, depth: int=12):
    dst = np.zeros_like(src, )
    print(dst.shape)
    for t in tqdm(range(src.shape[0])): # For every frame
        pmf_s, cdf_s = get_cdf(src[t, ...], depth=depth) 
        
        # All this BS is to "save time" on wasted computations where I is zero
        Is = np.nonzero(pmf_s)[0] # Saves important intensities 
        Ps = cdf_s[Is] # Probabilities 
        Js = np.searchsorted(cdf_r, Ps, side='left') # Find new intensities based on closest cumulative prob in each CDF
        
        for I, J in zip(Is, Js): # This indexing is horrendous
            dst[t, ...][src[t, ...] == I] = J # Map old intensities to new intensities
    return dst

def _pb_corr(src:np.ndarray, ref:np.ndarray, depth: int=12): 
    # A lot slower, a lot cleaner/maybe easier to understand, we'll call it a "reference impl"
    dst = np.zeros_like(src, )

    pmf_r, cdf_r = get_cdf(ref, depth=depth)
    for t in tqdm(range(src.shape[0])):
        pmf_s, cdf_s = get_cdf(src[t, ...], depth=depth)

        Js = np.searchsorted(cdf_r, cdf_s, side='left') 
        for I, J in enumerate(Js): 
            dst[t, ...][src[t, ...] == I] = J
    return dst

def pb_corr_parallel2(src:np.ndarray, ref:np.ndarray, depth:int=12, n_threads:int=8):
    pmf_r, cdf_r = get_cdf(ref, depth=depth)
    len_t = src.shape[0]
    if n_threads > 1:
        args = list([(src[i, ...], cdf_r, depth) for i in range(len_t)]) 
    with mp.Pool(n_threads) as p:
        result = p.starmap(histogram_match_cdf, args, chunksize=len_t // n_threads)
    return np.stack(result)

def pb_corr_parallel(src:np.ndarray, ref:np.ndarray, depth:int=12, n_jobs:int=8):
    pmf_r, cdf_r = get_cdf(ref, depth=depth)
    len_t = src.shape[0]
    args = list([(src[i, ...], cdf_r, depth) for i in range(len_t)]) 
    result = Parallel(n_jobs=n_jobs)(delayed(histogram_match_cdf)(*arg) for arg in args)
    return np.stack(result)

def pb_corr_all(img_tcyx:np.ndarray, ref_t:int=0, depth:int=12, n_jobs:int=48):
    ch_list = []
    for ch in range(img_tcyx.shape[1]):
        ch_im = img_tcyx[:, ch, ...]
        ch_list.append(pb_corr_parallel(ch_im, ch_im[ref_t,...], depth=depth, n_jobs=n_jobs))
    return np.transpose(np.stack(ch_list), (1, 0, 2, 3))

def max_binning(y, t, size):
    bins = np.arange(np.min(t), np.max(t), size)
    binned_values = np.zeros(bins.shape[0] - 1)
    t_binned = bins[0:-1]
    for i in range(1, bins.shape[0]):
        binned_values[i-1] = np.percentile(y[(t >= bins[i-1]) & (t < bins[i])], 90)
    return binned_values, t_binned
def mean_binning(y, t, size):
    bins = np.arange(np.min(t), np.max(t), size)
    binned_values = np.zeros(bins.shape[0] - 1)
    t_binned = bins[0:-1]
    for i in range(1, bins.shape[0]):
        binned_values[i-1] = np.mean(y[(t >= bins[i-1]) & (t < bins[i])])
    return binned_values, t_binned
def min_binning(y, t, size):
    bins = np.arange(np.min(t), np.max(t), size)
    binned_values = np.zeros(bins.shape[0] - 1)
    t_binned = bins[0:-1]
    for i in range(1, bins.shape[0]):
        binned_values[i-1] = np.percentile(y[(t >= bins[i-1]) & (t < bins[i])], 10)
    return binned_values, t_binned
def med_binning(y, t, size):
    bins = np.arange(np.min(t), np.max(t), size)
    binned_values = np.zeros(bins.shape[0] - 1)
    t_binned = bins[0:-1]
    for i in range(1, bins.shape[0]):
        binned_values[i-1] = np.percentile(y[(t >= bins[i-1]) & (t < bins[i])], 10)
    return binned_values, t_binned

def imshow(*args):
    shape = len(args)
    fig, ax = plt.subplot_mosaic(np.reshape(np.arange(shape), (1, shape)))
    for i, img in enumerate(args):
        ax[i].axis("off")
        ax[i].imshow(img, cmap="gray")
    plt.show()

def perc99(a, axis):
    return np.nanpercentile(a, 99.5, axis=axis)

def bin_images(sc_img, n_bin=2, px_size=0.22, dt = 100, stats=[np.nansum, np.nanmean, np.nanmax, perc99], names=["int_I", "mean_I", "max_I", "per99_I"]):
    
    T = np.arange(0, sc_img.shape[0]) * dt # seconds
    C = np.arange(0, sc_img.shape[1])
    bin_dim = sc_img.shape[2] // n_bin

    X = sc_img.shape[3] / 2 * ( np.arange(-n_bin//2, n_bin // 2) +  1/2) * px_size
    Y = sc_img.shape[2] / 2 * ( np.arange(-n_bin//2, n_bin // 2) +  1/2) * px_size

    TCYX = np.meshgrid(T, C, Y, X, indexing='ij')
    TCYX = [arr.reshape((sc_img.shape[0], sc_img.shape[1], n_bin**2)) for arr in TCYX]

    m = nn.Unfold(bin_dim, stride=bin_dim)
    unfold = m(torch.tensor(sc_img, dtype=torch.float64)).numpy()
    refold = unfold.reshape((sc_img.shape[0], sc_img.shape[1], bin_dim, bin_dim, n_bin ** 2))
    refold_nan = refold.astype(np.float64)
    refold_nan[refold_nan == 0] = np.nan

    df_arr = np.reshape(np.array(TCYX), (4, sc_img.shape[0] * sc_img.shape[1] * n_bin**2))
    df = pd.DataFrame(df_arr.T, columns=["t", "c", "y", "x"])
    
    for func, name in zip(stats, names):
        stat_arr = func(refold_nan, axis=(2, 3)).flatten()
        df[name] = stat_arr
    return df

def add_meta_data(df, data, rot_data):
    for i, (id, group) in enumerate(df.groupby("t")):
        group["T_abs"] = data["t"][i]
        group["X_abs"] = data["x"][i]
        group["Y_abs"] = data["y"][i]
        group["Theta"] = rot_data[i]
        #print(group)

def deterministic_denoise(img_stack, t_axis=0):
    noise = np.min(img_stack, axis=(t_axis,))
    return img_stack - noise

def prepend_timelapse(img_t, t):
    start_frame = img_t[t,...]
    return np.concatenate([start_frame[np.newaxis, ...], img_t])

def prepend_frame(img_t, img):
    return np.concatenate([img[np.newaxis, ...], img_t])

def sc_pb_corr_batch(sc_imgs, channels=(0,), dt=100, dx=0.22):
    ch_bool = np.isin(np.arange(sc_imgs[0].shape[1]), channels)
    pb_sc_imgs = []
    for sc_img in sc_imgs:
        df = bin_images(sc_img, 1, dx, dt)
        pb_sc = sc_img.copy()
        for ch, corr in zip(np.arange(sc_img.shape[1]), ch_bool):
            if corr:
                y = df[df["c"] == ch]["mean_I"]
                t = df[df["c"] == ch]["t"]
                plt.plot(y, t)
                plt.show()
                popt, y_est = fit_exp(y, t, lambd0=0.0128 / dt)
                y_est = np.broadcast_to(y_est.to_numpy()[:, np.newaxis, np.newaxis], sc_img[:, ch, ...].shape)
                pb_sc[:, ch,...] = np.round(sc_img[:,ch, ...] / y_est * (popt[1] + popt[2])).astype(np.uint16)
        pb_sc_imgs.append(pb_sc)
    return pb_sc_imgs

def sc_pb_corr(sc_img, ch_bool=[True], dt=100, dx=0.22):
    df = bin_images(sc_img, 1, dx, dt)
    pb_sc = sc_img.copy()
    for ch, corr in zip(np.arange(sc_img.shape[1]), ch_bool):
        if corr:
            y = df[df["c"] == ch]["mean_I"]
            t = df[df["c"] == ch]["t"]
            # popt, y_est = fit_exp(y, t, lambd0=0.0128 / dt)
            # y_est = np.broadcast_to(y_est.to_numpy()[:, np.newaxis, np.newaxis], sc_img[:, ch, ...].shape)
            # pb_sc[:, ch,...] = np.round(sc_img[:,ch, ...] / y_est * (popt[1] + popt[2])).astype(np.uint16)
            try:
                popt, y_est = fit_exp(y, t, lambd0=0.0128 / dt)
                y_est = np.broadcast_to(y_est.to_numpy()[:, np.newaxis, np.newaxis], sc_img[:, ch, ...].shape)
                pb_sc[:, ch,...] = np.round(sc_img[:,ch, ...] / y_est * (popt[1] + popt[2])).astype(np.uint16)
            except RuntimeError:
                print("Warning: Unable to pb correct")
                return sc_img
    return pb_sc

def sc_pb_corr_parallel(sc_imgs, channels=(0,), dt=100, dx=0.22, n_jobs=-1):
    ch_bool = np.isin(np.arange(sc_imgs[0].shape[1]), channels)
    args = list([(sc_img, ch_bool, dt, dx) for sc_img in sc_imgs])
    result = Parallel(n_jobs=n_jobs)(delayed(sc_pb_corr)(*arg) for arg in args)
    return list(result) 