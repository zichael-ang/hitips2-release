import numpy as np
import scipy.ndimage as im
from joblib import Parallel, delayed
from scipy.fft import fft2, fftshift, fft
import skimage.transform as transform

def get_border(sc_mask, thickness=1):
    # Mask is TYX
    struct = np.array([im.generate_binary_structure(2, 1)])
    struct = im.iterate_structure(struct, thickness)
    sc_mask_dilate = im.binary_dilation(sc_mask, struct)
    sc_mask_erode = im.binary_erosion(sc_mask, struct)
    return np.logical_and(sc_mask_dilate, ~sc_mask_erode)

def get_border_batch(sc_masks, thickness=1, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(get_border)(mask, thickness) for mask in sc_masks)
    return list(results)

def nuc_fluc_stats(self, border_masks):
    r_mean_table = np.zeros((len(border_masks), 360))
    r_var_table = np.zeros((len(border_masks), 360))
    NPS = []
    rtT = []
    for i, border in enumerate(border_masks):
        border_w = np.stack([transform.warp_polar(border[i, ...]) for i in range(border.shape[0])])
        r, theta  = np.meshgrid(np.arange(0, border_w.shape[-1]), np.arange(0, 360))
        border_r = (border_w * r).astype(np.float32)
        border_r[border_r==0] = np.nan
        r_t_theta = np.nanmean(border_r, axis=-1)
        r_mu = np.mean(r_t_theta, axis=0)
        r_var = np.var(r_t_theta, axis=0)
        R_f = fftshift(fft(r_t_theta, axis=0), axes=0).T
        f_space = np.linspace(-1/2, 1/2, R_f.shape[1])[R_f.shape[1] // 2 +1:]
        R_f = np.abs(R_f[:, R_f.shape[1] // 2 +1:])
        r_mean_table[i, :] = r_mu
        r_var_table[i, :] = r_var
        NPS.append((R_f, f_space))
        rtT.append(r_t_theta)
    return rtT, r_mean_table, r_var_table, NPS