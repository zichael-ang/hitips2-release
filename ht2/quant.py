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
# %%
def pearson_cross_correlation(sig1, sig2, mode="same", scale=True):
    if scale:
        mu1, mu2 = np.mean(sig1), np.mean(sig2)
        std1, std2 = np.std(sig1), np.std(sig2)
    else:
        mu1, mu2 = 0, 0
        std1, std2 = 1, 1
    xcorr = sig.correlate((sig1 - mu1) / std1, (sig2 - mu2) / std2, mode=mode)
    t = sig.correlation_lags(len(sig1), len(sig2), mode=mode)
    xcorr = xcorr / (len(sig1) - np.abs(t) + 1)
    return xcorr, t

def spearman_cross_correlation(sig1, sig2, mode="same"):
    r_sig1 = np.argsort(sig1).argsort()
    r_sig2 = np.argsort(sig2).argsort()
    s_xcorr = pearson_cross_correlation(r_sig1, r_sig2, mode=mode)
    return s_xcorr

def compG_multiTau(v, t, n=4, ctr=0):
    """v: data vector (channels=rows), t: time, n: bin every n steps.\n--> Matrix of G, time vector"""
    def compInd(v1,v2):
        if len(t)<2:
            return np.array([[], []]).T
        tau=[]; G=[]; t0=t*1.; i=0; dt=t0[1]-t0[0]
        while i<t0.shape[0]:
            tau.append(i*dt)
            G.append(np.mean(v1[:int(v1.shape[0]-i)]*v2[int(i):]))
            if i==n:
                i=i/2
                dt*=2
                t0,v1,v2=np.c_[t0,v1,v2][:int(t0.shape[0]/2)*2].T.reshape(3,-1,2).mean(2)
            i+=1
        return np.array([tau,G]).T
    if ctr: vCtr=((v.T-np.mean(v,1)).T)
    else: vCtr=v
    res=np.array([[ compInd(v1,v2) for v2 in vCtr] for v1 in vCtr])
    return ( res[:,:,:,1].T /(np.dot(np.mean(v,1).reshape(-1,1),np.mean(v,1).reshape(1,-1)))).T, res[0,0,:,0]

# %%
def label_states(arr: np.ndarray, state, inverse=False):
    blocks = np.cumsum(np.diff(arr, prepend=np.array([0])) != 0)
    if inverse:
        return (arr != state) * blocks
    return (arr == state) * blocks

def get_lengths(arr: np.ndarray):
    blocks = np.cumsum(np.diff(arr, prepend=np.array([0])) != 0)

    labeled = arr * blocks
    counts = np.bincount(labeled)[1:]
    return counts[counts > 0]

def compute_burst_size(intensity, states, off_correction=True):
    on_label = label_states(states, 1)
    off_label = label_states(states, 0)
    
    # Identify unique time domains (ons and offs), for each time, compute length and integral of the time domain, then return list of burst sizes, their start times and lengths, and their corrected values
    pass

def compute_frequency(intensity, states):
    pass