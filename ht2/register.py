import numpy as np
import skimage.registration as reg
import skimage.transform as transform
import skimage.filters as fil
from joblib import Parallel, delayed
"""

Ideally replace as much as possible with opencv:
ocv has Dual TVL1 algo that may be a good replacement
ocv unfortunately deprecated estimate rigid, so need to figure out how 
to make 

TODO: Reorder and refactoring

Author(s): Ike H Zhang

References: HiTIPS
"""

def remove_boundary_nuclei(tracks, bounds):
    """
    groupby ID, find max and min of x, max and min of y, 
    if min and max outside of pass_box, add to drop list or cut-off the track
    drop if track too short, most likely goes out of bounds at the beginning and end"""
    pass

def isolate_tracks(img, l_mask, tracks, sz = 128, ret_masked = True):
    half_sz = sz//2
    padded_mask = np.pad(l_mask, ((0, 0), (half_sz, half_sz), (half_sz, half_sz)), mode="constant", constant_values=0) # Padding combats boundary nuclei issues
    padded_img = np.pad(img, ((0, 0), (0, 0), (half_sz, half_sz), (half_sz, half_sz)), mode="constant", constant_values=0)
    sc_masks = []
    sc_imgs = []
    for idx, data in tracks.iterrows():
        X, Y, T = np.array(data["x"]),np.array(data["y"]), np.array(data["t"])
        X_, Y_ = np.round(X).astype(int), np.round(Y).astype(int)
        Xu, Yu = X_ + sz, Y_ + sz

        sc_l_mask = np.zeros((T.shape[0], sz, sz), dtype=np.uint16)
        sc_img_t = np.zeros((T.shape[0], img.shape[1], sz, sz, ), dtype=np.uint16)
        for i, t, in enumerate(T):
            sc_mask = padded_mask[t, Y_[i]:Yu[i], X_[i]:Xu[i]] # cropped mask
            sc_img = padded_img[t, :, Y_[i]:Yu[i], X_[i]:Xu[i]] # cropped image
            sc_l_mask[i, ...] = sc_mask == data["class_id"][i]
            if ret_masked:
                sc_img_t[i, ...] = sc_img * sc_l_mask[i, ...]
            else:
                sc_img_t[i, ...] = sc_img
        sc_masks.append(sc_l_mask)
        sc_imgs.append(sc_img_t)
    return sc_imgs, sc_masks

def reg_preprocess(img, j=0.5):
    im_ = img.copy()
    im_ = fil.median(im_, np.array([[True, True, True], [True, True, True], [True, True, True]]))
    im_ = transform.rotate(im_, 20, order=1, preserve_range=True)
    im_ = fil.gaussian(im_, 4)
    im_ = im_ * fil.window(('tukey', 0.3), im_.shape)
    return im_

def normalize_ch(img, weights = [8, 1]):
    img_ = np.zeros_like(img)
    for i in range(img.shape[1]):
        img_[:, i, ...] = img[:, i, ...] * weights[i]
    return img_

def apply_rotation_stack(sc_tyx, rot_arr, order=1):
    reg_sc_tyx = np.zeros_like(sc_tyx)
    for t in range(sc_tyx.shape[0]):
        reg_sc_tyx[t] = transform.rotate(sc_tyx[t], rot_arr[t], order=order, preserve_range=True)
    return reg_sc_tyx

def fix_rot_arr(rot_arr, fixed_t=0):
    new_rot = np.flip(np.cumsum(np.concatenate([[0], np.flip(rot_arr[1:]), ])))
    new_rot = new_rot - new_rot[fixed_t]
    return new_rot

def pc_reg_recursive(sc_tyx, sc_mask, order=1):
    n_t = sc_tyx.shape[0]
    rot_arr = np.zeros(n_t)
    fixed_ = reg_preprocess(sc_tyx[0])
    fixed_m = sc_mask[0]
    fixed_w = transform.warp_polar(fixed_, order=order)
    fixed_m_w = transform.warp_polar(fixed_m, order=0)
    for i in range(1, n_t):
        moving_ = reg_preprocess(sc_tyx[i])
        moving_m = sc_mask[i]
        moving_w = transform.warp_polar(moving_, order=order)
        moving_m_w = transform.warp_polar(moving_m, order=0)
        shifts, _, _ = reg.phase_cross_correlation(fixed_w, moving_w, upsample_factor=8, reference_mask=fixed_m_w, moving_mask=moving_m_w)
        rot_arr[i] = shifts[0]
        fixed_w, fixed_m_w = moving_w, moving_m_w
    return rot_arr

def phase_corr_sc(sc_img, sc_mask, fixed_t=0, channels=[0, 1], order=1, normalize=True):
    if normalize:
        ch1 = np.sum(normalize_ch(sc_img[:, channels, ...]), axis=1)
    else:
        ch1 = np.sum(sc_img[:, channels, ...], axis=1)
    rot_arr = pc_reg_recursive(ch1, sc_mask, order=order)
    rot_arr = fix_rot_arr(rot_arr, fixed_t)
    reg_sc_img = np.zeros_like(sc_img)
    reg_sc_mask = np.zeros_like(sc_mask)
    reg_sc_mask = apply_rotation_stack(sc_mask, rot_arr, order=0)
    for ch in range(sc_img.shape[1]):
        reg_sc_img[:, ch, ...] = apply_rotation_stack(sc_img[:, ch, ...], rot_arr, order=order)
    return reg_sc_img, reg_sc_mask, rot_arr

def phase_corr_batch(sc_imgs, sc_masks, fixed_t=0, channels=[0], order=1, n_jobs=-1):
    reg_sc_imgs = []
    reg_sc_masks = []
    args = list([(sc_img, sc_mask, fixed_t, channels, order) for sc_img, sc_mask in zip(sc_imgs, sc_masks)])
    results = Parallel(n_jobs=n_jobs)(delayed(phase_corr_sc)(*arg) for arg in args)
    reg_sc_imgs, reg_sc_masks, rotations = list(zip(*results))
    return reg_sc_imgs, reg_sc_masks, rotations


# def apply_tform_stack(img: np.ndarray, tform_stack: np.ndarray, order=1):
#     """
#     Applies a list of transforms on an stack of images: 
#     expects images in T, Y, X, C"""
#     assert tform_stack.shape[0] == img.shape[0] - 1 # something went very wrong

#     for i in range(tform_stack.shape[0]):
#         img[i+1, ...] = transform.warp(img[i+1, ...], 
#                                         tform_stack[i,...], 
#                                         order=order, 
#                                         mode="constant",
#                                         cval=0)
#     return img

# def register_nuclei(sc_img: np.ndarray, 
#                     sc_mask: np.ndarray, 
#                     channel=0, ttype="euclidean"):
#     """Using nucleus stack, calculate optical flow and deform nucleus to fit
#     first/last/best quality image or register some other method

#     sc_img_ is a single channel image used for registration, need to keep sc_img 
#     to apply transform later on

#     TODO: Add support for more ttypes, euclidean only supported right now to preserve nuclear shape
#     Testing (duh)
#     I bet this one will be slow asl
#     Optimizations: use more opencv, impl ilk (supposedly faster and more robust)

#     sc_img : np.ndarray - single cell/nucleus stack 
#     axis : int - time axis (should be 1 for CTXY images)
#     channel : select channel for registration
#     warp : warping method, euclid, affine, or unconstrained

#     skimage.transform.estimate_transform and apply matrix transform
#     """
#     sc_img_ = sc_img[..., channel].copy()
#     n_tform = sc_img_.shape[0] - 1
#     tform_stack = np.zeros((sc_img_.shape[0], 3, 3))
#     x, y = np.meshgrid(np.arange(sc_img_.shape[1]), 
#                        np.arange(sc_img_.shape[2]))
#     for i in range(n_tform):
#         fixed, moving = sc_img_[i,...], sc_img_[i+1,...]
#         moving_mask = sc_mask[i+1,...]
#         u, v = reg.optical_flow_tvl1(fixed, moving)
#         src_coord = np.stack((x[moving_mask > 0], y[moving_mask > 0]))
#         dst_coord = src_coord + np.stack((u[moving_mask > 0], 
#                                           v[moving_mask > 0]))
#         tform = transform.estimate_transform(ttype, src_coord, dst_coord)
#         tform_stack[i,...] = tform.params.copy()
#     # Transform composition
#     # tform0 = A0, tformn = A0...An-1An, An is the tform from In+1 to In
#     for i in range(1, n_tform):
#         tform_stack[i,...] = np.matmul(tform_stack[i-1,...], tform_stack[i,...])
#     # Apply transforms
#     reg_sc_img = apply_tform_stack(sc_img, tform_stack)
#     reg_sc_mask = apply_tform_stack(sc_mask, tform_stack)
#     return reg_sc_img, reg_sc_mask

# def batch_separate_register(img, l_mask, tracks, masked=True):
#     sc_imgs_r = []
#     sc_masks_r = []
#     sc_imgs, sc_masks, ids = isolate_tracks(img, l_mask, tracks, 100, 8, masked)
#     for sc_img, sc_mask in  zip(sc_imgs, sc_masks):
#         reg_sc_img, reg_sc_mask = register_nuclei(sc_img, sc_mask, )
#         sc_imgs_r.append(reg_sc_img)
#         sc_masks_r.append(reg_sc_mask)
#     return sc_imgs_r, sc_masks_r
# def phase_corr_rotation(fixed, moving, fixed_m=None, moving_m=None, order=3):
#     fixed_, moving_ = reg_preprocess(fixed, moving)
#     if fixed_m == None:
#         fixed_m= fixed>0
#     if moving_m == None:
#          moving_m = moving>0
#     fixed_w = transform.warp_polar(fixed_, order=1)
#     moving_w = transform.warp_polar(moving_, order=1)
#     fixed_m_w = transform.warp_polar(fixed_m, order=0)
#     moving_m_w = transform.warp_polar(moving_m, order=0)
#     shifts, _, _ = reg.phase_cross_correlation(fixed_w, moving_w, upsample_factor=10, reference_mask=fixed_m_w, moving_mask=moving_m_w)
#     rot = shifts[0]
#     return rot

# def pc_reg_preprocessed(fixed_w, moving, fixed_m_w, moving_m, channel=0, order=3):
#     """
#     Single frame registration using already preprocessed and warped fixed images
#     Moving is in YXC format"""
#     moving_ = reg_preprocess(moving[...,channel])
#     moving_w = transform.warp_polar(moving_, order=1)
#     moving_m_w = transform.warp_polar(moving_m, order=0)
#     shifts, _, _ = reg.phase_cross_correlation(fixed_w, moving_w, upsample_factor=10, reference_mask=fixed_m_w, moving_mask=moving_m_w)
#     rot = shifts[0]

#     reg_moving = np.zeros_like(moving)
#     if len(moving.shape) > 3:
#         for ch in range(moving.shape[-1]):
#             reg_moving[..., ch] = transform.rotate(moving[..., ch], -rot, order=order, preserve_range=True)
#     else:
#         reg_moving = transform.rotate(moving, -rot, order=order, preserve_range=True)
#     reg_moving_m = transform.rotate(moving_m, -rot, order=0, preserve_range=True)
#     return reg_moving, reg_moving_m, rot 

# def phase_corr_sc(sc_tyxc, sc_mask, fixed_t=0, channel=0, order=3):
#     n_t = sc_tyxc.shape[0]
#     reg_sc = np.zeros_like(sc_tyxc)
#     reg_sc_m = np.zeros_like(sc_mask)
#     rot_arr = np.zeros(n_t)

#     fixed = sc_tyxc[fixed_t, ..., channel]
#     fixed_m = sc_mask[fixed_t, ...]

#     fixed_ = reg_preprocess(fixed)
#     fixed_w = transform.warp_polar(fixed_, order=1)
#     fixed_m_w = transform.warp_polar(fixed_m, order=0)
#     for i in range(n_t):
#         reg_sc[i, ...], reg_sc_m[i, ...], rot_arr[i] = pc_reg_preprocessed(fixed_w, sc_tyxc[i, ...], fixed_m_w, sc_mask[i, ...], channel, order)
#     return reg_sc, reg_sc_m, rot_arr

# def phase_corr_batch(sc_imgs, sc_masks, fixed_t = 0, channel=0, order=3, n_jobs=-1):
#     reg_sc_imgs = []
#     reg_sc_masks = []
#     args = list([(sc_img, sc_mask, fixed_t, channel, order) for sc_img, sc_mask in zip(sc_imgs, sc_masks)])
#     results = Parallel(n_jobs=n_jobs)(delayed(phase_corr_sc)(*arg) for arg in args)
#     reg_sc_imgs, reg_sc_masks, rotations = list(zip(*results))
#     return reg_sc_imgs, reg_sc_masks, rotations