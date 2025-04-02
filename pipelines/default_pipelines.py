from tqdm import tqdm
from pathlib import Path
import pickle

import numpy as np
import tifffile as tif

from ht2.utils import *
from ht2.io_handler import *
from ht2.tracking import *
from ht2.register import *
from ht2.quant import *

import cellpose.models

def DENOISE_MIN_PROJ_TCYX(im):
    return deterministic_denoise(im, t_axis=0)

def batch_write(im_list, out_path, keyword, metadata={}, bigtiff=True):
    for i, im in enumerate(im_list):
        tif.imwrite(out_path.joinpath("{}_{}.tif".format(i, keyword)), im, bigtiff=bigtiff, metadata=metadata)

class CELLPOSE_SEGMENT:
    def __init__(self, 
            model_type = "cyto3",
            gpu=True,
            e_args={
            "batch_size":16, 
            "channels":[0, 1],
            "channel_axis":None,
            "invert":False,
            "normalize":True,
            "diameter":65},
            ignore_existing=False
        ):
        self.e_args = e_args
        self.model_instance = cellpose.models.Cellpose(model_type=model_type, gpu=gpu)
        self.ignore = ignore_existing

    def __call__(self, im, out_path):
        check_dir(out_path, create_if_not=True)
        if (self.ignore) or (check_f(out_path.joinpath("MASK.tif")) == False):
            print("segmenting...")
            mask_t = self.batch(im, "TCYX")
            print("writing mask...")
            tif.imwrite(out_path.joinpath("MASK.tif"), mask_t.astype(np.int16), bigtiff=True, metadata={'axes':'TYX'})
        else:
            mask_t = tif.imread(out_path.joinpath("MASK.tif"))
        return mask_t

    def test(self, im: np.ndarray, **kwargs):
        print(kwargs)
        for key in kwargs:
            self.e_args[key] = kwargs[key]
        mask, _, _, _  = self.model_instance.eval(im, **self.e_args)
        return mask

    def set_args(self, **kwargs):
        for key in kwargs:
            self.e_args[key] = kwargs[key]
            print("Set: ", key, " to: ", kwargs[key])

    def batch(self, im_stack: np.ndarray, dim_ord="TCYX", N=16, pad=4):
        # Hacky way to segmenting a large batch of images at once, cellpose doesn't play well with batched images for whatever reason
        im_stack_TYXC = reorder_img(im_stack, dim_ord, "TYXC")
        im_stack_TYXC_b = np.pad(im_stack_TYXC, ((0, 16 - (im_stack_TYXC.shape[0] % 16)), (pad,pad), (pad, pad), (0, 0)), mode="constant")
        pad_shape = im_stack_TYXC_b.shape
        im_stack_TYXC_b = np.reshape(im_stack_TYXC_b, (N, pad_shape[0] * pad_shape[1] // N, pad_shape[2], pad_shape[3]))
        mask_stack = np.zeros(im_stack_TYXC_b.shape[:-1])
        for i in tqdm(range(im_stack_TYXC_b.shape[0])):
            mask_stack[i,...], _, _, _  = self.model_instance.eval(im_stack_TYXC_b[i, ...], **self.e_args)
        masks = np.reshape(mask_stack, pad_shape[:-1])
        return masks[:im_stack_TYXC.shape[0], pad:-pad, pad:-pad]

class BTRACK_NUCLEUS_TRACKING:
    def __init__(self,
            min_t = 100, 
            max_area_std = 200,
            sz = 100,
            ret_masked=True):
        self.min_t = min_t  
        self.max_area_std = max_area_std
        self.sz = sz
        self.ret_masked = ret_masked

    def __call__(self, img_tcyx, mask_t, out_path):
        check_dir(out_path, create_if_not=True)
        tracker = btrack_model(Path(r"default_configs\tracker_config.json"))
        track_objs = tracker.get_objects(mask_t.astype(np.int64), img_tcyx[:, 0, ...])
        track_objs = prune_objs(track_objs)
        tracks = tracker.track_nuclei(track_objs)
        with open(out_path.joinpath("nuclei_tracks.pkl"), "wb") as f:
            pickle.dump(tracks, f)
            tracks.to_csv(out_path.joinpath("pruned_nuclei_tracks.csv"))

        tracks = prune_tracks(tracks, self.min_t, self.max_area_std)
        with open(out_path.joinpath("pruned_nuclei_tracks.pkl"), "wb") as f:
            pickle.dump(tracks, f)
            tracks.to_csv(out_path.joinpath("pruned_nuclei_tracks.csv"))

        sc_imgs, sc_masks = isolate_tracks(img_tcyx, mask_t, tracks, self.sz, self.ret_masked)

        crop_path = out_path.joinpath("cropped")
        check_dir(crop_path, create_if_not=True)
        meta_im = {'axes':'TCYX', 'DimensionOrder':'TCYX', 'SizeT': sc_imgs[0].shape[0], 'SizeC':sc_imgs[0].shape[1], 'SizeZ':1, 'SizeY':sc_imgs[0].shape[2], 'SizeX':sc_imgs[0].shape[3]}
        meta_mask = {'axes':'TYX', 'DimensionOrder':'TYX', 'SizeT': sc_imgs[0].shape[0], 'SizeZ':1, 'SizeY':sc_imgs[0].shape[2], 'SizeX':sc_imgs[0].shape[3]}

        batch_write(sc_imgs, crop_path, "raw", metadata=meta_im, bigtiff=True)
        batch_write(sc_masks, crop_path, "raw", metadata=meta_im, bigtiff=True)

        return sc_imgs, sc_masks

class SC_PROCESS:
    def __init__(self, 
                 reg_channels=(0,1), 
                 fixed_t=0,
                 order=1,
                 pb_channels=(0,), 
                 meta={ "dt":100,
                        "dx":0.22 }):
        self.meta = meta
        self.reg_channels = reg_channels
        self.fixed_t = fixed_t
        self.order = order
        self.pb_channels = pb_channels
    def __call__(self, sc_imgs, sc_masks, out_path):
        check_dir(out_path, create_if_not=True)
        pb_path = out_path.joinpath("pb")
        pbr_path = out_path.joinpath("pbr")
        check_dir(out_path.joinpath("pb"), create_if_not=True)
        check_dir(out_path.joinpath("pbr"), create_if_not=True)

        meta_im = {'axes':'TCYX', 'DimensionOrder':'TCYX', 'SizeT': sc_imgs[0].shape[0], 'SizeC':sc_imgs[0].shape[1], 'SizeZ':1, 'SizeY':sc_imgs[0].shape[2], 'SizeX':sc_imgs[0].shape[3]}
        meta_mask = {'axes':'TYX', 'DimensionOrder':'TYX', 'SizeT': sc_imgs[0].shape[0], 'SizeZ':1, 'SizeY':sc_imgs[0].shape[2], 'SizeX':sc_imgs[0].shape[3]}

        print("correcting photobleaching...")
        pb_sc_imgs = sc_pb_corr_parallel(sc_imgs, self.pb_channels, self.meta["dt"], self.meta["dt"])

        batch_write(pb_sc_imgs, pb_path, "pb", metadata=meta_im, bigtiff=True)

        print("registering nuclei...")
        pbr_sc_imgs, pbr_sc_masks, rots = phase_corr_batch(pb_sc_imgs, sc_masks, fixed_t=self.fixed_t, order=self.order, channels=self.reg_channels)

        batch_write(pbr_sc_imgs, pbr_path, "pbr", metadata=meta_im, bigtiff=True)
        batch_write(pbr_sc_masks, pbr_path, "pbr_mask", metadata=meta_mask)

        return pbr_sc_imgs, pbr_sc_masks

class IDL_POSTPROCESS:
    def __init__(self,
                 burst_th=1.2, 
                 t_th=10, 
                 n_bin=2, 
                 ch_map={0:"R", 1:"G"},
                 meta={ "dt":100,
                        "dx":0.22 }):
        self.burst_th = burst_th
        self.t_th = t_th
        self.n_bin = n_bin
        self.ch_map = ch_map
        self.meta = meta
    def __call__(self, pbr_sc_imgs, sc_imgs, idl_path):
        check_dir(idl_path, create_if_not=True)
        check_dir(idl_path.joinpath("bins"), create_if_not=True)
        check_dir(idl_path.joinpath("bursting"), create_if_not=True)
        check_dir(idl_path.joinpath("maxIP"), create_if_not=True)

        bins_path = idl_path.joinpath("bins")
        bursting_path = idl_path.joinpath("bursting")
        maxIP_path = idl_path.joinpath("maxIP")

        burst_th = self.burst_th
        t_th = self.t_th
        n_bin = self.n_bin
        ch_map = self.ch_map
        for i, reg_sc in enumerate(pbr_sc_imgs):
            df = bin_images(reg_sc, n_bin=1, px_size=self.meta["dx"], dt=self.meta["dt"])
            df.to_csv(bins_path.joinpath("{:03d}_bin{}.csv".format(i, 1)))
            df = bin_images(reg_sc, n_bin=n_bin, px_size=self.meta["dx"], dt=self.meta["dt"])
            df.to_csv(bins_path.joinpath("{:03d}_bin{}.csv".format(i, n_bin ** 2)))

            for (ch, ), group1 in df.groupby(["c"]):
                ch_im = reg_sc[:, int(ch), ...]

                n_burst = np.zeros(n_bin ** 2)
                t_burst = np.zeros(n_bin ** 2)

                mean_t = group1.groupby("t")["max_I"].mean()
                for j, (_, group2) in enumerate(group1.groupby(["y", "x"])):
                    norm_max = group2["max_I"].values / mean_t
                    bursting = norm_max > burst_th # TODO: Sort by burst size, not just first
                    n_bursting = np.sum(bursting)
                    if n_bursting > 0:
                        inds = np.where(np.atleast_1d(bursting))[0]
                        first = np.min(inds)
                    else:
                        first = np.nan
                    n_burst[j] = n_bursting
                    t_burst[j] = first
                    group2["quad_id"] = j

                accepted = (n_burst > 0) * (t_burst < t_th)
                burst_inds = t_burst[accepted]
                max_frame = np.max(ch_im, axis=0)

                tif.imwrite(maxIP_path.joinpath("{:03d}{}_max.tif".format(int(i), ch_map[int(ch)])), max_frame)

                pre_img = np.concatenate([max_frame[np.newaxis, ...], ch_im])
                tif.imwrite(bursting_path.joinpath( "{:03d}{}_spot{}_t{}.tif".format(int(i), ch_map[int(ch)], 2, pre_img.shape[0])), pre_img)

                pre_img = np.concatenate([max_frame[np.newaxis, ...], sc_imgs[i][:, int(ch), ...]])
                tif.imwrite(bursting_path.joinpath( "{:03d}{}_unregspot{}_t{}.tif".format(int(i), ch_map[int(ch)], 2, pre_img.shape[0])), pre_img)

                if len(burst_inds) > 2:
                    burst_inds = np.sort(burst_inds)[:2]

                for spot, T in enumerate(burst_inds):
                    pre_img = prepend_timelapse(reg_sc[:, int(ch), ...], int(T))
                    tif.imwrite(bursting_path.joinpath( "{:03d}{}_spot{}_t{}.tif".format(int(i), ch_map[int(ch)], spot, pre_img.shape[0])), pre_img)

                    pre_img = prepend_frame(sc_imgs[i][:, int(ch), ...], reg_sc[int(T), int(ch), ...])
                    tif.imwrite(bursting_path.joinpath( "{:03d}{}_unregspot{}_t{}.tif".format(int(i), ch_map[int(ch)], spot, pre_img.shape[0])), pre_img)

