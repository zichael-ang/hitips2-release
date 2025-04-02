from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# import ht2.segment as seg
from ht2.utils import *
from ht2.io_handler import *
from ht2.readers import ometiff_TCYX

from pipelines.default_pipelines import *

class IDL_PREPROCESS:
    def __init__(self, 
                 reader=None, 
                 preprocess=None, 
                 segment=None, 
                 nuc_track=None, 
                 sc_process=None, 
                 post_process=None,
                 meta={ "dt":100,
                        "dx":0.22 }):
        self.meta = meta
        if reader is None:
            reader = ometiff_TCYX
        self.reader = reader
        if preprocess is None:
            preprocess = DENOISE_MIN_PROJ_TCYX
        self.preprocess = preprocess
        if segment is None:
            segment = CELLPOSE_SEGMENT()
        self.segment = segment
        if nuc_track is None:
            nuc_track = BTRACK_NUCLEUS_TRACKING(min_t=200)
        self.nuc_track = nuc_track
        if sc_process is None:
            sc_process = SC_PROCESS(meta=meta)
        self.sc_process = sc_process
        if post_process is None:
            post_process = IDL_POSTPROCESS(meta=meta)
        self.post_process = post_process

        self.cached_image = None
    def load_image(self, fname):
        self.cached_image = self.reader(fname)
        return self.cached_image

    def run(self, fname, out_path):
        check_dir(out_path, create_if_not=True)
        sc_path    = out_path.joinpath("sc")
        idl_path   = out_path.joinpath("sc_idl")
        print("reading")
        im                        = self.reader(fname)
        print("preprocessing whole FOV")
        im                        = self.preprocess(im)
        print("segmenting image")
        mask                      = self.segment(im, out_path)
        print("isolating nuclei")
        sc_imgs, sc_masks         = self.nuc_track(im, mask, sc_path)
        print("photobleach + registration")
        pbr_sc_imgs, pbr_sc_masks = self.sc_process(sc_imgs, sc_masks, sc_path)
        print("post processing for IDL")
        _                         = self.post_process(pbr_sc_imgs, sc_imgs, idl_path)

    def test(self, diameters=[65]):
        im = self.cached_image
        t = np.arange(im.shape[0])
        mean_int = np.mean(im, axis=(2, 3))
        plt.plot(t, mean_int[:, 0], t, mean_int[:, 1])
        plt.title("Mean Image Intensity")
        plt.xlabel("t (frames)")
        plt.show()
        im = self.preprocess(self.cached_image)
        for t in np.arange(0, im.shape[0], 200):
            for d in diameters:
                test_mask = self.segment.test(im[t, ...], diameter=d)
                plt.imshow(test_mask)
                plt.title("Mask, t={}, d={}".format(t, d))
                plt.show()
                plt.imshow(im[t, 0, ...] + im[t, 1, ...])
                plt.title("Image, t={}, d={}".format(t, d))
                plt.show()

# class IDLPRE:
#     params = {
#         "meta": {        
#             "dt":100,
#             "dx":0.22
#         },
#         "seg": {
#             "batch_size":8, 
#             "channels":[0, 1],
#             "channel_axis":None,
#             "invert":False,
#             "normalize":True,
#             "diameter":65
#         },
#         "trk": {
#             "min_t":300,
#             "max_area_std":300,
#         },
#         "iso": {
#             "sz":100,
#             "ret_masked":True
#         },
#         "reg": {
#             "fixed_t":0,
#             "channels":[0, 1],
#             "order":1,
#         },
#         "pb": {
#             "channels":(0,)
#         },
#         "idl": {
#             "burst_th":1.2,
#             "never_th":1.15,
#             "t_th":100,
#             "n_bin":2,
#             "max_spots":2,
#             "ch_map":{0:"R", 1:"G"}
#         }
#     }

#     def __init__(self, gpu=True, n_jobs=-1):
#         if gpu:
#             device = torch.device("cuda")
#             print(device)
#             self.cp_model = seg.gpu_model(m_args={"model_type":"cyto3", 
#                             "gpu":gpu,
#                             "device":device})
#         else:
#             self.cp_model = seg.gpu_model(m_args={"model_type":"cyto3", "gpu":gpu})
#         self.n_jobs = n_jobs

#     def set_params(self, **kwargs):
#         for key in kwargs:
#             self.params[key].update(kwargs[key])
#             print(kwargs[key])
#         self.cp_model.update_e_args(self.params["seg"])
#         print("updated parameters: {}".format(kwargs))

#     def imread(self, fpath):
#         print("reading image...")
#         img_tcyx = tif.imread(fpath)
#         if len(img_tcyx.shape) < 4:
#             img_tcyx = np.expand_dims(img_tcyx, 1)
#         return img_tcyx

#     def get_mask(self, mpath, img_tcyx):
#         if check_f( mpath):
#             print("reading mask...")
#             mask_t = tif.imread(mpath)
#             # mask_t, _ = read_im(opath.joinpath("MASK_{}.ome.tif".format(name)))
#         else:
#             print("segmenting...")
#             mask_t = self.cp_model.batch(img_tcyx, "TCYX", e_args=self.params["seg"])
#             print("writing mask...")
#             tif.imwrite(mpath, mask_t.astype(np.int16), bigtiff=True, metadata={'axes':'TYX'})
#         return mask_t

#     def get_denoise(self, dpath, img_tcyx):
#         if check_f(dpath):
#             print("reading denoised...")
#             img_tcyx_denoise = tif.imread(dpath)
#             # img_tcyx_denoise, meta = read_im(dpath) 
#             if len(img_tcyx_denoise.shape) < 4:
#                 img_tcyx_denoise = np.expand_dims(img_tcyx_denoise, 1)
#         else:
#             print("denoising...")
#             img_tcyx_denoise = deterministic_denoise(img_tcyx)
#             tif.imwrite(dpath, img_tcyx, bigtiff=True, metadata={'axes':'TCYX'})
#         return img_tcyx_denoise

#     def get_tracks(self, img_tcyx, mask_t):
#         tracker = btrack_model(Path(r"\\shares2.dkisilon2.niddk.nih.gov\DKMIROSHNIKOVALAB\Lab Notebooks\Ike\Code\hitips2\default_configs\tracker_config.json"))
#         track_objs = tracker.get_objects(mask_t.astype(np.int64), img_tcyx[:, 0, ...])
#         track_objs = prune_objs(track_objs)
#         tracks = tracker.track_nuclei(track_objs)
#         return tracks

#     def run(self, name, fpath):
#         opath = fpath.parent.joinpath(name)
#         check_dir(opath, create_if_not=True)
#         mpath = opath.joinpath("MASK_{}.tif".format(name))
#         sc_path = opath.joinpath("sc".format(name))
#         check_dir(sc_path, create_if_not=True)
        
#         img_tcyx = self.imread(fpath)
#         mask_t = self.get_mask(mpath, img_tcyx)
#         img_tcyx = deterministic_denoise(img_tcyx)

#         print("nuclei tracking...")
#         tracks = self.get_tracks(img_tcyx, mask_t)
#         tracks.to_csv(opath.joinpath("nuclei_trk_{}.csv".format(name)))
#         tracks = prune_tracks(tracks, **self.params["trk"])
#         tracks.to_csv(opath.joinpath("pruned_nuclei_trk_{}.csv".format(name)))

#         print("isolating nuclei...")
#         sc_imgs, sc_masks = isolate_tracks(img_tcyx, mask_t, tracks, **self.params["iso"])

#         print("registering nuclei...")
#         sc_imgs = sc_pb_corr_parallel(sc_imgs, self.params["pb"]["channels"], self.params["meta"]["dt"], self.params["meta"]["dt"], self.n_jobs)

#         pbr_sc_imgs, pbr_sc_masks, rots = phase_corr_batch(sc_imgs, sc_masks, **self.params["reg"])

#         print("writing nuclei...")
#         for i, (sc_img, sc_mask, pbr_img, pbr_mask) in enumerate(zip(sc_imgs, sc_masks, pbr_sc_imgs, pbr_sc_masks)):
#             meta = {'axes':'TCYX', 'DimensionOrder':'TCYX', 'SizeT': sc_img.shape[0], 'SizeC':sc_img.shape[1], 'SizeZ':1, 'SizeY':sc_img.shape[2], 'SizeX':sc_img.shape[3]}

#             tif.imwrite(sc_path.joinpath("{}_unuc.tif".format(i,)), sc_img, bigtiff=True, metadata=meta)
#             tif.imwrite(sc_path.joinpath("{}_umask.tif".format(i,)), sc_mask, metadata={'axes':'TYX', 'DimensionOrder':'TYX'})

#             tif.imwrite(sc_path.joinpath("{}_rnuc.tif".format(i,)), pbr_img, bigtiff=True, metadata=meta)
#             tif.imwrite(sc_path.joinpath("{}_rmask.tif".format(i,)), pbr_mask, metadata={'axes':'TYX', 'DimensionOrder':'TYX'})

#         idl_path = opath.joinpath("sc_idl".format(name))

#         check_dir(idl_path, create_if_not=True)
#         check_dir(idl_path.joinpath("bins"), create_if_not=True)
#         check_dir(idl_path.joinpath("bursting"), create_if_not=True)
#         check_dir(idl_path.joinpath("maxIP"), create_if_not=True)

#         burst_th = self.params["idl"]["burst_th"]
#         never_th = self.params["idl"]["never_th"]
#         t_th = self.params["idl"]["t_th"]
#         n_bin = self.params["idl"]["n_bin"]
#         max_spots = self.params["idl"]["max_spots"]
#         ch_map = self.params["idl"]["ch_map"]
#         for i, (reg_sc, reg_mask) in enumerate(zip(pbr_sc_imgs, pbr_sc_masks)):
#             df = bin_images(reg_sc, n_bin=1, px_size=0.22, dt=100)
#             df.to_csv(idl_path.joinpath("bins", "{:03d}_bin{}.csv".format(i, 1)))
#             df = bin_images(reg_sc, n_bin=n_bin, px_size=0.22, dt=100)
#             df.to_csv(idl_path.joinpath("bins", "{:03d}_bin{}.csv".format(i, n_bin ** 2)))
#             for (ch, ), group1 in df.groupby(["c"]):
#                 ch_im = reg_sc[:, int(ch), ...]
                

#                 n_burst = np.zeros(n_bin ** 2)
#                 t_burst = np.zeros(n_bin ** 2)
#                 never = np.zeros(n_bin ** 2)

#                 mean_t = group1.groupby("t")["max_I"].mean()
#                 for j, (pos, group2) in enumerate(group1.groupby(["y", "x"])):
#                     norm_max = group2["max_I"].values / mean_t
#                     bursting = norm_max > burst_th # TODO: Sort by burst size, not just first
#                     n_bursting = np.sum(bursting)
#                     if n_bursting > 0:
#                         inds = np.where(np.atleast_1d(bursting))[0]
#                         first = np.min(inds)
#                     else:
#                         first = np.nan
#                     n_burst[j] = n_bursting
#                     t_burst[j] = first

#                     never[j] = np.all(norm_max < never_th)
#                     group2["quad_id"] = j

#                 accepted = (n_burst > 0) * (t_burst < t_th)
#                 burst_inds = t_burst[accepted]
#                 max_frame = np.max(ch_im, axis=0)

#                 tif.imwrite(idl_path.joinpath("maxIP", "{:03d}{}_max.tif".format(int(i), ch_map[int(ch)])), max_frame, metadata={'axes':'YX'})

#                 pre_img = np.concatenate([max_frame[np.newaxis, ...], ch_im])
#                 tif.imwrite(idl_path.joinpath("bursting", "{:03d}{}_spot{}_t{}.tif".format(int(i), ch_map[int(ch)], 2, pre_img.shape[0])), pre_img, metadata={'axes':'TYX'})
#                 pre_img = np.concatenate([max_frame[np.newaxis, ...], sc_imgs[i][:, int(ch), ...]])
#                 tif.imwrite(idl_path.joinpath("bursting", "{:03d}{}_unregspot{}_t{}.tif".format(int(i), ch_map[int(ch)], 2, pre_img.shape[0])), pre_img, metadata={'axes':'TYX'})
#                 if len(burst_inds) > 2:
#                     burst_inds = np.sort(burst_inds)[:2]

#                 for spot, T in enumerate(burst_inds):
#                     pre_img = prepend_timelapse(reg_sc[:, int(ch), ...], int(T))
#                     tif.imwrite(idl_path.joinpath("bursting", "{:03d}{}_spot{}_t{}.tif".format(int(i), ch_map[int(ch)], spot, pre_img.shape[0])), pre_img, metadata={'axes':'TYX'})

#                     pre_img = prepend_frame(sc_imgs[i][:, int(ch), ...], reg_sc[int(T), int(ch), ...])
#                     tif.imwrite(idl_path.joinpath("bursting", "{:03d}{}_unregspot{}_t{}.tif".format(int(i), ch_map[int(ch)], spot, pre_img.shape[0])), pre_img, metadata={'axes':'TYX'})
#         return True