from ht2.io_handler import check_f
from imaris_ims_file_reader.ims import ims
import zarr
import tifffile as tif
import numpy as np
import nd2

def ims_read(fpath):
    # Always reads in TCZYX
    check_f(fpath)
    a = ims(fpath, aszarr=True)
    arr = zarr.open(a, mode='r')
    arr = arr[:]
    return arr

def ometiff_TCYX(fpath):
    img_tcyx = tif.imread(fpath)
    if len(img_tcyx.shape) < 4:
        img_tcyx = np.expand_dims(img_tcyx, 1)
    return img_tcyx

def nd2_TCYX(fpath):
    img_btcyx = nd2.imread(fpath)
    return img_btcyx

# TODO CZI reader

# def batch_read_sc(f_list, template="{cell}_{name}_{im_type}.{ext}"):
#     im_list = []
#     meta_list = []
#     for fpath in f_list:
#         fpath = Path(fpath)
#         parsed_keys = parse.parse(template, fpath.name)
#         if parsed_keys is None:
#             raise NotImplementedError
#         if "ext" not in parsed_keys:
#             raise NotImplementedError
#         im_list.append(tif.imread(fpath))
#         meta_list.append(fpath.name)
#     return im_list, meta_list

if __name__=="__main__":
    # x = ims_read()
    x = nd2_TCYX(r"\\shares2.dkisilon2.niddk.nih.gov\DKMIROSHNIKOVALAB\Lab Notebooks\Ike\005\006_FN_0_Stretch_crop.nd2")
    print(x.shape)