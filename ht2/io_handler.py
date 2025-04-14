import shutil as sh
import glob
import parse
from pathlib import Path
import json
import pandas as pd
from typing import Union
"""
read ome tiffs, parse params, write images

Author(s): Ike H Zhang

References: HiTIPS
"""

def check_dir(dpath: Path, create_if_not = False, raise_if_false = False, raise_if_true = False):
    if create_if_not and not dpath.is_dir():
        dpath.mkdir()
    if raise_if_false and not dpath.is_dir():
        raise NotADirectoryError
    if raise_if_true and dpath.is_dir():
        raise IsADirectoryError
    return dpath.is_dir()

def check_f(fpath: Path, overwrite = False, create_if_not = False, raise_if_false = False, raise_if_true = False):
    if overwrite:
        fpath.touch(exist_ok=True)
    else:
        if create_if_not and not fpath.is_file():
            fpath.touch()
        if raise_if_false and not fpath.is_file():
            raise FileNotFoundError
        if raise_if_true and fpath.is_file():
            raise FileExistsError
    return fpath.is_file()

def create_dir(dpath: Path, raise_if_true = False):
    create_if_not = True 
    if create_if_not and not dpath.is_dir():
        dpath.mkdir()
    if raise_if_true and dpath.is_dir():
        raise IsADirectoryError
    return dpath

def read_params(in_path):
    check_f(Path(in_path), raise_if_false=True)
    with open(in_path, "r") as in_f:
        p_dict = json.load(in_f)
    return p_dict

def write_params(out_path, p_dict):
    check_f(Path(out_path), create_if_not=True, overwrite=False, raise_if_true=True)
    obj = json.dumps(out_path)
    with open(out_path, "w") as out_f:
        out_f.write(obj)
    return True

def get_uuid(row):
    return "_".join([row["field"], row["cell"], row["ID"]])

def batch_read_tracks(f_list, 
    trk_template="{prefix}_for_col1_row1_field{field}_cell{cell}_Ch{ch_num}_spot{spot}.{extension}"
    ):
    test_tracks = []
    for fpath in f_list:
        fpath = Path(fpath)
        parsed_keys = parse.parse(trk_template, fpath.name)
        if parsed_keys is None:
            raise NotImplementedError
        if parsed_keys["extension"] == "trk": # type: ignore
            temp = pd.read_csv(fpath, sep="\\s+")
            temp.columns = ["x", "y", "I", "t", "state"]
        elif parsed_keys["extension"] == "csv": # type: ignore
            temp = pd.read_csv(fpath, usecols=[5, 6, 7, 8])
            temp.columns = ["t", "x", "y", "I"]
        else:
            raise NotImplementedError
        temp["cell"] = int(parsed_keys["cell"]) # type: ignore
        temp["ID"] = int(parsed_keys["spot"]) # type: ignore
        temp["ch"] = int(parsed_keys["ch_num"])
        if "field" in parsed_keys: # type: ignore
            temp["field"] = int(parsed_keys["field"]) # type: ignore
            temp["UUID"] = temp["field"].astype(str) + "_" + temp["cell"].astype(str) + "_" + temp["ID"].astype(str)
        else:
            temp["UUID"] = temp["cell"].astype(str) + "_" + temp["ID"].astype(str)
        test_tracks.append(temp)
    full_df = pd.concat(test_tracks)
    full_df["ID"] = full_df["ID"].astype(int)
    return full_df

def insert_suffix(fpath, suffix):
    name = fpath.name
    name_list = name.split('.')
    name_list[0] = name_list[0] + suffix
    return fpath.parent.joinpath('.'.join(name_list))

def fwf_writer(fname, df):
    # TODO figure out how to automatically create format spec
    with open(fname, "w") as f:
        if isinstance(df, pd.DataFrame):
            for i, data in df.iterrows():
                f.write("".join(["{:>11.3f}    ".format(num) for num in data])+"\n")
        else:
            for data in df:
                f.write("".join(["{:>11.3f}    ".format(num) for num in data])+"\n")

# def read_matched(search_dirs: list[Union[str, Path]]):
    
#     for search_dir in search_dirs:

#     return g_im_list, r_im_list, g_trk_list, r_trk_list

compression_dict = {
    "zip": None,
    "gz": None,
    "bz2": None,
    "xz": None,
}

file_reader = {
    "ome": None,
    "tif": None,
    "tiff": None,
    "czi": None,
    "ims": None,
    "nd2": None
}

def get_file_type(fpath):
    fpath = Path(fpath)
    name = fpath.name.split(".")
    name.pop(0)
    decompression = None
    # Check Compression
    if name[-1] == "zip":
        compressed = "zipped"

    # ome tif
    # ims
    # nd2
    # tif
    # czi





def infer_image_handler(fpath, decompression):
    pass


from imaris_ims_file_reader.ims import ims
import zarr
import tifffile as tif
import czifile as czi
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

def czi_TCYX(fpath):
    return czi.imread(fpath)