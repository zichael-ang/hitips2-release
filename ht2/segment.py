import cellpose
import cellpose.models
import numpy as np
from tqdm import tqdm
from ht2.utils import reorder_img
"""
Objective: segmentation to wrap different API to achieve 
interop with newer models

main script should only be able to create, pass new 
model args and exec args, and run on images w args
being pulled from param file or user option

interacting w models:
select segmentation channel or channels
preprocess
eval
qc

TODO Testing, adding more models, change image format to TYXC, 

Thinking back, this was dumb: if someone wants to write a new pipeline,
there's a lot of extra bs to write regarding parsing parameters, 
if someone knows what parameters they're going to be using, just detail it
in the pipeline and not by parsing params

I think the problem is that this doesn't make sense with both user-written 
pipelines and user-written modules: I think user-written pipelines make 
way more sense. So i will probably ditch this python module. 

Maybe useful for creating drop-in, plug and play segmentation modules 
that contain all the pre and post processing necessary, 

For real drop in replacement modules, each model should have
its own parameter parser? to decrease dependence on pipeline to 
run reading and parsing for it?

btrack does this by having tracker configure itself from a config file

Author(s): Ike H Zhang

References: HiTIPS, Cellpose
"""

class model:
    def __init__(self, arg1, arg2):
        pass
    def eval(self, arg1, arg2):
        pass

class example_model_wrapper:
    """
    Wrapper for any kind of segmentation, allows extension to different models
    by allowing developers to specify how models are initialized and executed
    Parameters: 
    model - segmentation model, with some set of init args
    m_args - model/init args
    e_args - execution/eval args to be passed to model every call"""
    def __init__(self, 
                 m_args={"arg1":1, "arg2":2}, 
                 e_args={"arg1":1, "arg2":2}):
        self.model_class = model
        self.model_instance = self.model_class(**m_args)
        self.m_args = m_args
        self.e_args = e_args
    def update_m_args(self, m_args):
        # Recreate model instance w new arguments
        self.model_instance = self.model_class(**m_args)
    def transform(self, im):
        # Add any necessary preprocess steps here
        return im
    def eval(self, im):
        # Add any necessary evaluation steps or post processing here
        transformed = self.transform(im)
        mask = self.model_instance.eval(transformed, **self.e_args)
        return mask

class cellpose_model:
    def __init__(self, 
                 m_args={"model_type":"cyto3", 
                         "gpu":False,
                         "device":None},
                 e_args = {"batch_size":8, 
                         "channels":[0, 1],
                         "channel_axis":None,
                         "invert":False,
                         "normalize":True,
                         "diameter":60}):
        self.model_class = cellpose.models.Cellpose
        self.model_instance = self.model_class(**m_args)
        self.m_args = m_args
        self.e_args = e_args
    def update_m_args(self, m_args):
        # Recreate model instance w new arguments
        self.model_instance = self.model_class(**m_args)
    def update_e_args(self, e_args):
        self.e_args = e_args
        
    def transform(self, im):
        # Add any necessary preprocess steps here
        return im
    def eval(self, im, e_args=None):
        # Add any necessary evaluation steps or post processing here
        transformed = self.transform(im)
        if e_args is not None:
            masks, flows, styles, diams = self.model_instance.eval(transformed, **e_args)
        masks, flows, styles, diams = self.model_instance.eval(transformed, **self.e_args)
        return masks
    
    def batch(self, im_stack: np.ndarray, dim_ord="TCYX", e_args=None):
        im_stack_TYXC = reorder_img(im_stack, dim_ord, "TYXC")
        mask_stack = np.zeros(im_stack_TYXC.shape[:-1])
        for i in tqdm(range(im_stack_TYXC.shape[0])):
            if e_args is not None:
                mask_stack[i,...] = self.eval(im_stack_TYXC[i,...], e_args)
            mask_stack[i,...] = self.eval(im_stack_TYXC[i,...], self.e_args)
        return mask_stack

import cellpose
import cellpose.models
import numpy as np
from tqdm import tqdm
from utils import reorder_img


class gpu_model:
    """
    Only works with single channel images for now
    """
    def __init__(self, 
                 m_args={"model_type":"cyto3", 
                         "gpu":True,
                         "device":None},
                 e_args = {"batch_size":16000, 
                         "channels":[0, 0],
                         "channel_axis":None,
                         "invert":False,
                         "normalize":True,
                         "diameter":120}):
        self.model_class = cellpose.models.Cellpose
        self.model_instance = self.model_class(**m_args)
        self.m_args = m_args
        self.e_args = e_args
    def update_m_args(self, m_args):
        # Recreate model instance w new arguments
        self.model_instance = self.model_class(**m_args)
    def update_e_args(self, e_args):
        self.e_args = e_args
    
    # def batch(self, im_stack: np.ndarray, dim_ord="TYX", e_args=None):
    #     if e_args is None:
    #         e_args = self.e_args
    #     im_stack_TYXC = reorder_img(im_stack, dim_ord, "TYX")
    #     # im_stack_TYXC_b = torch.tensor(im_stack_TYXC)
    #     # im_stack_TYXC_b = im_stack_TYXC_b.unsqueeze(1)
    #     # print(im_stack_TYXC_b.shape)
    #     # masks, flows, styles, diams = self.model_instance.eval(im_stack_TYXC_b, e_args)
    #     mask_stack = np.zeros(im_stack_TYXC.shape[:3])
    #     for i in tqdm(range(im_stack_TYXC.shape[0])):
    #         mask_stack[i,...], _, _, _ = self.model_instance.eval(im_stack_TYXC[i,...], e_args)
    #     return mask_stack
    def run(self, im: np.ndarray, e_args=None, N=16, pad=4):
        if e_args is None:
            print("e_args is none")
            e_args = self.e_args
        mask, flow, style, diameter  = self.model_instance.eval(im, **e_args)
        return mask, flow, style, diameter

    def batch(self, im_stack: np.ndarray, dim_ord="TYXC", e_args=None, N=16, pad=4):
        if e_args is None:
            e_args = self.e_args
        im_stack_TYXC = reorder_img(im_stack, dim_ord, "TYXC")
        pad_extra = im_stack_TYXC.shape[0] % 16
        im_stack_TYXC_b = np.pad(im_stack_TYXC, ((0, 16 - (im_stack_TYXC.shape[0] % 16)), (pad,pad), (pad, pad), (0, 0)), mode="constant")
        print(im_stack_TYXC_b.shape)
        pad_shape = im_stack_TYXC_b.shape
        im_stack_TYXC_b = np.reshape(im_stack_TYXC_b, (N, pad_shape[0] * pad_shape[1] // N, pad_shape[2], pad_shape[3]))
        # im_stack_TYXC_b = torch.tensor(im_stack_TYXC)
        # im_stack_TYXC_b = im_stack_TYXC_b.unsqueeze(-1)
        print(im_stack_TYXC_b.shape)
        mask_stack = np.zeros(im_stack_TYXC_b.shape[:-1])
        for i in tqdm(range(im_stack_TYXC_b.shape[0])):
            mask_stack[i,...], _, _, _  = self.model_instance.eval(im_stack_TYXC_b[i, ...], **e_args)
        masks = np.reshape(mask_stack, pad_shape[:-1])
        return masks[:im_stack_TYXC.shape[0], pad:-pad, pad:-pad]

class gpu_model_2ch:
    """
    Only works with single channel images for now
    """
    def __init__(self, 
                 m_args={"model_type":"cyto3", 
                         "gpu":True,
                         "device":None},
                 e_args = {"batch_size":16000, 
                         "channels":[0, 1],
                         "channel_axis":None,
                         "invert":False,
                         "normalize":True,
                         "diameter":120}):
        self.model_class = cellpose.models.Cellpose
        self.model_instance = self.model_class(**m_args)
        self.m_args = m_args
        self.e_args = e_args
    def update_m_args(self, m_args):
        # Recreate model instance w new arguments
        self.model_instance = self.model_class(**m_args)
    def update_e_args(self, e_args):
        self.e_args = e_args
    
    # def batch(self, im_stack: np.ndarray, dim_ord="TYX", e_args=None):
    #     if e_args is None:
    #         e_args = self.e_args
    #     im_stack_TYXC = reorder_img(im_stack, dim_ord, "TYX")
    #     # im_stack_TYXC_b = torch.tensor(im_stack_TYXC)
    #     # im_stack_TYXC_b = im_stack_TYXC_b.unsqueeze(1)
    #     # print(im_stack_TYXC_b.shape)
    #     # masks, flows, styles, diams = self.model_instance.eval(im_stack_TYXC_b, e_args)
    #     mask_stack = np.zeros(im_stack_TYXC.shape[:3])
    #     for i in tqdm(range(im_stack_TYXC.shape[0])):
    #         mask_stack[i,...], _, _, _ = self.model_instance.eval(im_stack_TYXC[i,...], e_args)
    #     return mask_stack
    def run(self, im: np.ndarray, e_args=None, N=16, pad=4):
        if e_args is None:
            print("e_args is none")
            e_args = self.e_args
        mask, flow, style, diameter  = self.model_instance.eval(im, **e_args)
        return mask, flow, style, diameter

    def batch(self, im_stack: np.ndarray, dim_ord="TYXC", e_args=None, N=16, pad=4):
        if e_args is None:
            e_args = self.e_args
        im_stack_TYXC = reorder_img(im_stack, dim_ord, "TYXC")
        pad_extra = im_stack_TYXC.shape[0] % 16
        im_stack_TYXC_b = np.pad(im_stack_TYXC, ((0, 16 - (im_stack_TYXC.shape[0] % 16)), (pad,pad), (pad, pad), (0, 0)), mode="constant")
        print(im_stack_TYXC_b.shape)
        pad_shape = im_stack_TYXC_b.shape
        im_stack_TYXC_b = np.reshape(im_stack_TYXC_b, (N, pad_shape[0] * pad_shape[1] // N, pad_shape[2], pad_shape[3]))
        # im_stack_TYXC_b = torch.tensor(im_stack_TYXC)
        # im_stack_TYXC_b = im_stack_TYXC_b.unsqueeze(-1)
        print(im_stack_TYXC_b.shape)
        mask_stack = np.zeros(im_stack_TYXC_b.shape[:-1])
        for i in tqdm(range(im_stack_TYXC_b.shape[0])):
            mask_stack[i,...], _, _, _  = self.model_instance.eval(im_stack_TYXC_b[i, ...], **e_args)
        masks = np.reshape(mask_stack, pad_shape[:-1])
        return masks[:im_stack_TYXC.shape[0], pad:-pad, pad:-pad]

