import btrack
from btrack import utils as btu
import numpy as np
import pandas as pd
import scipy.ndimage as sim
"""
use btrack to track nuclei
Author(s): Ike H Zhang

References: HiTIPS, Btrack
"""

from btrack.constants import BayesianUpdates
def batch_label(mask: np.ndarray):
    return np.stack([sim.label(mask[i, ...])[0] for i in range(mask.shape[0])]) # type: ignore

class btrack_model:
    objects = None

    def __init__(self, config):
        self.config = config

    def get_objects(self, mask, img=None):
        self.objects = btu.segmentation_to_objects(
            mask, intensity_image=img, properties=('area','orientation', 'axis_major_length', 'axis_minor_length', 'centroid', 'eccentricity'), # type: ignore
            assign_class_ID=True,
        )
        return self.objects

    def track_nuclei(self, objects):
        with btrack.BayesianTracker() as tracker:
            tracker.configure_from_file(self.config)

            tracker.max_search_radius = 12
            # tracker.volume=((0, 1200), (0, 1200), (-1e5, 64.))

            tracker.append(self.objects)
            tracker.track(step_size=100)
            tracker.optimize()

            tracks = tracker.tracks
            tracks = [track.to_dict() for track in tracks]
            tracks = pd.DataFrame(tracks)
        return tracks
    
    def get_tracks(self, mask, img):
        self.get_objects(mask.astype(np.int64), img)
        self.objects = prune_objs(self.objects)
        tracks = self.track_nuclei(self.objects)
        return tracks

def prune_objs(obj_list):
    new_list = []
    for obj in obj_list:
        if obj.properties["area"] > 1:
            new_list.append(obj)
    return new_list

def prune_tracks(tracks, min_t=100, max_area_std= 200):
    new_tracks = tracks[tracks["t"].apply(len) >= min_t]
    new_tracks = new_tracks[new_tracks["area"].apply(np.std) < max_area_std]
    return new_tracks