import pandas as pd
import numpy as np
"""
Collection of quality control scripts for tracks,
Collision detection: drop tracks that overlap at any point in time
Max Displacement: drop tracks that jump too far within one frame
Min Duration: drop tracks that are too short

TODO: 
- add proportional/temporal threshold for collisions, e.g. tracks bound
to other tracks for less than x number of frames of y% of track duration
should be retained

- Cross channel: check if spots exists on other channels
- HMM fitting: Check if HMM fit is too improbable
- Motion QC: Check if track's motion makes sense

Author(s): Ike H Zhang

References: HiTIPS
"""
def find_outliers(img_array, xy=(2, 3), threshold=2, mode="lower"):
    mean_int = np.mean(img_array, axis=xy)
    t = np.arange(mean_int.shape[0])
    p_fit = np.polyfit(t, mean_int, 3)
    poly = [np.poly1d(p_fit[:, i])(t) for i in range(mean_int.shape[1])]
    mean_int -= np.stack(poly).transpose()
    stdev = np.std(mean_int, axis=0)
    if mode=="lower":
        outliers = mean_int < (-threshold * stdev)
    elif mode=="upper":
        outliers = mean_int > (threshold * stdev)
    elif mode=="absolute":
        outliers = np.abs(mean_int) > (threshold * stdev)
    outlier_mask = np.sum(outliers, axis=1) > 0
    keep_mask = np.logical_not(outlier_mask)
    drop_idx = np.where(outlier_mask)[0]
    keep_idx = np.where(keep_mask)[0]
    return drop_idx, keep_idx, outlier_mask, keep_mask

def time_lapse_qc(img, mode="lower"):
    _, keep, _, _ = find_outliers(img, mode="lower")
    return img[keep, ...]

def df_range(x: pd.Series):
    return x.max() - x.min()

def square_displacement(x, y):
    dx2 = (x.diff()) ** 2
    dy2 = (y.diff()) ** 2
    dr2 = dx2 - dy2 
    return dr2

def get_max_displacement(track):
    new_track = track.sort_values("t").copy()
    new_track["dr2"] = square_displacement(new_track["x"], new_track["y"])
    return new_track["dr2"].max()

def get_qc_metrics(track):
    max_disp2 = get_max_displacement(track).max()
    t_range = df_range(track["t"])
    return pd.Series({"max_disp2": max_disp2, "t_range": t_range})

def track_qc(tracks, min_T = 100, max_disp=100, min_sep = 10):
    """
    Main function in QC

    filter out tracks that don't meet criteria, remove tracks that
    are too short, jump around, or overlap
    tracks : pd.DataFrame - dataframe with all tracks
    min_T: int (TODO allow for seconds instead of n_frames) - minimum duration
    max_T: int - maximum tracked duration
    max_disp: int - maximum displacement from frame to frame
    
    """
    disp_thresh = max_disp ** 2
    new_tracks = tracks.copy()
    qc_metrics = new_tracks.groupby("UUID")[["t", "x", "y"]].apply(get_qc_metrics, include_groups=False)
    drop_ids = qc_metrics.index[(qc_metrics["max_disp2"] > disp_thresh) | (qc_metrics["t_range"] < min_T)]
    new_tracks = new_tracks[~new_tracks["UUID"].isin(drop_ids)].copy()
    drop_ids = prune_collisions(new_tracks, min_sep)
    new_tracks = new_tracks[~new_tracks["UUID"].isin(drop_ids)].copy()
    return new_tracks

def check_dist_get_ind(x: np.ndarray, thresh: float):
    dx = np.diff(x)
    idxs = np.where(dx < thresh)[0]
    return idxs

def compare_channel(x: np.ndarray):
    dx = np.diff(x)
    idxs = np.where(dx != 0)[0]
    return idxs

def prune_collisions(tracks: pd.DataFrame, threshold: float):
    """
    Sweep and prune algo:

    for every frame:
        sort particles by x coordinate
        find distance between particles as a vec
        for every index i corresponding to distance less than a threshold:
            check y_distance between y_i and y_i+1
            if y_distance < threshold:
                drop particle i+1
    May be faster with a C impl, since actual SAP algo doesn't need to 
    subtract x coords and performs comparisons, but diff should be 
    O(n) time, except assumes a maximum of 2 collisions
    Parameters:
        tracks - DataFrame with all tracks for one cell
        threshold - distance between particles that's considered a collision
    
    TODO: In theory, could miss a few collisions if y is different with a 
    particle that is inbetween another two in the x,
    """
    drop_ids = []
    new_tracks = tracks.reset_index() # lmao say goodbye to readability right here
    # Iterates frame by frame
    for t_frame, group in new_tracks.groupby(["t"]):
        # drop ids already marked to be dropped to reduce wasted checks
        group = group[~group["UUID"].isin(drop_ids)].copy()
        group.sort_values("x", inplace=True)
        # get indices of x-colliding particles
        idxs = check_dist_get_ind(group["x"].to_numpy(), threshold)
        for idx in idxs: # Potential vectorization w/ masks instead of iter
            y_dist = np.abs(group["y"].iloc[idx] - group["y"].iloc[idx+1])
            if y_dist < threshold:
                drop_ids.append(group["UUID"].iloc[idx+1])
    return drop_ids

def collision_lifetime(track1: pd.DataFrame, track2: pd.DataFrame, thresh: float):
    start_time = max((track1["t"].min(), track2["t"].min()))
    end_time = min((track1["t"].max(), track2["t"].max()))
    track1 = track1[(track1["t"] >= start_time) * (track1["t"] <= end_time)]
    track2 = track2[(track2["t"] >= start_time) * (track2["t"] <= end_time)]
    dx = track1["x"] - track2["x"]
    dy = track1["y"] - track2["y"]
    collided = (dx**2 + dy**2) < thresh ** 2
    if collided.sum() == 0:
        #print("No collisions found")
        return np.nan, np.nan, np.nan
    return collided.sum(), np.min(np.where(collided)), np.max(np.where(collided))

def naive_match_collisions(tracks1, tracks2, 
                           threshold:float = 10, min_lifetime: float = 0):
    matched_ids = []
    for tid1, group1 in tracks1.groupby(["UUID"]):
        for tid2, group2 in tracks2.groupby(["UUID"]):
            if "_".join(tid1[0].split("_")[:-1]) == "_".join(tid2[0].split("_")[:-1]):
                lt, start, end = collision_lifetime(group1, group2, 
                                                    thresh=threshold)
                if lt > min_lifetime:
                    matched_ids.append((tid1 + tid2))
    return matched_ids


def match_collisions(tracks1, tracks2, threshold: float):
    """
    Same SAP algo as prune_collisions, but now we want to keep spots that have collided across channels

    TODO Is there an efficient way of doing this across channels? 
    naively across channels, going pairwise, O(n^2), since each ch1 point 
    needs to check x collision with every point in ch2. If ch1 tracks and ch2
    tracks are concatenated, then its... idk

    For each frame, check ch_id is different (can also use np.diff for this!) 
    check x-collision, check y collision, add to data structure

    could use nxnxt matrix, kind of like adjacency matrix, but memory heavy and very bad
    basically a graph that evolves through time, we want to check the lifetime of each edge

    algo: hash ID pairs using cantor pairing function, add to adjacency list?, 
    then check lifetime of each pair by counting 

    Oh wait... there's should be a maximum of 10 spots per cell, so adjacency 
    matrix is sufficiently small (100 * ~1000 * 8 bytes, 10 kB)
    
    """
    tracks1["ch_id"] = 1
    tracks2["ch_id"] = 2
    tracks = pd.concat((tracks1, tracks2))
    adj_mat = np.zeros((max(tracks1["t"].max(), tracks2["t"].max()),
                        tracks1["UUID"].astype(int).max()+1, 
                        tracks2["UUID"].astype(int).max()+1))
    new_tracks = tracks.reset_index() # lmao say goodbye to readability right here
    # Iterates frame by frame
    for t_frame, group in new_tracks.groupby(["t"]):
        # drop ids already marked to be dropped to reduce wasted checks
        group.sort_values("x", inplace=True)
        diff_mask = np.diff(group["ch_id"].to_numpy()) != 0
        diff_dist = np.diff(group["x"].to_numpy()) < threshold
        idxs = np.where(diff_mask * diff_dist)[0]
        for idx in idxs: # Potential vectorization w/ masks instead of iter
            y_dist = np.abs(group["y"].iloc[idx] - group["y"].iloc[idx+1])
            if y_dist < threshold:
                adj_mat[t_frame, int(group["UUID"].iloc[idx]), int(group["UUID"].iloc[idx])]
    matched_ids = []
    return matched_ids


    