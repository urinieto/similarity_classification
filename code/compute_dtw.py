#!/usr/bin/env python
"""
TODO
"""
import argparse
import cPickle as pickle
import dtw
import matplotlib.pyplot as plt
import librosa
import logging
import mir_eval
import numpy as np
import os
import pandas as pd
import scipy
import sklearn
import time

from joblib import Parallel, delayed
import msaf
from msaf import jams2


# Directory to store the features
features_dir = "../features_beats"

# Distances to use for the DTW
dist_dict = {
    "L1": scipy.spatial.distance.cityblock,
    "L2": scipy.spatial.distance.euclidean,
    "correlation": scipy.spatial.distance.correlation
}

# Normalization techniques for the threshold and f-measure computation
norms = ["none", "min", "max", "hmean"]


def compute_threshold(intervals=None, labels=None, scores=None, norm=None):
    """Computes the thresholds for the given inputs.

    Parameters
    ----------
    intervals : np.array
        Estimated segment boundary intervals.
    labels : np.array
        Estimated segment labels.
    scores : np.array
        DTW scores.
    norm : str
        Normalization method.

    Returns
    -------
    thr : float > 0
        Threshold for which to optimally cut the DTW score matrix.
    fmeasure : float > 0
        F-measure resulting using threshold.
    """

    label_agreement = np.zeros((len(labels), len(labels)), dtype=bool)

    for i in range(len(labels)):
        for j in range(i, len(labels)):
            label_agreement[i, j] = (labels[i] == labels[j])
            label_agreement[j, i] = label_agreement[i, j]

    time_norm = 1

    durations = np.diff(intervals, axis=1).ravel()

    if norm == 'min':
        time_norm = np.minimum.outer(durations, durations)

    elif norm == 'max':
        time_norm = np.maximum.outer(durations, durations)

    elif norm == 'hmean':
        time_norm = 2./np.add.outer(durations, durations)
        time_norm *= np.multiply.outer(durations, durations)

    # TODO: have the label agreement index out nan-valued scores

    scores = scores / time_norm

    label_agreement[np.tril_indices_from(label_agreement, k=0)] = False

    label_agreement[~np.isfinite(scores)] = False

    label_disagreement = ~label_agreement

    label_disagreement[np.tril_indices_from(label_disagreement, k=0)] = False

    label_disagreement[~np.isfinite(scores)] = False

    tp_scores = scores[label_agreement]
    fp_scores = scores[label_disagreement]

    num_pos = np.sum(label_agreement)
    num_neg = np.sum(label_disagreement)

    if num_pos == 0 or num_neg == 0:
        return 0.0, 0.0

    y_true = np.concatenate([np.zeros(len(tp_scores)), np.ones(len(fp_scores))])
    y_score = np.concatenate([tp_scores, fp_scores])

    fpr, tpr, thr = sklearn.metrics.roc_curve(y_true, y_score)

    tp = num_pos * tpr
    fp = num_neg * fpr

    precision = tp / (tp + fp)
    recall = tpr

    fmeasure = np.asarray([mir_eval.util.f_measure(p, r)
                           for p, r in zip(precision, recall)])

    k = np.argmax(fmeasure)

    return thr[k], fmeasure[k]


def read_features(features_file):
    """Reads the features from the pickle file.
    Parameters
    ----------
    features_file : str
        Path to the features file.

    Returns
    -------
    cqgram : np.array
        Subseg-sync constant-Q power spectrogram.
    intframes : np.array
        The frame indeces.
    """
    with open(features_file, "r") as f:
        features = pickle.load(f)
    return features["cqgram"], features["intframes"]


def save_features(cqgram, intframes, subseg, features_file):
    """Reads the features from the pickle file.
    Parameters
    ----------
    cqgram : np.array
        Subseg-sync constant-Q power spectrogram.
    intframes : np.array
        The frame indeces.
    subseg : np.array
        Subseq-index times.
    features_file : str
        Path to the output features file.
    """
    features = {}
    features["cqgram"] = cqgram
    features["intframes"] = intframes
    features["subseg"] = subseg
    with open(features_file, "w") as f:
        pickle.dump(features, f, protocol=-1)


def compute_features(audio_file, intervals, level):
    """Computes the subseg-sync cqt features from the given audio file, if
    they are not previously computed. Saves the results in the feat_dir folder.

    Parameters
    ----------
    audio_file : str
        Path to the audio file.
    intervals : np.array
        Intervals containing the estimated boundaries.
    level : str
        Level in the hierarchy.

    Returns
    -------
    cqgram : np.array
        Subseg-sync constant-Q power spectrogram.
    intframes : np.array
        The frame indeces.
    """
    # Check if features have already been computed
    if level == "small_scale":
        features_file = os.path.join(features_dir, os.path.basename(audio_file).split('.')[0] +
                                    "_small_scale.mp3.pk")
    else:
        features_file = os.path.join(features_dir, os.path.basename(audio_file) +
                                    ".pk")
    if os.path.isfile(features_file):
        return read_features(features_file)

    y, sr = librosa.load(audio_file, sr=11025)

    # Default hopsize is 512
    hopsize = 512
    cqgram = librosa.logamplitude(librosa.cqt(y, sr=sr, hop_length=hopsize)**2, ref_power=np.max)

    # Track beats
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr,
                                           hop_length=hopsize)

    # Synchronize
    cqgram = librosa.feature.sync(cqgram, beats, aggregate=np.median)

    intframes = None
    if intervals is not None:
        # convert intervals to frames
        intframes = librosa.time_to_frames(intervals, sr=sr, hop_length=hopsize)

        # Match intervals to subseg points
        intframes = librosa.util.match_events(intframes, beats)

    # Save the features
    save_features(cqgram, intframes, beats, features_file)

    return cqgram, intframes


def make_cost_matrix(audio_file, intervals, labels, dist, level):
    """Computes the cost matrix of the DTW from the given audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file.
    intervals : np.array
        Intervals containing the estimated boundaries.
    labels : np.array
        Estimated segment labels.
    dist : fun
        Distance function to be used for the DTW
    level : str
        Level in the hierarchy.

    Returns
    -------
    D : np.array
        DTW scores.
    P : list
        List containing np.arrays() representing the DTW paths.
    """
    # Computes the features (return existing ones if already computed)
    cqgram, intframes = compute_features(audio_file, intervals, level)

    # Score matrix
    D = np.nan * np.zeros((len(labels), len(labels)), dtype=np.float32)
    np.fill_diagonal(D, 0)

    # Path matrix
    P = []
    for i in range(len(labels)):
        P.append([np.nan] * len(labels))
    for i in range(len(labels)):
        P[i][i] = 0

    for i in range(len(labels)):
        x_slice = cqgram[:, intframes[i, 0]:intframes[i, 1]].T
        if intframes[i, 1] - intframes[i, 0] < 2:
            continue
        for j in range(i+1, len(labels)):
            if intframes[j, 1] - intframes[j, 0] < 2:
                continue
            y_slice = cqgram[:, intframes[j, 0]:intframes[j, 1]].T

            dtw_cost, distance, path = dtw.dtw(x_slice, y_slice, dist=dist)
            D[i, j] = dtw_cost
            D[j, i] = D[i, j]
            path = list(path)
            path[0] = np.asarray(path[0], dtype=np.int32)
            path[1] = np.asarray(path[1], dtype=np.int32)
            P[i][j] = path

    return D, P


def compute_score(file_struct, level, dist_key):
    """Computes the DTW scores for the given file.

    Parameters
    ----------
    file_struct : FileStruct (msaf)
        Object containing the struct.
    level : str
        Level of the hierarchy to be considered.
    dist_key : str
        Distance measure identifier.

    Returns
    -------
    ret : dict
        Dictionary with the results, including the following  keys:
            intervals : reference boundary intervals,
            labels : reference segment labels,
            scores : DTW scores,
            paths : DTW paths,
            thresholds : thresholds found for the different normalizations,
            fmeasures : fmeasures computes for the different normalizations,
            file_name : name of the file
    """
    try:
        ref_inter, ref_labels = jams2.converters.load_jams_range(
            file_struct.ref_file, "sections", annotator=0, context=level)

        assert len(ref_labels) > 0

        D, P = make_cost_matrix(file_struct.audio_file, ref_inter, ref_labels,
                                dist=dist_dict[dist_key], level=level)
        thresholds = {}
        fmeasures = {}
        for norm in norms:
            thresholds[norm], fmeasures[norm] = compute_threshold(
                intervals=ref_inter, labels=ref_labels, scores=D, norm=norm)
    except IndexError as e:
        logging.warning("warning: problem computing threshold %s at level %s" %
                        (file_struct.audio_file, level))
        ref_inter = None
        ref_labels = None
        D = None
        P = None
        thresholds = None
        fmeasures = None
    except (AssertionError, IOError) as e:
        logging.warning("warning: no annotations for %s" %
                        file_struct.audio_file)
        ref_inter = None
        ref_labels = None
        D = None
        P = None
        thresholds = None
        fmeasures = None
    finally:
        cqgram, intframes = compute_features(file_struct.audio_file, None, level)
    ret = {
        "intervals": ref_inter,
        "labels": ref_labels,
        "scores": D,
        "paths": P,
        "thresholds": thresholds,
        "fmeasures": fmeasures,
        "file_name": os.path.basename(file_struct.audio_file)
    }
    return ret


def save_results(dataset, level, dist_key, scores):
    """Saves the results.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    level : str
        Level of dataset being considered.
    dist_key : str
        Type of distance
    scores : dict
        Dictionary containing the scores for all the files in the dataset.
    """
    result = {
        "level": level,
        "dist": dist_key,
        "file_scores": scores
    }
    out_file = "scores_datasetE%s_levelE%s_distE%s.pk" % (dataset, level,
                                                          dist_key)
    with open(out_file, "w") as f:
        pickle.dump(result, f, protocol=-1)


def main(ds_path, n_jobs):
    """Main function to compute DTW scores for a given root dataset and
    number of processors.

    Parameters
    ----------
    ds_path : str
        Path to the root of the dataset.
    n_jobs : int > 0
        Number of processes to use.
    """

    # Datasets from which to compute the DTWs
    datasets = ["SALAMI", "Isophonics"]

    # Different levels for the datasets
    dataset_levels = {
        "Isophonics": ["function"],
        #"SALAMI": ["function", "large_scale", "small_scale"]
        #"SALAMI": ["function", "large_scale"]
        "SALAMI": ["function"]
    }

    # Make sure the features folder exists
    msaf.utils.ensure_dir(features_dir)

    # Main loop
    for dataset in datasets:
        # Obtain all the files for the given dataset
        files = msaf.io.get_dataset_files(ds_path, ds_name=dataset)

        # Compute results for the specific level and distance
        for level in dataset_levels[dataset]:
            for dist_key in dist_dict.keys():
                if dataset != "SALAMI" and level != "function":
                    continue
                logging.info("Computing: %s, %s, %s" %
                             (dataset, level, dist_key))

                # Compute scores using multiple cpus
                scores = Parallel(n_jobs=n_jobs)(delayed(compute_score)(
                    file_struct, level, dist_key)
                    for file_struct in files[:])

                # Save all results
                save_results(dataset, level, dist_key, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the DTW scores, paths, thresholds, and f-measures"
        " for multiple collections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path",
                        action="store",
                        help="Input path to dataset")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=1,
                        type=int,
                        help="The number of threads to use")

    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Call main function
    main(args.ds_path, args.n_jobs)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))
