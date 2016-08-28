# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pyAudioAnalysis import audioFeatureExtraction

__all__ = [
    'FeatureExtractor'
]


class FeatureExtractor:
    """
    Derive features
    """

    def __init__(self, label=None):
        if label is None:
            self.label = ''
        else:
            self.label = label

    @classmethod
    def features(cls, audiodata, samplerate, window_size, step):
        """
        Extract features using audioFeatureExtraction of pyAudioAnalysis package

        For each short-term window a set of features is extracted.

        :param audiodata: the input signal samples
        :param samplerate: the sampling freq (in Hz)
        :param window_size: the short-term window size (in samples)
        :param step: the short-term window step (in samples)
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        return audioFeatureExtraction.stFeatureExtraction(audiodata, samplerate, window_size, step)

    def avg_features(self, features):
        """
        Average short-term window features to get global features of the whole signal

        :param features: a numpy array (numOfFeatures x numOfShortTermWindows)
        :return: dataframe (numOfFeatures)
        """

        col = ['zcr', 'energy', 'en_entropy', 'sp_centroid', 'sp_spread', 'sp_entropy', 'sp_flux', 'sp_rolloff',
               'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11',
               'MFCC12', 'MFCC13', 'chroma1', 'chroma2', 'chroma3', 'chroma4', 'chroma5', 'chroma6', 'chroma7',
               'chroma8', 'chroma9', 'chroma10', 'chroma11', 'chroma12', 'chroma_dev']

        features_avg = features.mean(axis=1)

        features_matrix = np.matrix(features_avg)

        features_df = pd.DataFrame(features_matrix, columns=col)

        features_df['label'] = self.label

        return features_df
