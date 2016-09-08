# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pyAudioAnalysis import audioFeatureExtraction

__all__ = [
    'FeatureEngineer'
]


class FeatureEngineer:
    """
    Derive features
    """

    def __init__(self, audio_data, sample_rate, window_size, step):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step = step

    def avg_features(self):
        """
        Average short-term window features to get global features of the whole signal

        1st step: features: a numpy array (numOfFeatures x numOfShortTermWindows)
        2nd step: avg_features over

        :return: dataframe (numOfFeatures)
        """

        features_df = audioFeatureExtraction.stFeatureExtraction(self.audio_data, self.sample_rate,
                                                                 self.window_size, self.step)

        # col = ['zcr', 'energy', 'en_entropy', 'sp_centroid', 'sp_spread', 'sp_entropy', 'sp_flux', 'sp_rolloff',
        #        'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11',
        #        'MFCC12', 'MFCC13', 'chroma1', 'chroma2', 'chroma3', 'chroma4', 'chroma5', 'chroma6', 'chroma7',
        #        'chroma8', 'chroma9', 'chroma10', 'chroma11', 'chroma12', 'chroma_dev']

        features_avg = features_df.mean(axis=1)

        # features_matrix = np.matrix(features_avg)
        #
        # features_final_df = pd.DataFrame(features_matrix, columns=col)
        #
        return features_avg.reshape(1, -1)
