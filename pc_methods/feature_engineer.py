# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from librosa.feature import zero_crossing_rate, rmse, mfcc, spectral_centroid, spectral_rolloff, chroma_cens

__all__ = [
    'FeatureEngineer'
]


class FeatureEngineer:
    """
    Feature engineering
    """

    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples

    # Features' names
    COL = ['zcr', 'rms_energy',
           'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
           'mfcc12', 'mfcc13',
           'sp_centroid', 'sp_rolloff',
           'chroma1', 'chroma2', 'chroma3', 'chroma4', 'chroma5', 'chroma6', 'chroma7',
           'chroma8', 'chroma9', 'chroma10', 'chroma11', 'chroma12']

    def __init__(self, label=None):
        if label is None:
            self.label = ''
        else:
            self.label = label

    def feature_engineer(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged [median].
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        zcr_feat = zero_crossing_rate(y=audio_data, hop_length=self.FRAME)

        rmse_feat = rmse(y=audio_data, hop_length=self.FRAME)

        mfcc_feat = mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)

        spectral_centroid_feat = spectral_centroid(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        spectral_rolloff_feat = spectral_rolloff(y=audio_data, sr=self.RATE, hop_length=self.FRAME, roll_percent=0.90)

        chroma_cens_feat = chroma_cens(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        concat_feat = np.concatenate((zcr_feat,
                                      rmse_feat,
                                      mfcc_feat,
                                      spectral_centroid_feat,
                                      spectral_rolloff_feat,
                                      chroma_cens_feat), axis=0)

        median_feat = np.median(concat_feat, axis=1, keepdims=True).transpose()

        features_df = pd.DataFrame(data=median_feat, columns=self.COL, index=None)

        features_df['label'] = self.label

        return features_df
