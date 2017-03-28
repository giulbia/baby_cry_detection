# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import timeit
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth,\
    chroma_cens, rmse

__all__ = [
    'FeatureEngineer'
]


class FeatureEngineer:
    """
    Feature engineering
    """

    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples

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

        logging.info('Computing zero_crossing_rate...')
        start = timeit.default_timer()

        zcr_feat = zero_crossing_rate(y=audio_data, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing rmse...')
        start = timeit.default_timer()

        rmse_feat = rmse(y=audio_data, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing mfcc...')
        start = timeit.default_timer()

        mfcc_feat = mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral centroid...')
        start = timeit.default_timer()

        spectral_centroid_feat = spectral_centroid(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral rolloff...')
        start = timeit.default_timer()

        spectral_rolloff_feat = spectral_rolloff(y=audio_data, sr=self.RATE, hop_length=self.FRAME, roll_percent=0.90)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral bandwidth...')
        start = timeit.default_timer()

        spectral_bandwidth_feat = spectral_bandwidth(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        # logging.info('Computing chroma cens...')
        # start = timeit.default_timer()
        #
        # # http://stackoverflow.com/questions/41896123/librosa-feature-tonnetz-ends-up-in-typeerror
        # chroma_cens_feat = chroma_cens(y=audio_data, sr=self.RATE, hop_length=self.FRAME)
        #
        # stop = timeit.default_timer()
        # logging.info('Time taken: {0}'.format(stop - start))

        concat_feat = np.concatenate((zcr_feat,
                                      rmse_feat,
                                      mfcc_feat,
                                      spectral_centroid_feat,
                                      spectral_rolloff_feat,
                                      # chroma_cens_feat,
                                      spectral_bandwidth_feat
                                      ), axis=0)

        logging.info('Averaging...')
        start = timeit.default_timer()

        mean_feat = np.mean(concat_feat, axis=1, keepdims=True).transpose()

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        return mean_feat, self.label
