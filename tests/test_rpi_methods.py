from __future__ import division
import numpy as np

from tests import TestBabyCry
from baby_cry_detection.rpi_methods import Reader
from baby_cry_detection.rpi_methods.feature_engineer import FeatureEngineer
from baby_cry_detection.rpi_methods.baby_cry_predictor import BabyCryPredictor
from baby_cry_detection.rpi_methods.majority_voter import MajorityVoter


class RpiMethodsTest(TestBabyCry):
    """
    Test pc_methods
    """

    def test_read_audio_file(self):

        reader = Reader(file_name=self.file_name)

        tracks = reader.read_audio_file()

        self.assertEqual(len(tracks), 5)

        for track in tracks:
            self.assertEqual(track.size, 5*44100)

    def test_feature_engineer(self):

        feature_engineer = FeatureEngineer()

        features = feature_engineer.feature_engineer(self.rpi_sample)

        self.assertEqual(features.shape, (1, 18))

    def test_compute_librosa_features(self):

        feature_engineer = FeatureEngineer()

        expected_computed_points = int(round(feature_engineer.RATE*5/feature_engineer.FRAME, 0))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feat_name='zero_crossing_rate').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feat_name='rmse').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feat_name='mfcc').shape,
                         (13, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feat_name='spectral_centroid').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feat_name='spectral_rolloff').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feat_name='spectral_bandwidth').shape,
                         (1, expected_computed_points))

    def test_is_baby_cry(self):

        for category in self.categories:
            if category == '301 - Crying baby':
                self.assertEqual(BabyCryPredictor._is_baby_cry(category), 1)
            else:
                self.assertEqual(BabyCryPredictor._is_baby_cry(category), 0)

    def test_vote(self):

        prediction_array = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]
        )

        for i in range(prediction_array.shape[0]):
            prediction_list = prediction_array[i, :]
            voter = MajorityVoter(prediction_list=prediction_list)
            if i in range(3):
                self.assertEqual(voter.vote(), 0)
            else:
                self.assertEqual(voter.vote(), 1)
