# -*- coding: utf-8 -*-

import argparse
import os
import pickle

from rpi_methods import Reader
from rpi_methods.baby_cry_predictor import BabyCryPredictor
from rpi_methods.feature_engineer import FeatureEngineer


def main():
    # /!\ ADAPT PATHS /!\
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default="~/Documents/baby_cry/Data/Recordings")
    parser.add_argument('--load_path_model',
                        default="~/Data/Model/")
    parser.add_argument('--save_path',
                        default="~/Data/Prediction/")

    # Arguments
    args = parser.parse_args()
    load_path_data = args.load_path_data
    load_path_model = args.load_path_model
    save_path = args.save_path

    ####################################################################################################################
    # READ RAW SIGNAL
    ####################################################################################################################

    # Read signal
    file_name = os.listdir(load_path_data)         # [0] /!\ in the real usage there will only be one file in the folder
    file_reader = Reader(os.path.join(load_path_data, file_name))
    play_list = file_reader.read_audio_file()

    ####################################################################################################################
    # iteration
    ####################################################################################################################

    # iterate on play_list for feature engineering and prediction

    ####################################################################################################################
    # EXTRACT FEATURES
    ####################################################################################################################

    # Feature extraction
    feature_extractor = FeatureEngineer()

    # for
    features_df = feature_extractor.avg_features()

    ####################################################################################################################
    # MAKE PREDICTION
    ####################################################################################################################

    with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
        model = pickle.load(fp)

    predictor = BabyCryPredictor(model)
    # for
    prediction = predictor.classify(features_df)

    ##################################
    # MAJORITY VOTE
    ####################

    # call majority_vote

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(prediction))

if __name__ == '__main__':
    main()
