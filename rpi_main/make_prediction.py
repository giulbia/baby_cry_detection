# -*- coding: utf-8 -*-

import argparse
import os
import pickle

from rpi_methods import Reader
from rpi_methods.baby_cry_predictor import BabyCryPredictor
from rpi_methods.feature_extractor import FeatureExtractor


def main():
    # /!\ ADAPT PATHS /!\
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/external_input/")
    parser.add_argument('--load_path_model',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Model/")
    parser.add_argument('--save_path',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Prediction/")

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
    new_signal, sample_rate = file_reader.read_audio_file()

    ####################################################################################################################
    # PRE-PROCESSING
    ####################################################################################################################

    # /!\ signal pre-processing to implement: should be a class in the __init__ or in dedicated script
    # * 1 channel signal.
    # * sampling to 44100 Hz
    # * cut the signal (5 seconds)
    # DONE BY USING LIBROSA TO READ FILES. MP3 ACCEPTED, WAV FOR SURE

    ####################################################################################################################
    # EXTRACT FEATURES
    ####################################################################################################################

    # Feature extraction
    feature_extractor = FeatureExtractor(audio_data=new_signal, sample_rate=sample_rate,
                                         window_size=0.05*sample_rate, step=0.025*sample_rate)
    features_df = feature_extractor.avg_features()

    ####################################################################################################################
    # MAKE PREDICTION
    ####################################################################################################################

    with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
        model = pickle.load(fp)

    predictor = BabyCryPredictor(model)
    prediction = predictor.classify(features_df)

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(prediction))

if __name__ == '__main__':
    main()
