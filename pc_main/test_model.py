# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import logging
import timeit

from pc_methods import Reader
from rpi_methods.feature_engineer import FeatureEngineer
from rpi_methods.baby_cry_predictor import BabyCryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default='%s/../../external_input' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--load_path_model',
                        default='%s/../../output/model/' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--save_path',
                        default='%s/../../output/prediction/' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--file_name', default='V_2017-04-01+08_04_36=0_13.mp3')
    parser.add_argument('--log_path',
                        default='%s/../' % os.path.dirname(os.path.abspath(__file__)))

    # Arguments
    args = parser.parse_args()
    load_path_data = args.load_path_data
    load_path_model = args.load_path_model
    file_name = args.file_name
    save_path = args.save_path
    log_path = args.log_path

    ####################################################################################################################
    # Set up logging
    ####################################################################################################################

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_pc_methods_test_model.log'),
                        filemode='w',
                        level=logging.INFO)

    ####################################################################################################################
    # READ RAW SIGNAL
    ####################################################################################################################

    logging.info('Reading {0}'.format(file_name))
    start = timeit.default_timer()

    # Read signal (first 5 sec)
    file_reader = Reader(os.path.join(load_path_data, file_name))
    signal, _ = file_reader.read_audio_file()

    stop = timeit.default_timer()
    logging.info('Time taken for reading file: {0}'.format(stop - start))

    ####################################################################################################################
    # FEATURE ENGINEERING
    ####################################################################################################################

    logging.info('Starting feature engineering')
    start = timeit.default_timer()

    # Feature extraction
    engineer = FeatureEngineer()

    processed_signal = engineer.feature_engineer(signal)
    # processed_signal.drop('label', axis=1, inplace=True)

    stop = timeit.default_timer()
    logging.info('Time taken for feature engineering: {0}'.format(stop - start))

    ####################################################################################################################
    # MAKE PREDICTION
    ####################################################################################################################

    logging.info('Predicting...')
    start = timeit.default_timer()

    with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
        model = pickle.load(fp)

    predictor = BabyCryPredictor(model)

    prediction = predictor.classify(processed_signal)

    stop = timeit.default_timer()
    logging.info('Time taken for prediction: {0}. Is it a baby cry?? {1}'.format(stop - start, prediction))

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    logging.info('Saving prediction...')

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(prediction))

    logging.info('Saved! {0}'.format(os.path.join(save_path, 'prediction.txt')))


if __name__ == '__main__':
    main()
