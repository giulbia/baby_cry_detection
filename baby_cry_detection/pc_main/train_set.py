# -*- coding: utf-8 -*-

import argparse
import logging
import os
import re
import timeit
import numpy as np

from baby_cry_detection.pc_methods import Reader
from baby_cry_detection.pc_methods.feature_engineer import FeatureEngineer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../../data'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../../output/dataset/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--log_path',
                        default='{}/../../'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_pc_methods_feat_eng.log'),
                        filemode='w',
                        level=logging.INFO)

    # READ FILES IN SUB-FOLDERS of load_path and FEATURE ENGINEERING

    # list load_path sub-folders
    regex = re.compile(r'^[0-9]')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

    # initialize empty array for features
    X = np.empty([1, 18])

    # initialise empty array for labels
    y = []

    logging.info('Creating training set...')
    start = timeit.default_timer()

    # iteration on sub-folders
    for directory in directory_list:

        # Instantiate FeatureEngineer
        feature_engineer = FeatureEngineer(label=directory)

        file_list = os.listdir(os.path.join(load_path, directory))

        # iteration on audio files in each sub-folder
        for audio_file in file_list:
            file_reader = Reader(os.path.join(load_path, directory, audio_file))
            data, sample_rate = file_reader.read_audio_file()
            avg_features, label = feature_engineer.feature_engineer(audio_data=data)

            X = np.concatenate((X, avg_features), axis=0)
            y.append(label)

    # X.shape is (401, 18) as I'm not using indexing. First line is made of zeros and is to be removed
    X = X[1:, :]

    stop = timeit.default_timer()
    logging.info('Time taken for reading files and feature engineering: {0}'.format(stop - start))

    # Save to numpy binary format
    logging.info('Saving training set...')
    np.save(os.path.join(save_path, 'dataset.npy'), X)
    np.save(os.path.join(save_path, 'labels.npy'), y)

    logging.info('Saved! {0}'.format(os.path.join(save_path, 'dataset.npy')))
    logging.info('Saved! {0}'.format(os.path.join(save_path, 'labels.npy')))


if __name__ == '__main__':
    main()
