# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import logging
import re
import timeit

from pc_methods import Reader
from pc_methods.feature_engineer import FeatureEngineer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='%s/../data' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--save_path',
                        default='%s/../../output/dataset/' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--log_path',
                        default='%s/../' % os.path.dirname(os.path.abspath(__file__)))

    # Arguments
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path
    log_path = args.log_path

    ####################################################################################################################
    # Set up logging
    ####################################################################################################################

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_pc_methods_feat_eng.log'),
                        filemode='w',
                        level=logging.INFO)

    ####################################################################################################################
    # READ FILES IN SUB-FOLDERS of load_path and FEATURE ENGINEERING
    ####################################################################################################################

    # list load_path sub-folders
    regex = re.compile(r'^[0-9]')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

    # initialize empty data frame for results
    concat_features = pd.DataFrame()

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
            avg_features = feature_engineer.feature_engineer(audio_data=data)

            concat_features = pd.concat([concat_features, avg_features]).reset_index(drop=True)

    stop = timeit.default_timer()
    logging.info('Time taken for reading files and feature engineering: {0}'.format(stop - start))

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    logging.info('Saving training set...')

    # Save DataFrame
    concat_features.to_csv(os.path.join(save_path, 'dataset.csv'), index=False)

    logging.info('Saved! {0}'.format(os.path.join(save_path, 'dataset.csv')))


if __name__ == '__main__':
    main()
