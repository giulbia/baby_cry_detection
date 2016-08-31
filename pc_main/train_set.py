# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import re

from pc_methods import Reader
from pc_methods.feature_extractor import FeatureExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/")
    parser.add_argument('--save_path',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Dataset/")

    # Arguments
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path

    ####################################################################################################################
    # READ FILES IN SUB-FOLDERS of load_path
    ####################################################################################################################

    # list load_path sub-folders
    regex = re.compile(r'^[0-9]')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

    # initialize empty data frame for results
    concat_features = pd.DataFrame()

    # iteration on sub-folders
    for directory in directory_list:
        # Instantiate FeatureExtractor
        feature_extractor = FeatureExtractor(label=directory)

        file_list = os.listdir(os.path.join(load_path, directory))

        # iteration on audio files in each sub-folder
        for audio_file in file_list:
            file_reader = Reader(os.path.join(load_path, directory, audio_file))
            data, samplerate = file_reader.read_audio_file()
            all_features = feature_extractor.features(audiodata=data, samplerate=samplerate,
                                                      window_size=0.05*samplerate, step=0.025*samplerate)
            avg_features = feature_extractor.avg_features(all_features)

            concat_features = pd.concat([concat_features, avg_features]).reset_index(drop=True)

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save DataFrame
    concat_features.to_csv(os.path.join(save_path, 'dataset.csv'), index=False)


if __name__ == '__main__':
    main()
