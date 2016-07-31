# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd

from compute import Reader, FeatureExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default="path/to/directory/")
    parser.add_argument('--save_path',
                        default="/path/to/directory")
    parser.add_argument('--label',
                        default=None)

    # Arguments
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path
    label = args.label

    ####################################################################################################################
    # READ FILES IN FOLDER load_path
    ####################################################################################################################

    file_list = os.listdir(load_path)
    feature_extractor = FeatureExtractor(label=label)
    concat_features = pd.DataFrame()

    # for label in label:
    for audio_file in file_list:
        # load features data
        data, samplerate = Reader.read_audio_file('/'.join([load_path, audio_file]))
        all_features = feature_extractor.features(audiodata=data, samplerate=samplerate,
                                                  window_size=0.05*samplerate, step=0.025*samplerate)
        avg_features = feature_extractor.avg_features(all_features)

        concat_features = pd.concat([concat_features, avg_features]).reset_index(drop=True)

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save DataFrame


if __name__ == '__main__':
    main()
