# -*- coding: utf-8 -*-

import argparse
import os
import librosa
import numpy as np
import logging
import timeit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='%s/../../external_input' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--save_path',
                        default='%s/../data/301 - Crying baby/' % os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--file_name', default=None)
    parser.add_argument('--log_path',
                        default='%s/../' % os.path.dirname(os.path.abspath(__file__)))

    # Arguments
    args = parser.parse_args()
    load_path = args.load_path
    file_name = args.file_name
    save_path = args.save_path
    log_path = args.log_path

    ####################################################################################################################
    # Set up logging
    ####################################################################################################################

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_pc_methods_expand_trainset.log'),
                        filemode='w',
                        level=logging.INFO)

    ####################################################################################################################
    # READ RAW SIGNAL
    ####################################################################################################################

    logging.info('Reading {0}'.format(file_name))
    start = timeit.default_timer()

    # Read full signal
    signal, sr = librosa.load(os.path.join(load_path, file_name), sr=44100, mono=True)

    stop = timeit.default_timer()
    logging.info('Time taken for reading file: {0}'.format(stop - start))

    ####################################################################################################################
    # COMPUTE NUMBER OF 5 sec LENGTH CHUNKS
    ####################################################################################################################

    # Duration
    y = np.floor(librosa.get_duration(signal, sr=sr))

    # nb of chunks = duration - 5 sec + 1
    x = int(y - 5 + 1)

    logging.info('Total duration: {0} sec. Nb of 5 sec chunks: {1}.'.format(y, x))

    ####################################################################################################################
    # SAVE CHUNKS
    ####################################################################################################################

    logging.info('Saving chunks...')
    start = timeit.default_timer()

    # Read and save chunks
    for i in range(x):
        chunk, sr = librosa.load(os.path.join(load_path, file_name), sr=44100, mono=True, offset=i, duration=5.0)
        librosa.output.write_wav(os.path.join(save_path, '{0}_{1}.wav'.format(file_name, str(i))), chunk, sr)

    stop = timeit.default_timer()
    logging.info('Time taken for reading and saving chunks: {0}'.format(stop - start))
    logging.info('Saved! {0}'.format(save_path))


if __name__ == '__main__':
    main()
