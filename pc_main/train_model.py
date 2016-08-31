import argparse
import json
import os
import pickle

import pandas as pd

from pc_methods.train_classifier import TrainClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Dataset/")
    parser.add_argument('--save_path',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Model/")

    # Arguments
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path

    ####################################################################################################################
    # TRAIN MODEL
    ####################################################################################################################

    train_set = pd.read_csv(os.path.join(load_path, 'dataset.csv'))

    train_classifier = TrainClassifier()
    performance, parameters, best_estimator = train_classifier.train(train_set)

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save performances
    with open(os.path.join(save_path, 'performance.json'), 'w') as fp:
        json.dump(performance, fp)

    # Save parameters
    with open(os.path.join(save_path, 'parameters.json'), 'w') as fp:
        json.dump(parameters, fp)

    # Save model
    with open(os.path.join(save_path, 'model.pkl'), 'wb') as fp:
        pickle.dump(best_estimator, fp)

if __name__ == '__main__':
    main()
