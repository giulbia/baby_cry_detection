import argparse
import os
import pandas as pd
import pickle

from compute.model.make_prediction import BabyCryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/NewData/")
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
    # MAKE PREDICTION
    ####################################################################################################################

    new_signal = pd.read_csv(os.path.join(load_path_data, ''))
    with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
        model = pickle.load(fp)

    predictor = BabyCryPredictor(model)
    prediction = predictor.classify(new_signal)

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save performances
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(prediction))

if __name__ == '__main__':
    main()
