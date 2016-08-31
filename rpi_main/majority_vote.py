import argparse
import os

from rpi_methods.majority_voter import MajorityVoter


def main():

    # /!\ ADAPT PATHS /!\
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_parent_predicitions',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Prediction/")
    parser.add_argument('--save_path',
                        default="/Users/giuliabianchi/Documents/Xebia/XebiCon16/ESC-10/Scripts/Output/Prediction/")

    # Arguments
    args = parser.parse_args()
    load_path_parent_predicitions = args.load_path_parent_predicitions
    save_path = args.save_path

    ####################################################################################################################
    # READ PREDICTIONS
    ####################################################################################################################

    # list load_path sub-folders
    directory_list = os.listdir(load_path_parent_predicitions)

    predictions = list()

    for prediction_folder in directory_list:

        with open(os.path.join(load_path_parent_predicitions, prediction_folder, 'prediction.txt'), 'r') as f:
            read_data = f.read()

        predictions.append(read_data)

    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save majority vote result
    with open(os.path.join(save_path, 'majority_vote_prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(majority_vote))

if __name__ == '__main__':
    main()

