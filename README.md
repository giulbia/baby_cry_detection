# Baby cry detection - Building the model
### Recognition of baby cry from audio signals

The aim is to automatically recognize a baby crying while sleeping. In such case, a lullaby is played to calm the baby
down.

This is done by implementing a machine learning algorithm on a Raspberry Pi. The idea is to train a model on a computer
and to deploy it on Raspberry Pi, which is used to record a signal and use the model to predict if it is a baby cry or
not. In the former case a lullaby is played, in the latter the process (recording and predicting steps) starts again.

### Code organisation

The code is organised as follows.

- `./baby_cry_detection/pc_main` and `./baby_cry_detection/pc_methods` folders: to run on a computer, they implement the training part
- `./baby_cry_detection/rpi_main` and `./baby_cry_detection/rpi_methods` folders: to run on a Raspberry Pi, they implement the predicting part


##### TRAINING

It includes all the steps required to train a machine learning model. First, it reads the data, it performs feature
engineering and it trains the model.

The model is saved to be used in the prediction step. The _training step_ is performed
on a powerful machine, such as a personal computer.

Code to run this part is included in `pc_main` and `pc_methods`.

##### PREDICTION

It includes all the steps needed to make a prediction on a new signal. It reads a new signal (9 second long), it cuts
it into 5 overlapping signals (5 second long), it applies the pipeline saved from the training step to make a
prediction.

The _prediction_ step is performed on a Raspberry Pi 2B. Please check
[baby_cry_rpi](https://github.com/giulbia/baby_cry_rpi.git) for deployment on Raspberry Pi.

Code to run this part is included in `rpi_main` and `rpi_methods`.

##### SIMULATION

There is a script to test the prediction step on your computer before deployment on Raspberry Pi.

A script `prediction_simulation.py` and 2 audio signals are provided in folder `./baby_cry_detection/prediction_simulation`.

### Run

To make it run properly, clone this repo in a folder. In the same parent folder you should also create the following
tree structure:
* PARENT FOLDER
  * baby_cry_detection *this cloned repo*
  * output
    * dataset
    * model
    * prediction
  * recording

From your command line go to baby_cry_detection folder and run the following python scripts.

##### TRAINING

This step allows you to train the model. Please note that the model itself is not provided.

```
# Create and save trainset
python baby_cry_detection/pc_main/train_set.py
```
```
# Train and save model
python baby_cry_detection/pc_main/train_model.py
```

Script `train_set.py` saves the trainset in folder _dataset_ and, script `train_model.py` saves the model in folder
 _model_. Folders _dataset_ and _model_ are parameters with default values that fits with the organisation shown
 above, they can be changed as wished.

##### PREDICTION

This step is to be executed on Raspberry Pi. Please refer to [baby_cry_rpi](https://github.com/giulbia/baby_cry_rpi.git)

##### SIMULATION

This step allows you to test the model on your computer. It uses scripts from `rpi_methods` folder.

```
python baby_cry_detection/prediction_simulation/prediction_simulation.py
```

### Logs

Log files are created for each step, they are saved in folder `baby_cry_detection`.




>Part of the data used for training comes from
[ESC-50: Dataset for environmental sound classification](https://github.com/karoldvl/ESC-50)