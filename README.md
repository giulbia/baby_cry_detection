# Baby cry detection - Building the model
Recognition of baby cry from audio signal

The aim is to automatically recognize a baby crying while sleeping. In such case, a lullaby is started to calm the baby
down.

This is done by implementing a machine learning algorithm on a Raspberry Pi. The idea is to train a model on a computer
and to deploy it on Raspberry Pi, which is used to record a signal and use the model to predict if it is a baby cry or
not. In the former case a lullaby is played, in the latter the process (recording and predicting steps) starts again.

### Overall schema

![alt text](/Users/giuliabianchi/Documents/Xebia/XebiCon16/recap/recap.001.png "schema")

### Code organisation

The code is organised as follows.

- `pc_*` folders: to run on a computer, they implement the training part
- `rpi_*` folders: to run on a Raspberry Pi, they implement the predicting part


##### TRAINING

It includes all the steps required to train a machine learning model. First, it reads the data, it performs feature
engineering and in trains the model.

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

##### TEST

There is a script to test the prediction step on your computer before use on Raspberry Pi.

**TODO provide test signal**


### Run TODO

To make it run properly, clone this repo in a folder. In the same parent folder you should also create the following
tree structure:
* PARENT FOLDER
  * baby_cry_detection *this cloned repo*
  * output
    * dataset
    * model
    * prediction
  * recording

Script `train_set.py` saves the trainset in folder _dataset_ and, script `train_model.py` saves the model in _model._
Folders _dataset_ and _model_ are parameters with default values as shown above, they can be changed as wished.



Please note that the model itself is not provided, you should run `train_set.py` and `train_model.py` to generate it.

>Part of the data used for training comes from
[ESC-50: Dataset for environmental sound classification](https://github.com/karoldvl/ESC-50)