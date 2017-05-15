# Baby cry detection
Recognition of baby cry from audio signal

The aim is to automatically recognize a baby crying while sleeping. In such case, a lullaby is started to calm the baby
down.

This is done by implementing a machine learning algorithm on a Raspberry Pi.

The code is organised according to following schema:

+ **TRAINING**

  It includes all the steps required to train a machine learning model. First, it reads the data, it performs feature
  engineering and in trains the model. The model is saved to be used in the prediction step. The _training_ is performed
  on a powerful machine, such as a personal computer.

+ **PREDICTION**

  It includes all the steps needed to make a prediction on a new signal. It reads a new signal (9 second long), it cuts
  it into 5 signals (5 second long), it applies the pipeline save from the training step to make a prediction. The
  _prediction_ step is performed on a Raspberry Pi 2B.

To make it run properly, clone this repo in a folder. In the same parent folder you should also create the following
tree structure:
* PARENT FOLDER
  * baby_cry_detection
  * output
    * dataset
    * model

Script `train_set.py` saves the trainset in folder _dataset_ and, script `train_model.py` saves the model in _model._
Folders _dataset_ and _model_ are parameters with default values as shown above, they can be changed as wished.

>Part of the data used for training comes from
[ESC-50: Dataset for environmental sound classification](https://github.com/karoldvl/ESC-50)