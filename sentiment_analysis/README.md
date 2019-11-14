# Sentiment Analysis

## Setup
Please install requirements: `pip install -r requirements.txt`

## Directory structure
Data for model building and training found in `data/'
Outputs of model training (csv of accuracy, loss) found in `outputs/'
Saved models found in `models/`

## Development
Models should be defined in `network.py`. The current model is a customizable RNN with various hyperparameters.
The actual training is done by running `python train.py`. Various hyperparameters can be selected during training and by default the training curves and outputs will be saved. Do not commit these unless they show significant results!