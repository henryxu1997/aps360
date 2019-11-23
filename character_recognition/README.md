# Character Recognition
Convolutional neural network to recognize individual characters given an image of that character.

## Setup
Download data from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
Unzip into `data/` folder. There should be 62 directories (0-9, A-Z, a-z).
Run `rename_data_dirs.py` to rename the folders to the actual classes.

## Data
All images must be 128 x 128.

## Models
The best models have been saved to `models/` directory.
They can be loaded and tested using functions in `train.py`.
