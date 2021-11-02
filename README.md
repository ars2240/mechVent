# lurie_mechanical_ventilation
This is the repo for the Fall 2020 Lurie Mechanical Ventilation Capstone Project. Our report contains more information about the problem structure and approach. 

## Laurie branch

In the laurie branch (corresponds to laurie directory on Prado and on PTUJ) are files related to fitting LSTM models:
- With resampling
- With oversampline
- With and without static predictors
- With Keras instead of PyTorch 

The clean_data_laurie.py, lamk_data_prep.py, and lamk_nnfuncs.py are used for data preparation. 

The files with LSTM in the name and model_keras.py are the actual modeling scripts. They are designed to start from raw (or imputed) data CSV files; run the data formatting and train/val/test splitting; and then fit a model with the specified architecture. 

## Jieda branch

The Jieda branch contains code related to SMOTE oversampling and LSTM architecture search with shmoo.

## Irene branch

The Irene branch contains code related to incorprating static predictors into the LSTM architecture.
