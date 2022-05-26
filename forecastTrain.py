import os
import random
import sys
import numpy
import tensorflow
from Utils.ArgumentParser import ArgumentParser
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from Utils.Curve import Curve
from Utils.Parser import parse


# Utility function to ensure the reproducibility of the results
# The function's functionality is described in the following link :
# - https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def experimentParameters():
    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    numpy.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)
    tensorflow.config.threading.set_intra_op_parallelism_threads(1)


if __name__ == '__main__':

    experimentParameters()

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)

    if not argumentParser.parse(sys.argv):
        exit(1)

    path = argumentParser.getArgument("-d")

    curves = parse(path)

    features = 1
    timesteps = 10
    batchSize = 64
    epochs = 10
    validationSplit = 0.25
    trainLength = int(0.8 * len(curves[0]))


    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, features)))
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(features))
    model.compile(optimizer='adam', loss='mse')


    xTrain, yTrain = Curve.sampleSet(curves, timesteps, length=trainLength, front=True, includeY=True, normalise=True)
    xTrain = numpy.reshape(xTrain, (xTrain.shape[0], timesteps, features))
    fitSummary = model.fit(xTrain, yTrain, batch_size=batchSize, epochs=epochs, verbose=1, validation_split=validationSplit)
