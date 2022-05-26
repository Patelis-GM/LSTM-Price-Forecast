import os.path
import random
import sys
import numpy
from matplotlib import pyplot
from tensorflow.python.keras.models import load_model
from Utils.ArgumentParser import ArgumentParser
from Utils.Parser import parse

if __name__ == '__main__':

    argumentParser = ArgumentParser()
    argumentParser.addArgument(argument="-d", type="path", mandatory=True)
    argumentParser.addNumericArgument(argument="-n", type="int", floor=1, ceiling=359, mandatory=True)
    argumentParser.addArgument(argument="-m", type="bool", mandatory=False)

    if not argumentParser.parse(sys.argv):
        exit(1)

    path = argumentParser.getArgument("-d")
    n = argumentParser.getArgument("-n")
    manipulate = argumentParser.getArgument("-m")

    if manipulate is None:
        manipulate = False

    curves = parse(path)

    if len(curves) == 0:
        exit(1)

    # Select at random the indices of the Curves that will be plotted and a prediction will be made for them
    indices = list(range(len(curves)))
    random.shuffle(indices)
    indices = indices[:n]

    timesteps = 10
    features = 1
    modelPath = "Model"
    model = load_model(modelPath)

    predictionPercent = 0.2
    predictionLength = int(0.2 * len(curves[0]))
    subplots = 2

    # In case the -m parameter was provided skew the randomly produced indices
    # so that the indices of the Curves with a dedicated model will be included
    if manipulate:
        for i in range(len(curves)):
            curve = curves[i]
            if (curve.getID() == "leu" or curve.getID() == "aiv" or curve.getID() == "agn") and i not in indices:
                indices.append(i)
        random.shuffle(indices)

    for index in indices:

        if subplots > 1:
            subplots = 1

        curve = curves[index]

        #  In case the corresponding Curve has a dedicated model 2 subplots will be needed
        if os.path.isdir("Dedicated-Models/" + curve.getID()):
            subplots += 1

        figure, axes = pyplot.subplots(subplots)
        xAxis = numpy.arange(len(curve))

        x, y = curve.sample(timesteps=timesteps, length=predictionLength, front=False, includeY=True, normalise=True)
        x = numpy.reshape(x, (x.shape[0], timesteps, features))

        # Model X makes a prediction Y
        # Y gets denormalized
        # Y gets reshaped to an 1D array
        # Calculate the MSE using the model.evaluate() method
        modelPrediction = model.predict(x)
        modelPrediction = curve.denormalise(modelPrediction)
        modelPrediction = modelPrediction[:, 0]
        mse = model.evaluate(x, y, verbose=0)

        if subplots == 1:
            axes.set_title("MSE : {:.2e}".format(mse))
            axes.plot(xAxis, curve.getValues(), label=curve.getID())
            axes.plot(xAxis[len(curve) - predictionLength + timesteps:], modelPrediction, label="Prediction")

        else:
            axes[0].set_title("MSE : {:.2e}".format(mse))
            axes[0].plot(xAxis, curve.getValues(), label=curve.getID())
            axes[0].plot(xAxis[len(curve) - predictionLength + timesteps:], modelPrediction, label="Prediction")
            axes[0].legend()

        if subplots > 1:

            dedicatedModel = load_model("Dedicated-Models/" + curve.getID() + "/Model")

            dedicatedTimesteps = None

            # Each of the following Curves has a dedicated model with a different number of timesteps
            if curve.getID() == "aiv":
                dedicatedTimesteps = 20
            elif curve.getID() == "leu":
                dedicatedTimesteps = 50
            elif curve.getID() == "agn":
                dedicatedTimesteps = 5

            x, y = curve.sample(timesteps=dedicatedTimesteps, length=predictionLength, front=False, includeY=True, normalise=True)
            x = numpy.reshape(x, (x.shape[0], dedicatedTimesteps, features))

            # Dedicated Model X makes a prediction Y
            # Y gets denormalized
            # Y gets reshaped to an 1D array
            # Calculate the MSE using the evaluate() method
            dedicatedModelPrediction = dedicatedModel.predict(x)
            dedicatedModelPrediction = curve.denormalise(dedicatedModelPrediction)
            dedicatedModelPrediction = dedicatedModelPrediction[:, 0]
            dedicatedMse = dedicatedModel.evaluate(x, y, verbose=0)

            axes[1].set_title("MSE : {:.2e}".format(dedicatedMse))
            axes[1].plot(xAxis, curve.getValues(), label=curve.getID())
            axes[1].plot(xAxis[len(curve) - predictionLength + dedicatedTimesteps:], dedicatedModelPrediction, label="Dedicated Prediction")
            axes[1].legend()

        pyplot.legend()
        pyplot.show()
