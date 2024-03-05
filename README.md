# Stock Price Prediction with LSTM Recurrent Neural Network
This project aims to predict stock prices using a Long Short-Term Memory (LSTM) Recurrent Neural Network implemented with the TensorFlow framework. The model utilizes historical stock price data obtained from Yahoo Finance.

# Overview
The project consists of two main Python files:

forecastTrain.py: This script showcases the training process of the LSTM model. It requires a mandatory -d argument, which is the path to the CSV file containing historical stock prices. Each entry in this file should follow the format: STOCK_NAME stock_price_day_1, stock_price_day_2, ..., stock_price_day_n. Note that the model is trained on 80% of each stock's price data.

forecast.py: This script demonstrates the usage of the trained model for making predictions. It requires a mandatory -d argument, which is the path to the CSV file containing historical stock prices, following the same format as described above. Additionally, it requires a mandatory -n argument, indicating how many stocks will be made predictions for. Lastly, it optionally accepts a -m bool argument, which determines whether to use dedicated models trained for specific stocks during the prediction process.

# Training the Model
python forecastTrain.py -d path/to/stock_prices.csv

# Making Predictions
python forecast.py -d path/to/stock_prices.csv -n num_stocks -m true\false

# Requirements
- Python 3.x
- TensorFlow
- pandas
- numpy

# Acknowledgments
This project was developed for the "Algorithms in Software Development" course at the Department of Informatics and Telecommunications of the National and Kapodistrian University of Athens.
