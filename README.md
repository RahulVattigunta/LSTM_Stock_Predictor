# LSTM Stock Price Prediction

## Overview
This project implements a **Long Short-Term Memory (LSTM) model** to predict stock prices using historical data. The model is built using **TensorFlow/Keras** and trained on stock data fetched from **Yahoo Finance**.

## Features
- Fetches historical stock data (Yahoo Finance API)
- Preprocesses and normalizes stock prices
- Splits data into **training and testing** sets
- Trains an LSTM-based **neural network model**
- Predicts future stock prices
- Visualizes **actual vs. predicted** prices

## Installation
To set up the project on your system, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/lstm-stock-prediction.git
cd lstm-stock-prediction
````
### **2. Create a Virtual Environment**
````bash
python3 -m venv tensorflow_env
source tensorflow_env/bin/activate
````
### **3. Install Dependencies**
````bash
pip install -r requirements.txt
````
## Usage
Run the following command to train the LSTM model and generate predictions:
````bash
python lstm_stock.py
````

## Requirements
The project requires the following Python libraries:
````bash
numpy
pandas
matplotlib
tensorflow
scikit-learn
yfinance
````
Alternatively, install them using:
````bash
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance
````

## Dataset
The stock data is fetched from Yahoo Finance using the yfinance package. You can modify the ticker symbol in the script to predict any stock.

## Visualization
The model outputs a graph comparing actual vs predicted stock prices.

## License
This project is licensed under the MIT License.

## Author
Rahul Vattigunta
