# Stock Price Predictor

This project aims to predict stock prices using machine learning techniques and advances models. By analyzing historical stock data, the models forecast future stock prices to assist investors in making better decisions.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Results](#results)

## Project Overview

The Stock Price Predictor utilizes historical stock data from 380 different companies from 2013 to 2017 to train machine and deep learning models that forecast future stock prices. The project is structured as follows:

- **data/**: Contains historical stock data used for training and testing.
- **src/**: Includes the source code for data processing and model buidling, including SVR (Support Vector Regression), Random Forest, and LSTM (Long-Short Term Memory).
- **result/**: Stores the graph output of each stock for each of the models.
- **evaluation.py**: The main script responsible for iterating over each stock, calling the models and evaluating their performances.

## Data

The `data/` directory contains CSV files with historical stock prices. Each file includes:

- Date
- Open price
- High price
- Low price
- Close price (Target Column)
- Volume
- Name
- Return
- Rolling_mean
- Rolling_std

## Results

### R2 Score Results:
- LSTM: 0.9866
- Random Forest: 0.9976
- SVR: 0.9851

### Mean-Squared-Error Results:
- LSTM: 9.1808
- Random Forest: 0.9591
- SVR: 3.0306
