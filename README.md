# CSC2555 Final Project: Fairness Through Unawareness in Credit Default Classification

## Setup

1. Clone this repository or download the project files.

2. The dataset used in this project is `UCI_Credit_Card.csv`.  
   Make sure the dataset and notebook are in the same directory.

## Usage Instructions

The notebook is designed to run in Google Colab.

1. Upload `CSC2555_Project_Taiwan_Credit_Final.ipynb` to Google Colab.
2. Upload `UCI_Credit_Card.csv` to the Colab working directory.
3. Run all notebook cells in order.

## What the Notebook Does

1. Loads and preprocesses the UCI credit card default dataset.
2. Trains Logistic Regression models with and without SEX and AGE.
3. Evaluates predictive performance and group fairness metrics.
4. Performs proxy-strength experiments using Logistic Regression and Random Forest.
5. Performs classification threshold analysis.
6. Compares fairness results for SEX and AGE.
