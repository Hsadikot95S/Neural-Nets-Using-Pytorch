# CS672 – Introduction to Deep Learning
## Fall 2023 - Project #2

This repository contains the materials for Project #2 of the CS672 course - Intro to Deep Learning, taught by Prof. Sarbanes in the Fall of 2023.

### Project Overview

In this project, we aim to tackle two fundamental tasks in Machine Learning and Deep Learning:

- Perform Exploratory Data Analysis (EDA) on a dataset regarding quantitative measures of diabetes progression.
- Develop a Machine Learning / Deep Learning Model for Regression Analysis using PyTorch.

### Dataset

The dataset is based on data from 442 diabetes patients and includes a set of features used to predict disease progression. The data is accessible from the `scikit-learn` library and details can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

### Installation

Before beginning, ensure that the PyTorch libraries are installed. Follow the instructions on the official [PyTorch Get Started page](https://pytorch.org/get-started/locally/).

### Tasks

#### A) Exploratory Data Analysis (EDA)

- Prep the data for model input, handle missing values, and identify outliers.
- Transform data to ensure all entries are numeric.
- Perform an EDA using both classic (Pandas, NumPy) and PyTorch-based tools (if available).
- Identify feature dependencies and correlations, and list the features of importance for the target label.

#### B) Deep Learning Model

- Develop models using PyTorch:
    1. MLP (Multi-Layer Perceptron)
    2. Linear Regression
    3. DNN (Deep Neural Network with at least 2 hidden layers)
- Divide the dataset into training (80%) and validation (20%).
- Utilize MSE and MAE as loss functions and SGD, Adam, RMSProp as optimizers with various learning rates.
- Train the models for 100 epochs and plot training vs validation loss.
- Select and present the best model based on performance metrics.

### Results

Results of the analysis, including EDA visualizations and performance plots, are available in the Jupyter notebook included in this repository.

### Usage

To run the Jupyter notebook:
1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook or JupyterLab installed.
3. Navigate to the repository directory and run `jupyter notebook`.
4. Open the notebook `deep_learning_project_2.ipynb`.
