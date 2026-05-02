# Bachelor_Thesis_Stock_Market_Predictions
Abstract
This thesis investigates the prediction of binary directional movement (up/down) of stock market prices using financial news and historical stock prices. It compares traditional machine learning models against advanced deep neural networks, and frequency-based Natural Language Processing against Context-Aware Transformers. The research utilizes the FNSPID dataset, containing over 15 million financial news articles spanning the 2006–2023 period, alongside historical price data for five technology-sector stocks: Apple, Amazon, Google, Nvidia, and Microsoft. TF-IDF and FinBERT are evaluated for text preprocessing, utilizing Principal Component Analysis (PCA) for dimensionality reduction. For the predictive modeling, Logistic Regression, Random Forests, Multi-Layer Perceptron (MLP), and Long Short-Term Memory (LSTM) networks are compared based on their accuracy in forecasting next-day directional stock price movements. While the MLP and LSTM architectures achieved higher validation accuracies, they ultimately overfit the training data. When forecasting the binary up/down movement of the subsequent trading day, a Logistic Regression model achieved the highest individual testing accuracy at 53.30%, and TF-IDF proved to be the most efficient preprocessing method. When averaging the top five testing configurations for each architecture, all four models converged around a 51% test accuracy rate. This performance parity–and the models' convergence just above a random 50% baseline–confirms the robustness of traditional statistical models and reinforces the efficient market hypothesis within noisy financial environments.

News-Based Stock Prediction - Thesis Code
This repository contains the code developed for the bachelor thesis comparing traditional machine learning and artificial neural networks for stock prediction.

1. System Requirements
Operating System: Linux (Ubuntu 22.04 or similar recommended)

Hardware:

Minimum 32GB RAM (due to large dataset processing).

NVIDIA GPU with CUDA support (Code was tested on RTX 3070 / RTX 3080).

Software: Micromamba (or Conda/Mamba) package manager.

2. Environment Setup
Due to dependency conflicts between cuML (used for traditional ML models) and PyTorch (used for deep neural networks), the pipeline is split into two separate environments.

You can recreate these environments exactly as they were used during development via the provided .yml files:

Environment 1: PyTorch (For MLP & LSTM)

Bash
micromamba env create -f env_pytorch.yml
Environment 2: cuML (For Logistic Regression & Random Forest)

Bash
micromamba env create -f env_cuml.yml
3. How to Run the Code