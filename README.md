# News-Based Stock Prediction - Thesis Code

This repository contains the code developed for the bachelor thesis comparing traditional machine learning and artificial neural networks for stock prediction.

## Abstract
This thesis investigates the prediction of binary directional movement (up/down) of stock market prices using financial news and historical stock prices. It compares traditional machine learning models against advanced deep neural networks, and frequency-based Natural Language Processing against Context-Aware Transformers. The research utilizes the FNSPID dataset, containing over 15 million financial news articles spanning the 2006–2023 period, alongside historical price data for five technology-sector stocks: Apple, Amazon, Google, Nvidia, and Microsoft. TF-IDF and FinBERT are evaluated for text preprocessing, utilizing Principal Component Analysis (PCA) for dimensionality reduction. For the predictive modeling, Logistic Regression, Random Forests, Multi-Layer Perceptron (MLP), and Long Short-Term Memory (LSTM) networks are compared based on their accuracy in forecasting next-day directional stock price movements. While the MLP and LSTM architectures achieved higher validation accuracies, they ultimately overfit the training data. When forecasting the binary up/down movement of the subsequent trading day, a Logistic Regression model achieved the highest individual testing accuracy at 53.30%, and TF-IDF proved to be the most efficient preprocessing method. When averaging the top five testing configurations for each architecture, all four models converged around a 51% test accuracy rate. This performance parity–and the models' convergence just above a random 50% baseline–confirms the robustness of traditional statistical models and reinforces the efficient market hypothesis within noisy financial environments.

---

## 1. System Requirements

* **Operating System:** Linux (openSUSE Tumbleweed or similar recommended)
* **Hardware:** 
  * Minimum 32GB RAM (due to large dataset processing).
  * NVIDIA GPU with CUDA support (Code was tested on RTX 3070 / RTX 3080).
* **Software:** Micromamba (or Conda/Mamba) package manager, `tmux` (highly recommended for long-running scripts).

## 2. Environment Setup

Due to dependency conflicts between `cuML` (used for traditional ML models) and `PyTorch` (used for deep neural networks), the pipeline is split into two separate environments. You can recreate these environments exactly as they were used during development via the provided `.yml` files:

**Environment 1: PyTorch (For MLP, LSTM, & NLP Processing)**
```bash
micromamba env create -f env_pytorch.yml
```

**Environment 2: cuML (For Logistic Regression & Random Forest)**
```bash
micromamba env create -f env_cuml.yml
```

---

## 3. Execution Pipeline (How to run the code)

**Important Note on Time Span:** When running the data pipelines, ensure the time span is set to `1.10.2006` to `1.1.2025` to capture all relevant data without missing edge cases.

### Phase 1: Data Preprocessing
*Activate the `env_pytorch` PyTorch environment for this phase.*

1. **`01_preprocessing_pipeline_2.ipynb`**: Run this notebook first to clean and assess the raw FNSPID and stock market data.
2. **`02_TF-IDF_FinBERT_3.ipynb`**: Run this to preprocess the text using TF-IDF.
3. **`finbert_train.py`**: *(Optional alternative to the notebook for FinBERT)* Because FinBERT extraction takes a significant amount of time, it is highly recommended to run this Python script inside a `tmux` session rather than relying on a Jupyter Notebook.

### Phase 2: Model Training
*Because training takes hours, it is highly recommended to execute all of the following scripts inside `tmux` sessions to prevent timeouts.*

**Traditional ML Models (Activate `env_cuml`):**
* Run `lr_run.py` to train the Logistic Regression models.
* Run `rf_run.py` to train the Random Forest models.

**Deep Neural Networks (Activate `env_pytorch`):**
* Run `mlp_run.py` to train the Multi-Layer Perceptron.
* Run `lstm_run.py` to train the Long Short-Term Memory network.

### Phase 3: Model Evaluation
Once training is complete, the models are evaluated and the best hyperparameter configurations are saved into a central `optimal_parameters.json` file.

**Evaluating Traditional Models (Activate `env_cuml`):**
* Run `03_LR_RF_cuML_based_LR_eval.ipynb`
* Run `03_LR_RF_cuML_based_RF_eval.ipynb`

**Evaluating Deep Learning Models (Activate `env_pytorch`):**
* Run `mlp_eval.py` *(Run via `tmux`)*
* Run `lstm_eval.py` *(Run via `tmux`)*

Once these evaluation scripts finish, the full pipeline is complete and the results will be populated in your outputs.