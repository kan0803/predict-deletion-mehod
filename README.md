# Method Survival Prediction Framework

This repository contains scripts to analyze and predict whether a software method will survive or be deleted. The study is divided into two main Research Questions (RQ):

1.  **RQ1:** Statistical analysis of features (Mann-Whitney U test, Cohen's d, and Violin plots).
2.  **RQ2:** Prediction comparison using Random Forest (RF), a Random Baseline, and Large Language Models (LLM).

## Dataset Description

### Raw Data
* **`pre_features.csv`**: Contains extracted software metrics/features used for statistical analysis (RQ1) and Random Forest prediction (RQ2).
* **`method.csv`**: Contains the raw source code of the methods used for LLM prediction (RQ2).

### Generated Data
* **`/dataset` directory**: Created by `make_dataset.py`. Contains 10-fold stratified splits of the data (e.g., `pre_features_0.csv`, `method_0.csv`) used for Cross-Validation in RQ2.

---

## Configuration

Before running the scripts (especially for RQ2), you must set up your environment variables to access the OpenAI API.

1.  Create a `.env` file in the root directory.
2.  Add your OpenAI API Key.

**`.env` file content:**

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Usage Workflow

Follow the steps below to reproduce the analysis and predictions.

### 1. RQ1: Statistical Analysis
Analyze the features in `pre_features.csv` to understand the differences between surviving and deleted methods.

Run the following command:

```bash
python rq1.py
```

**Outputs:**
* Calculates **Mann-Whitney U test** and **Cohen's d** statistics for each feature.
* Generates the following visualizations:
    * **`RQ1 violine.pdf`**: Violin plots for top 3 features.
    * **`RQ1 violine_all.pdf`**: Violin plots for all features.

### 2. Dataset Preparation (Pre-requisite for RQ2)
Before running the prediction experiments, you must generate the stratified 10-fold dataset. This script aligns `pre_features.csv` and `method.csv`, then splits them ensuring the ratio of deleted/survived methods is consistent.

Run the following command:

```bash
python make_dataset.py
```

**Outputs:**
* Creates a `dataset/` directory.
* Generates `pre_features_*.csv` and `method_*.csv` (files 0 to 9).

### 3. RQ2: Prediction Models
This section performs a comparative analysis of three approaches:
1.  **Random Forest (RF):** Trained on 9 folds (with undersampling to balance classes) and tested on 1 fold.
2.  **Random Baseline:** A uniform random predictor (50/50 probability).
3.  **GPT-5.2:** Predicts survival based on source code using the OpenAI API.

Run the following command:

```bash
python rq2.py
```

**What this script does:**
* Loads the 10-fold datasets from the `dataset/` directory.
* **RF Experiment:** Performs 10-fold Cross-Validation. Training data is undersampled to handle class imbalance, while test data remains original.
* **Random Experiment:** Evaluates a Uniform Dummy Classifier across all folds.
* **GPT Experiment:** Calls the OpenAI API for each fold (results are cached in `gpt_results/` to avoid redundant calls).
* **Evaluation:** Outputs **Accuracy**, **AUC**, **F1**, **Recall**, and **Precision**.
    * *Note:* Metrics are calculated targeting the **Deleted (0)** class as the positive class.

**Outputs:**
* Console output comparing the average metrics of RF, Random, and GPT-5.2.
* `gpt_results/`: Directory containing cached prediction results from the LLM.