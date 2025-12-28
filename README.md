# Method Survival Prediction Framework

This repository contains scripts to analyze and predict whether a software method will survive or be deleted. The study is divided into two main Research Questions (RQ):
1.  **RQ1:** Statistical analysis of features (Mann-Whitney U test, Cohen's d, and Violin plots).
2.  **RQ2:** Prediction using Random Forest (RF) baselines and Large Language Models (LLM).

## Dataset Description

* **`pre_features.csv`**: Contains extracted software metrics/features used for statistical analysis (RQ1) and Random Forest prediction (RQ2).
* **`method.csv`**: Contains the raw source code of the methods used for LLM prediction (RQ2).

---

## Configuration

Before running the scripts, you must set up your environment variables to access the OpenAI API and GitHub API (if raw data fetching is needed).

1.  Create a `.env` file in the root directory.
2.  Add your OpenAI API Key and GitHub Personal Access Tokens.

**`.env` file content:**

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
TOKENA=ghp_xxxxxxxxxxxxxxxxxxxxxxxx
TOKENB=ghp_xxxxxxxxxxxxxxxxxxxxxxxx
TOKENC=ghp_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Usage Workflow

Follow the steps below to reproduce the analysis and predictions.

### RQ1: Statistical Analysis
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

### RQ2: Prediction Models

This section compares the performance of a traditional machine learning model (Random Forest) against a Large Language Model (`gpt-5.2`).

#### 1. Random Forest Prediction
Perform predictions using the Random Forest (RF) model based on `pre_features.csv`. This script outputs the accuracy of the RF model and compares it with a random baseline.

```bash
python rf_tuned.py
```

#### 2. LLM Prediction
This process involves generating predictions using the LLM and then evaluating the accuracy on a balanced dataset.

**Step 2a: Generate Predictions**
Use `gpt-5.2` to predict method survival based on the source code in `method.csv`.

```bash
python predict_method_llm.py
```
* **Input:** `method.csv`
* **Output:** `predict.csv` (Contains the raw predictions from the LLM)

**Step 2b: Evaluate LLM Accuracy**
Calculate the classification accuracy using the generated predictions. This script balances the data before evaluation to ensure fair metrics.

```bash
python llm_acc.py
```
* **Input:** `predict.csv`
* **Output:** Accuracy metrics for the LLM.