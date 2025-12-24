## Dataset Description
The file `pre_features.csv` contains extracted features for both surviving methods and deleted methods.

## Usage Workflow

### 1. Random Forest & Random Baseline Prediction
Run the following command to perform predictions using Random Forest (RF) and a random baseline:

```bash
python rf_tuned.py
```

### 2. Dataset Preparation for LLM
Execute the following scripts sequentially to generate the dataset required for Large Language Model (LLM) predictions:

```bash
python getmethod.py
python clean_extractedmethod.py
```

### 3. LLM Prediction
Finally, run the script below to obtain the prediction results using the LLM:

```bash
python predict_method_llm.py
```