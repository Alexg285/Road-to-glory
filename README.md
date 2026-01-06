# Predicting Team Performance in FIFA World Cups

## Research Question
Can national teams’ progression in FIFA World Cups be predicted using pre-tournament characteristics and group-stage performance data?

## Set Up

# Create environment
conda env create -f environment.yml
conda activate worldcup-project
## Usage
python main.py
Running the main script prints model evaluation results to the console, including:

- Training and test set sizes
- Class distribution in the test set
- Accuracy and Macro F1-score for each model
- Detailed classification reports
- Confusion matrices (raw counts and row-normalized percentages)
- Identification of the best-performing model on the test set

# Project Structure
Road-to-glory/
├── data/
│   ├── raw/                # Original data sources
│   └── clean/              # Cleaned and processed dataset
│       └── final_dataset1.csv
│
├── notebooks/
│   ├── cleaning_data.ipynb
│   └── EDA_Results_Road_to_Glory.ipynb
│
├── src/
│   ├── config.py           # Feature sets and constants
│   ├── data_loader.py      # Data loading and splitting logic
│   ├── models.py           # Model training functions
│   └── evaluation.py       # Evaluation utilities
│
├── results/
│   ├── metrics.csv
│   └── plots/
│
├── main.py                 # End-to-end training and evaluation script
├── environment.yml
└── README.md

## Results

Four models were evaluated on the test set: Logistic Regression and XGBoost, each with a baseline
and an enriched feature set.

The best overall performance is achieved by the XGBoost baseline model, which uses group-stage
performance features only. It reaches an accuracy of 0.641 and a Macro F1-score of 0.449, clearly
outperforming all other configurations.

Logistic Regression provides a strong interpretable baseline but struggles with intermediate and
advanced tournament stages. In contrast, XGBoost substantially improves performance on Quarter-final, Semi-final, and Final outcomes, which explains the large increase in Macro F1-score.

Adding pre-tournament features (strength, host status, average age) does not improve performance
and in some cases degrades it, likely due to increased noise and limited sample size. This result
is consistent with correlation analysis, which shows that group-stage performance variables dominate the predictive signal.

Performance on the Winner class remains limited due to the very small number of observations in the
test set, which is a structural limitation of the dataset.

## Requirements
- Python 3.11
- scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost, pathlib


