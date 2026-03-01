# Titanic Survival Prediction

A complete end-to-end machine learning project on the Titanic dataset, covering
exploratory data analysis, feature engineering, classification modeling, and an
interactive prediction application built with Streamlit.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Project Workflow](#project-workflow)
5. [Models Used](#models-used)
6. [Results Summary](#results-summary)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [How to Run the Notebook](#how-to-run-the-notebook)
10. [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
11. [Future Improvements](#future-improvements)

---

## Project Overview

This project applies the full data science workflow to the Titanic dataset —
one of the most well-known classification problems in machine learning.
Starting from raw data exploration, the project builds two classification models
to predict whether a passenger would have survived the disaster, and deploys
them through an interactive web application where users can enter passenger
details and receive a real-time prediction.

The project is intended as a portfolio piece demonstrating practical skills in
data cleaning, exploratory data analysis, feature engineering, model evaluation,
and deployment.

---

## Problem Statement

On April 15, 1912, the RMS Titanic sank after colliding with an iceberg.
Of the estimated 2,224 passengers and crew aboard, more than 1,500 died.
Historical records show that survival was not random — it was strongly
influenced by factors such as passenger class, sex, age, and ticket fare.

**Objective:** Train a binary classification model to predict whether a given
passenger survived (`1`) or did not survive (`0`) based on their personal and
travel attributes.

---

## Dataset Description

The dataset is the Titanic dataset provided by the `seaborn` library,
sourced originally from Kaggle. It contains **891 passenger records** with
the following features:

| Feature | Type | Description |
|:---|:---|:---|
| `pclass` | Ordinal | Passenger class (1 = First, 2 = Second, 3 = Third) |
| `sex` | Nominal | Sex of the passenger |
| `age` | Continuous | Age in years |
| `sibsp` | Discrete | Number of siblings or spouses aboard |
| `parch` | Discrete | Number of parents or children aboard |
| `fare` | Continuous | Ticket fare in GBP |
| `embarked` | Nominal | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
| `deck` | Ordinal | Deck of the cabin (used to derive `has_cabin`) |
| `survived` | Binary | Target variable — 1 = Survived, 0 = Did Not Survive |

**Class distribution:** 61.6% did not survive, 38.4% survived.

---

## Project Workflow

### 1. Data Cleaning

- `deck` (77.1% missing) was converted to a binary flag `has_cabin`
  (1 if a cabin was recorded, 0 otherwise) and the original column was dropped.
- `age` (19.9% missing) was imputed using group-wise medians grouped by
  `pclass` and `sex`. A binary indicator `age_was_missing` was retained.
- `embarked` (2 missing rows) was filled with the mode (Southampton).
- Columns that could cause data leakage (`alive`, `who`, `adult_male`,
  `embark_town`, `class`) were removed.

### 2. Exploratory Data Analysis

A thorough EDA was conducted covering univariate distributions, bivariate
survival analysis, and a correlation overview. Key findings include:

- **Sex** is the strongest single predictor: female survival rate ~74%
  versus ~19% for males.
- **Passenger class** is the second strongest predictor: first-class survival
  at ~63%, third-class at ~24%.
- **Fare** shows a strong positive association with survival after
  log-transformation.
- **Family size** shows a non-linear (inverted-U) relationship with survival —
  small families (2–4 members) survived at higher rates than solo travellers
  or very large families.
- **Age** has a non-linear relationship; children under 13 had the highest
  survival rate (~58%).

### 3. Feature Engineering

Three new features were derived from the existing data:

| Feature | Description |
|:---|:---|
| `family_size` | `sibsp + parch + 1` — total group size including the passenger |
| `log_fare` | `log(fare + 1)` — corrects the strong right skew in ticket fare |
| `is_child` | Binary flag: 1 if `age < 13`, else 0 |

After encoding (`sex` via LabelEncoder, `embarked` via one-hot encoding with
`drop_first=True`), the final feature set used for modeling is:

```
['pclass', 'sex_enc', 'age', 'log_fare', 'has_cabin', 'age_was_missing',
 'alone_enc', 'is_child', 'family_size', 'emb_Q', 'emb_S']
```

### 4. Modeling

The dataset was split into training (80%) and test (20%) sets using
`train_test_split` with `stratify=y` to preserve the class distribution.

- **Training set:** 712 samples
- **Test set:** 179 samples

A `StandardScaler` was fitted on the training set only and applied to both
sets for Logistic Regression. The Decision Tree does not require scaling.

### 5. Model Comparison

Both models were evaluated on the held-out test set using Accuracy, Precision,
Recall, and F1-Score. Logistic Regression outperformed the Decision Tree
across all metrics and was selected as the primary model.

---

## Models Used

### Logistic Regression

- **Parameters:** `C=1.0`, `penalty='l2'`, `max_iter=1000`, `random_state=42`
- **Scaling:** StandardScaler applied
- **Strengths:** Stable, interpretable coefficients, low variance
- **Key finding:** `sex_enc` carries the largest negative coefficient,
  confirming sex as the dominant predictor

### Decision Tree

- **Parameters:** `max_depth=5`, `random_state=42`
- **Scaling:** None required
- **Strengths:** Captures non-linear interactions, visually interpretable
- **Key finding:** First split is on `sex_enc`; subsequent splits use
  `pclass` and `log_fare`

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | **0.7989** | **0.7463** | 0.7246 | **0.7353** |
| Decision Tree | 0.7654 | 0.6849 | 0.7246 | 0.7042 |

**Train vs. Test Gap (Overfitting Check):**

| Model | Train Accuracy | Test Accuracy | Gap |
|:---|:---:|:---:|:---:|
| Logistic Regression | 0.8104 | 0.7989 | 0.0115 |
| Decision Tree | 0.8539 | 0.7654 | 0.0886 |

Logistic Regression shows a very small train-test gap (1.2%), indicating
strong generalisation with no meaningful overfitting. The Decision Tree has a
larger gap (8.9%), which is expected for an unconstrained tree-based model.

**Conclusion:** Logistic Regression is the recommended model for deployment.
It achieves higher test accuracy, superior F1-Score, and significantly better
generalisation.

---

## Project Structure

```
Titanic-Survival-Prediction/
│
├── Mohamed_Tamer_Titanic.ipynb   # Main analysis notebook
├── app.py                        # Streamlit prediction application
│
├── logistic_model.pkl            # Saved Logistic Regression model
├── decision_tree_model.pkl       # Saved Decision Tree model
├── titanic_scaler.pkl            # Saved StandardScaler
├── titanic_features.pkl          # Saved ordered feature list
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Installation

**Prerequisites:** Python 3.8 or higher.

Clone the repository and install all dependencies:

```bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
```

**requirements.txt:**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

---

## How to Run the Notebook

1. Ensure all dependencies are installed (see [Installation](#installation)).
2. Open the notebook:

```bash
jupyter notebook Mohamed_Tamer_Titanic.ipynb
```

3. Run all cells in order from top to bottom.
   The notebook will load the Titanic dataset directly from `seaborn`,
   perform the full analysis, and re-save the model artifacts.

> **Note:** The notebook requires an active internet connection to download
> the Titanic dataset via `seaborn.load_dataset('titanic')` on first run.

---

## How to Run the Streamlit App

1. Ensure the four `.pkl` files are in the same directory as `app.py`.
2. Run the application:

```bash
streamlit run app.py
```

3. The app will open in your browser at `http://localhost:8501`.

**App features:**

- Select between Logistic Regression and Decision Tree from a dropdown.
- Enter passenger details (class, sex, age, fare, port, family members).
- Click **Predict** to receive a survival prediction with confidence
  probabilities and a passenger profile summary.
- Expand the raw feature vector to inspect the exact input sent to the model.

---

## Future Improvements

- **Hyperparameter tuning:** Apply `GridSearchCV` with `StratifiedKFold`
  cross-validation to optimise `C` for Logistic Regression and `max_depth`,
  `min_samples_leaf` for the Decision Tree.

- **Interaction feature:** Create a `sex_pclass` interaction term to explicitly
  capture the combined effect of sex and class, which the EDA shows is the
  most predictive combination in the data.

- **Additional models:** Evaluate Random Forest and Gradient Boosting
  classifiers, which are better suited to capture non-linear interactions
  without requiring extensive feature engineering.

- **Fare per person:** Engineer `fare / family_size` as a more accurate
  proxy for individual wealth than raw fare.

- **Expanded deployment:** Deploy the Streamlit app to Streamlit Community
  Cloud for public access without requiring a local installation.

---

*Dataset source: Seaborn built-in Titanic dataset (originally from Kaggle).*
*Project completed as part of a personal data science portfolio.*
