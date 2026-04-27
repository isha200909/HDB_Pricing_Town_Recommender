# W.O.W. — Worth-Or-What?

**Your Singapore HDB Resale Adviser**

A data analytics sprint project by the team at **WOW! Real Estate**, a fictional Singapore property agency. We analyse over 270,000 HDB resale transactions to help home buyers make smarter decisions — from predicting fair resale prices to recommending the best towns for their lifestyle.

---

## Problem Statement

Singapore's HDB resale market is one of the most active public housing markets in the world, yet pricing remains opaque for the average buyer. Resale prices are influenced by a complex mix of factors — flat attributes, location, remaining lease, and proximity to amenities — making it difficult for buyers to gauge whether a listing is fairly priced or which towns best fit their needs and budget.

**How can we leverage data to empower HDB resale buyers with accurate price estimates and personalised town recommendations?**

## Our Solution

We built **W.O.W. (Worth-Or-What?)**, an end-to-end data product that combines exploratory data analysis, machine learning, and a user-friendly web application:

1. **Exploratory Data Analysis** — Uncovered key pricing drivers and spatial patterns across 26 towns and 76 features
2. **Regression Modelling** — Trained and benchmarked multiple ML models to predict resale prices, selecting **LightGBM** as the best-performing regressor
3. **Classification & Clustering** — Built a **LightGBM classifier** on top of **K-Means clusters** to recommend towns based on buyer lifestyle preferences
4. **Web Application** — Delivered insights through an interactive Flask app with two tools:
   - **Worth-O-Meter** — Price estimator with confidence range and price-per-sqft breakdown
   - **What-Town-Next?** — Town recommender with match scoring and town profiles

---

## Team

| Role | Member(s) | Responsibilities |
|------|-----------|------------------|
| **Product Manager** | Lucas | Designed and developed the web application; overall project coordination |
| **EDA Team** | Claire, Ganesh, Stan | Exploratory data analysis — uncovering key insights from the dataset |
| **Modelling Team** | Nuriesya, Din, Manyu | Machine learning — training, evaluating, and selecting the best models for price prediction and town recommendation |

---

## Features

### Worth-O-Meter (Price Estimator)
- LightGBM regression model predicting resale price with a **±10% confidence range**
- Visual **price range gauge** showing where the estimate sits
- **Price-per-sqft** calculation for easy comparison
- All fields optional — missing inputs gracefully fall back to dataset medians
- Liveability index computed from amenity proximity (MRT, hawker, mall, school)

### What-Town-Next? (Town Recommender)
- LightGBM classifier trained on K-Means lifestyle clusters
- Recommends towns with **match scores** based on L2 distance in scaled feature space
- **Town profile cards** showing maturity status and region
- Filter by region, budget, flat size, and amenity preferences

### General
- Light / dark mode toggle
- Responsive layout with pill selectors, typeahead search, and range sliders
- Accessible design with ARIA roles, skip links, and keyboard navigation
- Data attribution footer citing source and model training date

---

## Results & Model Performance

### Price Prediction (Regression)

Five models were trained and evaluated on the HDB resale dataset. **LightGBM** was selected for deployment due to its minimal train-test discrepancy (best generalisation) and fastest runtime, despite XGBoost achieving a marginally higher R².

| Model | Train R² | Test R² | Train–Test Gap | Test RMSE |
|-------|----------|---------|----------------|-----------|
| XGBoost | 0.9777 | 0.9711 | 0.0066 | $29,672 |
| Random Forest | 0.9719 | 0.9605 | 0.0115 | $34,669 |
| CatBoost | 0.9540 | 0.9529 | 0.0011 | $37,846 |
| **LightGBM** ✅ | **0.9525** | **0.9513** | **0.0013** | **$38,508** |
| Linear Regression | 0.7897 | 0.7881 | 0.0016 | $80,287 |

> **Why LightGBM over XGBoost?** While XGBoost scored highest on test R², its larger train-test gap (0.0066 vs 0.0013) indicates slight overfitting. LightGBM offers a better balance of accuracy, generalisation, and speed — critical for a production web app.

### Town Recommendation (Classification)

Five classifiers were trained to predict K-Means lifestyle clusters. **LightGBM** was again selected — within 0.01 of the best test accuracy while being 4× faster than Random Forest.

| Model | Train Accuracy | Test Accuracy | Generalization Gap | Runtime (s) |
|-------|----------------|---------------|--------------------|-------------|
| Random Forest | 1.0000 | 0.9998 | 0.0002 | 44.75 |
| **LightGBM** ✅ | **0.9889** | **0.9971** | **−0.0082** | **11.23** |
| XGBoost | 0.9676 | 0.9764 | −0.0088 | 15.94 |
| CatBoost | 0.9344 | 0.9355 | −0.0011 | 27.95 |
| Logistic Regression | 0.7123 | 0.6887 | 0.0236 | 3.42 |

> **Selection criteria:** Among models within 0.01 of the best test accuracy, pick the one with the lowest runtime. LightGBM's 99.71% accuracy at 11s runtime made it the clear choice.

---

## Dataset

`data.csv` — HDB resale transaction data (**270,620 rows, 76 columns**) sourced from [data.gov.sg](https://data.gov.sg). Tracked via Git LFS.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:** Flask, NumPy, joblib, scikit-learn, pandas, LightGBM (+ XGBoost, matplotlib, seaborn for notebooks)

### 2. Run the notebooks to export model artefacts

The app requires trained model files. Run the notebooks in order:

1. **`notebooks/01_exploratory_data_analysis.ipynb`** — EDA and feature engineering
2. **`notebooks/02A_Regression_Models.ipynb`** — Trains and exports the regression model → `app/models/lgbm_regressor.joblib`
3. **`notebooks/02B_classification_model.ipynb`** — Trains and exports the classifier → `app/models/lgbm_classifier.joblib`

### 3. Start the app

Open https://hdb-predictor-recommender-qiej.onrender.com/recommender

Note: It will take a few minutes to load

---

## Project Structure

```
├── app/
│   ├── app.py                  # Flask routes & model inference
│   ├── models/                 # Exported model artefacts (.joblib + .json)
│   ├── static/
│   │   ├── style.css           # Full stylesheet (light/dark, components)
│   │   └── wow-logo.png        # Brand logo
│   └── templates/
│       ├── base.html           # Shared layout (hero, nav, footer)
│       ├── index.html          # Worth-O-Meter (price estimator)
│       └── recommender.html    # What-Town-Next? (town recommender)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02A_Regression_Models.ipynb
│   └── 02B_classification_model.ipynb
├── data.csv                    # Source data (Git LFS)
└── requirements.txt
```
