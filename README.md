# Purchase-Value-Predictor-MLP-Project

## 📌 Problem Statement
The **Purchase Value Predictor** is a machine learning project designed to estimate a customer's potential purchase value based on their multi-session behavior across digital touchpoints. By analyzing anonymized user interactions (browser types, traffic sources, device details, and geo-location), this model helps marketing teams identify high-value users and optimize engagement strategies.

## 🎯 Objective
To build a regression model that predicts `purchaseValue` using the R-squared ($R^2$) metric for evaluation.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, Category Encoders, Seaborn/Matplotlib.
- **Techniques:** Exploratory Data Analysis (EDA), Feature Engineering (Target Encoding), Model Pipelining, Hyperparameter Tuning.

## 🧠 Approach & Methodology

### 1. Data Cleaning
- Dropped columns with more than **70% missing values**
- Removed high-cardinality identifier columns (`userId`, `sessionId`)
- Filled missing `purchaseValue` with **0**

### 2. Feature Engineering & Preprocessing
- Separated numerical and categorical features
- Numerical features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - One-hot encoding
- Used `ColumnTransformer` for clean preprocessing

### 3. Modeling
- Tried multiple baseline models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Final model:
  - **XGBoost Regressor**
  - Integrated using **Scikit-learn Pipeline**
  - Hyperparameter tuning using **RandomizedSearchCV**

### 4. Evaluation
- Metric used: **R² Score**
- Cross-validation to ensure robustness
- Final model achieved improved generalization compared to baselines

---

## 📊 Key Learnings
- Importance of handling sparse and skewed targets
- Effectiveness of two-stage and ensemble-based approaches
- Clean pipelines prevent data leakage
- Hyperparameter tuning significantly impacts performance

---

## 📁 Files
- `Purchase_Value_Predictor.ipynb` – Complete solution with EDA, preprocessing, modeling & evaluation

---

## 🔗 Author
**Ayush Aryamaan Puhan**  
### Ayush Aryamaan Puhan 

B.Tech + M.Tech (Dual Degree), Mining Engineering - National Institute of Technology, Rourkela 

BS Data Science and Applications - Indian Institute of Technology, Madras
