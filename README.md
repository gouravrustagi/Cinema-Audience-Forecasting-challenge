# 🎬 Cinema Audience Forecasting Challenge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Time%20Series-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **Predicting daily theatre audience counts across multiple locations using advanced machine learning and statistical modeling techniques.**

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Installation & Setup](#installation--setup)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Results](#results)
- [Key Insights](#key-insights)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

This project tackles the challenge of **forecasting daily cinema audience attendance** across multiple theater locations. Using historical booking data, theater characteristics, and temporal patterns, the solution employs a comprehensive machine learning pipeline that combines:

- **Advanced Feature Engineering** with lag features, rolling statistics, and cyclical encodings
- **Multiple ML Models** including Random Forest, Gradient Boosting, XGBoost, and LightGBM
- **Ensemble Techniques** with weighted statistical modeling
- **Time Series Analysis** with seasonality and trend detection

**Final Achievement**: Developed a robust prediction system with **R² score of 0.716** and **RMSE of 16.37** on validation data.

---

## 🎪 Problem Statement

Cinema chains need to accurately forecast daily audience counts to:
- **Optimize staffing** levels based on expected crowd
- **Manage inventory** (concessions, tickets) efficiently
- **Plan marketing campaigns** for low-attendance periods
- **Maximize revenue** through dynamic pricing strategies

**Challenge**: Predict the `audience_count` for each theater on specific future dates using historical patterns, booking data, and external factors.

---

## 📊 Dataset Description

The project utilizes **8 interconnected datasets** containing theater operations data:

### Primary Datasets

| Dataset | Description | Key Features |
|---------|-------------|--------------|
| **booknow_visits** | Historical daily audience counts | `book_theater_id`, `show_date`, `audience_count` |
| **booknow_theaters** | Theater characteristics | `theater_type`, `theater_area`, location info |
| **booknow_booking** | Individual booking transactions | `show_datetime`, `booking_datetime`, `tickets_booked` |
| **date_info** | Calendar information | `show_date`, `day_of_week`, `holiday_flg` |
| **sample_submission** | Submission format | `ID`, `audience_count` (to predict) |

### Secondary Datasets

| Dataset | Description |
|---------|-------------|
| **cinePOS_theaters** | Alternative theater system data |
| **cinePOS_booking** | Alternative booking system data |
| **movie_theater_id_relation** | Theater ID mapping between systems |

### Data Characteristics
- **Total Records**: ~250,000+ visits
- **Date Range**: March 2023 - January 2024
- **Theaters**: 100+ unique locations
- **Target Variable**: `audience_count` (range: 2-151)

---

## 🔄 Project Workflow

```
┌─────────────────────┐
│  Data Collection    │
│  & Loading          │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Data Cleaning &    │
│  Preprocessing      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Exploratory Data   │
│  Analysis (EDA)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Feature            │
│  Engineering        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Model Training &   │
│  Comparison         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Hyperparameter     │
│  Tuning             │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Ensemble &         │
│  Statistical Model  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Prediction &       │
│  Submission         │
└─────────────────────┘
```

---

## 🔍 Exploratory Data Analysis

### Key Findings

#### 1. **Distribution Analysis**
- **Right-skewed distribution**: Most theaters have low to moderate audience counts
- **Outliers**: ~15% of records are statistical outliers (IQR method)
- **Mean audience**: 34.2 people per show
- **Median audience**: 28 people per show

#### 2. **Temporal Patterns**
- **Weekly Seasonality**: Clear weekend effect
  - **Peak days**: Sunday (highest), Saturday, Monday
  - **Low days**: Tuesday-Thursday
- **Monthly Seasonality**: 
  - **Peak months**: December 2023 (holiday season), March 2023
  - **Low months**: Mid-2023 (summer dip), January 2024
- **Structural break**: July 2023 - audience tripled and sustained higher level

#### 3. **Theater Characteristics**
- **Theater Types**: "Other" (48%), "Comedy" (25%), "Drama" (20%), "Action" (7%)
- **High variability** across individual theaters
- **Location matters**: Theater area significantly impacts attendance

### Visualization Highlights

The project includes **12+ comprehensive visualizations**:
- Outlier boundary scatter plots
- Theater type distribution (pie chart)
- Monthly trend analysis
- Weekly attendance patterns (bar chart)
- Distribution plots (histogram, box plot, density)
- Daily time series trends
- Heatmap: Month vs Day of Week attendance patterns

---

## ⚙️ Feature Engineering

### 1. **Temporal Features** (13 features)
```python
- year, month, day, day_of_week_num
- quarter, day_of_year, is_weekend
- Cyclical encodings: month_sin, month_cos, day_sin, day_cos
```

**Purpose**: Capture seasonality and cyclical patterns correctly (e.g., December → January continuity)

### 2. **Lag Features** (4 features)
```python
- audience_lag_1d   # Yesterday's attendance
- audience_lag_7d   # Same day last week
- audience_lag_14d  # Two weeks ago
- audience_lag_30d  # One month ago
```

**Purpose**: Use recent history to predict future (time series autoregression)

### 3. **Rolling Statistics** (6 features)
```python
- audience_mean_7d, audience_std_7d    # Weekly patterns
- audience_mean_14d, audience_std_14d  # Biweekly patterns
- audience_mean_30d, audience_std_30d  # Monthly patterns
```

**Purpose**: Smooth noise and capture trends; standard deviation measures volatility

### 4. **Theater-Level Statistics** (5 features)
```python
- theater_mean, theater_median, theater_std
- theater_min, theater_max
```

**Purpose**: Baseline characteristics of each theater's typical performance

### 5. **Day-of-Week Theater Patterns** (3 features)
```python
- dow_mean    # Average attendance for this theater on this day
- dow_median  # Median attendance pattern
- dow_count   # Sample size for reliability
```

**Purpose**: Capture theater-specific weekly patterns (e.g., Theater A is busy on Fridays)

### 6. **Encoded Categorical Features** (2 features)
```python
- theater_type_encoded   # Label encoding of theater type
- theater_area_encoded   # Label encoding of theater area
```

### 7. **Exponential Moving Averages (EMA)** (1 feature)
```python
- ema_7  # Weighted recent average with exponential decay
```

### 8. **Difference Features** (2 features)
```python
- diff_1  # Change from yesterday
- diff_7  # Change from last week
```

**Purpose**: Capture momentum and trend direction

### 9. **External Features** (1 feature)
```python
- tickets_booked  # Advanced bookings (leading indicator)
```

**Total Features**: **27 engineered features** fed into models

---

## 🤖 Model Development

### Model Comparison

| Model | Train RMSE | Val RMSE | Train MAE | Val MAE | Train R² | Val R² |
|-------|------------|----------|-----------|---------|----------|--------|
| **Random Forest** | 13.13 | 16.82 | 9.39 | 11.31 | 0.844 | 0.700 |
| **Gradient Boosting** | 14.71 | 16.39 | 10.70 | 11.04 | 0.805 | 0.715 |
| **XGBoost** ⭐ | 14.86 | **16.37** | 10.71 | 11.02 | 0.801 | **0.716** |
| **LightGBM** | 17.44 | 16.60 | 11.97 | 11.28 | 0.726 | 0.708 |

**Winner**: **XGBoost** (lowest validation RMSE)

### Hyperparameter Tuning

Applied **RandomizedSearchCV** with 10 iterations on XGBoost:

**Parameter Space**:
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0]
}
```

**Best Parameters**:
```python
{
    'subsample': 0.9,
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05
}
```

**Result**: Original model performed better → kept original configuration

### Advanced Statistical Ensemble

Developed a **multi-level adaptive weighting model** that combines:

1. **Theater-DOW patterns** (40% weight)
2. **Recent lag features** (30% weight for lag_1)
3. **Rolling averages** (20% weight for 7-day rolling)
4. **EMA smoothing** (10% weight)
5. **Booking tickets** (22% weight as leading indicator)
6. **Theater baseline** (8% weight for stability)

**Contextual Adjustments**:
- Holiday boost: +12%
- Weekend boost: +8%

**Calibration**: Dynamic calibration factor (0.7 blend) to match training distribution

---

## 📈 Results

### Model Performance

- **Validation R² Score**: **0.716** (71.6% variance explained)
- **Validation RMSE**: **16.37**
- **Validation MAE**: **11.02**

### Prediction Quality

- **Mean Prediction**: 34.18 (vs Training Mean: 34.2)
- **Prediction Range**: [2, 151] (valid bounds)
- **Calibration**: Successfully matched training distribution

### Key Success Factors

1. ✅ **Rich temporal features** captured seasonality
2. ✅ **Lag features** provided strong autoregressive signals
3. ✅ **Theater-specific patterns** improved personalization
4. ✅ **Ensemble approach** balanced multiple signals
5. ✅ **Calibration** ensured realistic predictions

---

## 💡 Key Insights

### Business Insights

1. **Weekend Effect**: Theaters see **30-40% higher attendance** on Saturday/Sunday
2. **Monday Surprise**: Monday has second-highest attendance (post-weekend carryover)
3. **Holiday Peak**: December shows **3x normal attendance** (holiday movie season)
4. **Mid-week Slump**: Tuesday-Thursday consistently lowest (opportunity for promotions)
5. **Theater Variability**: High variance between theaters suggests **location-specific strategies**

### Technical Insights

1. **Lag-1 is King**: Yesterday's attendance is the **strongest predictor**
2. **Rolling features reduce noise**: 7-day rolling average provides stable signal
3. **Cyclical encoding matters**: Sin/cos transforms handle month/day continuity
4. **XGBoost outperforms**: Better at handling non-linear patterns than linear models
5. **Ensemble adds value**: Statistical model complements ML predictions

---

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook or VS Code
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
```

---

## 📖 Usage

### 1. **Load the Notebook**
```bash
cd "c:\Users\goura\OneDrive\Desktop\viva procedure"
jupyter notebook "23f3000511-notebook-t32025 (7).ipynb"
```

### 2. **Update Data Paths**
Modify the dataset paths in the notebook to point to your data location:
```python
d1 = pd.read_csv("path/to/booknow_visits.csv")
d2 = pd.read_csv("path/to/booknow_theaters.csv")
# ... etc
```

### 3. **Run All Cells**
Execute the notebook sequentially from top to bottom.

### 4. **Output**
The final submission file `submission.csv` will be generated with format:
```
ID,audience_count
BK_THEATER001_2024-01-15,45
BK_THEATER001_2024-01-16,52
...
```

---

## 📁 Project Structure

```
viva procedure/
│
├── 23f3000511-notebook-t32025 (7).ipynb    # Main analysis notebook
├── README.md                                # This file
├── submission.csv                           # Final predictions
│
├── complete.html                            # Exported reports
├── final.html
├── Workshop_Master_System.html.html
│
└── data/                                    # (User's data directory)
    ├── booknow_visits.csv
    ├── booknow_theaters.csv
    ├── booknow_booking.csv
    ├── date_info.csv
    ├── sample_submission.csv
    ├── cinePOS_theaters.csv
    ├── cinePOS_booking.csv
    └── movie_theater_id_relation.csv
```

---

## 🎓 Methodology Summary

### 1. **Data Preprocessing**
- Handled missing values (median imputation for coordinates)
- Removed duplicates (drop_duplicates)
- Sorted by theater and date for time series continuity
- Merged multiple datasets (theater info, date info, bookings)

### 2. **Feature Engineering**
- Created 27 features across 9 categories
- Used domain knowledge (weekend, holiday effects)
- Applied mathematical transformations (sin/cos, rolling, EMA)

### 3. **Model Training**
- Trained 4 different models
- Used 80-20 time-series split (no shuffling)
- Evaluated using RMSE, MAE, R² metrics

### 4. **Ensemble & Calibration**
- Developed weighted statistical model
- Applied dynamic calibration to match training distribution
- Light smoothing for stability (2-period rolling)

### 5. **Prediction & Validation**
- Feature propagation to test set
- Applied prediction pipeline
- Clipped to valid range [2, 151]

---

## 🏆 Author

**Gourav Rustagi**

- 🎓 Data Scientist & Machine Learning Engineer
- 💼 Specialization: Time Series Forecasting, Predictive Analytics
- 🔗 LinkedIn:(https://www.linkedin.com/in/gourav-rustagi-a2121a54/)

---

## 📜 License

This project is available for educational and research purposes.

---

## 🙏 Acknowledgments

- **Kaggle Community** for dataset and competition platform
- **Scikit-learn & XGBoost Teams** for excellent ML libraries
- **Open Source Community** for tools and frameworks

---

## 📚 References

1. **Time Series Forecasting**: Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice*
2. **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
3. **Feature Engineering**: Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ by Gourav Rustagi

</div>

