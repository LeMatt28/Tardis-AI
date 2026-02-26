# TARDIS 

> **Train Delay Analysis & Prediction System**
> SNCF Data Analysis Service Project

---

## Project Overview

TARDIS is a data science project developed to analyze historical SNCF train delay data and build a predictive model capable of estimating train delays before departure.

The project follows four major steps:

1. **Data Cleaning & Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Predictive Modeling**
4. **Interactive Streamlit Dashboard**

The final result is a machine learning-powered dashboard that allows users to explore delay patterns and predict future train delays.

---

## Objectives

* Clean and preprocess historical train delay data
* Perform exploratory data analysis with visual insights
* Build and evaluate a regression model to predict delay duration (in minutes)
* Deploy an interactive Streamlit dashboard integrating the trained model

---

## Repository Structure

```
📁 TARDIS
├── cleared_dataset.csv
├── data
│   └── project_dataset.csv
├── model.pkl
├── requirements.txt
├── tardis_dashboard.py
├── tardis_eda.ipynb
└── tardis_model.ipynb
```

### File Descriptions

| File                  | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `requirements.txt`    | Project dependencies                                   |
| `tardis_eda.ipynb`    | Data cleaning, preprocessing, and exploratory analysis |
| `cleared_dataset.csv` | Cleaned dataset exported from EDA notebook             |
| `tardis_model.ipynb`  | Model training, evaluation, and selection              |
| `model.pkl`           | Trained regression model                               |
| `tardis_dashboard.py` | Streamlit interactive dashboard                        |
| `README.md`           | Project documentation                                  |

---

# Step 1 – Data Cleaning & Preprocessing

Performed in **`tardis_eda.ipynb`**

### Tasks Completed

* Dataset loading and inspection
* Missing value handling
* Duplicate removal
* Data type conversion
* Feature engineering (e.g., date-based features such as month/year extraction)

###  Output

* `cleared_dataset.csv`

---

#  Step 2 – Exploratory Data Analysis

Also performed in **`tardis_eda.ipynb`**

### Analysis Included

* Summary statistics
* Delay distribution visualization
* Station-based delay comparison
* Time-based trend analysis

Each visualization includes written interpretation explaining insights and patterns discovered.

---

# Step 3 – Predictive Modeling

Implemented in **`tardis_model.ipynb`**

### Target

Predict:

```
Delay duration (in minutes)
```

###  Process

* Feature encoding (categorical variables with One-Hot Encoding)
* Train/test split
* Model training (regression model)
* Performance evaluation

###  Evaluation Metrics

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

The selected model outperforms a baseline predictor (mean delay).

###  Output

* `model.pkl` (used in the dashboard)

---

#  Step 4 – Streamlit Dashboard

Implemented in **`tardis_dashboard.py`**

## Features

###  Delay Distribution Visualization

Histogram showing delay patterns.

###  Summary Statistics Panel

* Total trips
* Average delay
* Total cancelled trains

### Interactive Filters

Users can:

* Filter by departure station
* Explore station-specific delay averages

###  Prediction Interface

Users input:

* Departure station
* Arrival station
* Service type
* Planned trips
* Cancelled trains

The dashboard returns:

```
Estimated delay (in minutes)
```

---

#  Installation & Setup

##  Clone the repository

```bash
git clone <your-repo-url>
cd TARDIS
```

##  Install dependencies

```bash
pip install -r requirements.txt
```

##  Run the Streamlit dashboard

```bash
streamlit run tardis_dashboard.py
```

The application will open automatically in your browser.

---

#  Technologies Used

* Python
* pandas
* numpy
* matplotlib
* scikit-learn
* joblib
* streamlit
