# 🎯 Guess My Focus — AI-Powered Focus & Productivity Predictor

> *“Focus is not about time spent, but about energy directed.”*  
> — This project bridges **data science** and **self-awareness**, helping individuals take control of their study and work habits.

---

![Focus Illustration](https://images.unsplash.com/photo-1553877522-43269d4ea984?auto=format&fit=crop&w=1200&q=60)

---

## 🧭 Overview

In today’s hyper-connected world, **staying focused** is one of the biggest challenges faced by students and professionals alike.  
The ability to concentrate deeply is influenced by multiple lifestyle and behavioral factors such as **sleep, screen time, stress, study duration, motivation, and exercise habits**.

This project leverages **Machine Learning (Ridge Regression)** to predict an individual’s **FocusScore (0–100)** —  
a measure of how effectively they can maintain attention and productivity throughout the day.

Unlike generic productivity trackers, **Guess My Focus** doesn’t just predict — it **explains**.  
It highlights *which habits most influence focus* and provides **personalized improvement suggestions** based on data-driven insights.

---

## 🚀 Key Features

| Feature | Description |
|----------|-------------|
| 🧠 **FocusScore Prediction (0–100)** | Predicts concentration levels based on study & lifestyle patterns |
| ⏱️ **Estimated Actual Focus Time** | Calculates true focused hours from total study time |
| 💬 **Personalized Tips & Feedback** | Smart recommendations to boost focus |
| 📊 **Interactive Visualization** | Gauge, radar, and line charts to visualize focus efficiency |
| ⚙️ **Auto Model Scaling** | Automatically adjusts input scaling (works even without `scaler.pkl`) |
| 🧩 **Explainability** | Understand which features affect focus and how |

---

## 🧩 Tech Stack

**Language:** Python 🐍  
**Frameworks & Libraries:**
- [Streamlit](https://streamlit.io/) — Web App Framework  
- [scikit-learn](https://scikit-learn.org/) — Machine Learning  
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — Data Processing  
- [Plotly](https://plotly.com/python/) — Interactive Visualizations  
- [Joblib](https://joblib.readthedocs.io/en/latest/) — Model Serialization  

---

## 🧠 Machine Learning Workflow

1. **EDA + Feature Engineering** — Exploring correlations & creating meaningful features  
2. **Preprocessing** — Handling missing data, scaling, and transformations  
3. **Model Training** — Linear, Ridge, and Lasso Regression  
4. **Model Tuning** — Using `GridSearchCV` for hyperparameter optimization  
5. **Cross Validation** — 10-fold validation for model robustness  
6. **Evaluation Metrics** — MAE, RMSE, and R² for both train/test  
7. **Deployment** — Interactive app using Streamlit  

---

## 📈 Sample Predictions

| Study Hours | Sleep Hours | Motivation | Stress | Predicted FocusScore | Focus Time (hrs) | Level |
|--------------|-------------|-------------|---------|----------------------|------------------|--------|
| 8 | 7 | 9 | 3 | **91.8** | **7.34** | 🔥 High Focus |
| 5 | 6 | 6 | 5 | **59.4** | **2.97** | 😐 Moderate Focus |
| 2 | 5 | 3 | 8 | **32.1** | **0.64** | ⚠️ Low Focus |

---

## 💡 Personalized Tips Example

| Focus Level | Recommendation |
|--------------|----------------|
| 🔥 High Focus | Maintain routine, short breaks, avoid burnout |
| 😐 Moderate Focus | Plan sessions, reduce multitasking, light exercise |
| ⚠️ Low Focus | Sleep 7–8 hrs, digital detox, Pomodoro technique |

---

## 🖥️ How to Run Locally

### 🧰 Prerequisites
- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
guess-my-focus/
│
├── data/
│   ├── raw_data/                 # Original dataset
│   ├── processed_data/           # Cleaned dataset after preprocessing
│   ├── app/
│   │   ├── app.py                # Streamlit main app
│   │   ├── ridge_model.pkl       # Trained Ridge Regression model
│   │   └── scaler.pkl (optional)
│
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_model_tuning_and_regularization.ipynb
│   └── 07_final_interpretation_visuals.ipynb
│
└── README.md
