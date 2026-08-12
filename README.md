# ❤️ Heart Disease Dashboard (Streamlit)

An interactive dashboard for exploring heart disease risk factors and predicting diagnosis, built with **Python + Streamlit**. Upgraded from an earlier Excel version.

🔗 **Live demo:** _[add your Streamlit Community Cloud link here after deploying]_
📊 **Excel version:** _[link to your original Excel repo]_

## Features

**Explore tab**
- Interactive filters — age range, gender, chest pain type, disease status
- KPI cards — patient count, disease presence %, average age, average cholesterol
- Age distribution, chest pain type breakdown, cholesterol vs max heart rate scatter, gender breakdown
- Correlation heatmap across all numeric features

**Predict tab**
- Choose between Logistic Regression or Random Forest
- Train the model live and view accuracy, precision, recall, F1 score
- Confusion matrix and ROC curve (AUC)
- Feature importance chart (Random Forest)
- Live prediction form — enter a patient's vitals and get an instant prediction with probability

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Plotly
- Scikit-learn

## Run it locally

```bash
git clone https://github.com/SaiSanthosh1308/HEART-DISEASE-DASHBOARD-STREAMLIT.git
cd HEART-DISEASE-DASHBOARD-STREAMLIT
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

270 patient records with 13 clinical features (age, gender, chest pain type, blood pressure, cholesterol, ECG results, max heart rate, exercise-induced angina, ST depression, and more), labeled with heart disease presence/absence.

## About This Project

This is a Python/Streamlit rebuild of an Excel dashboard I originally built for the same dataset, adding an interactive machine learning prediction layer — part of a broader effort to move from static Excel reporting to interactive, code-driven analytics tools as I build toward a data analyst role.
