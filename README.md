# 💰 Salary Prediction Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://salary-prediction-sahil-shankar.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline and web application designed to predict employee salaries with high precision. This project transforms raw demographic and professional data into actionable financial insights through a user-friendly interface.

---

## 🔗 Live Demo
Experience the predictive engine in real-time:  
👉 **[Salary Prediction Web App](https://salary-prediction-sahil-shankar.streamlit.app/)**

---

## 🚀 Overview
Predicting compensation is a critical task for both HR departments and job seekers. This project implements a robust predictive model that analyzes factors such as **Years of Experience**, **Education Level**, **Job Role**, and **Location** to estimate fair market value for various positions.

---

## ✨ Features
- **Live Web Interface:** Interactive Streamlit dashboard for real-time salary estimation.
- **Automated Preprocessing:** Comprehensive cleaning including outlier detection and handling missing values.
- **Feature Engineering:** Intelligent transformation of categorical data (Label/One-Hot Encoding) and feature scaling.
- **Interactive Visualizations:** Deep-dive EDA with correlation heatmaps and distribution plots.
- **Scalable Design:** Modular code structure for easy maintenance and updates.

---

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Deployment:** Streamlit Cloud

---

## 📂 Project Structure
├── app.py                  # Main Streamlit application
├── data/                   # Raw and processed datasets
├── notebooks/              # Exploratory Data Analysis & Model Training
├── src/                    # Source code for the pipeline
│   ├── preprocessing.py    # Data cleaning and encoding
│   └── train_model.py      # Script to train and save the model
├── models/                 # Saved model weights (.pkl or .h5)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

---

## Clone the Repository
**Open your terminal and run:**
git clone [https://github.com/sahil-shankar110/Salary-Prediction.git](https://github.com/sahil-shankar110/Salary-Prediction.git)
cd Salary-Prediction

---

## Create a Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

---

## Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

