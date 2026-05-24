# 🚀 Student Performance Prediction System

An end-to-end Machine Learning project that predicts student maths performance
using demographic and academic features through a live Flask web application.

🌐 **Live Demo:**  
https://student-performance-prediction-etz2.onrender.com

---

## 📌 Overview

This project demonstrates a complete production-style ML workflow including:

- Data Ingestion
- Data Transformation
- Model Training
- Model Evaluation
- Prediction Pipeline
- Flask Web Deployment

The system predicts a student's maths score based on:
- Gender
- Race/Ethnicity
- Parental Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

---

## ✨ Features

- 📊 Real-time student score prediction
- 🤖 Multiple ML algorithms benchmarking
- ⚡ Automated best-model selection
- 🌙 Interactive Flask UI
- 📈 Production-style modular pipeline
- ☁️ Live deployment on Render

---

## 🧠 Machine Learning Models Used

The project compares multiple regression algorithms:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- XGBoost Regressor
- CatBoost Regressor

The best-performing model is automatically selected based on evaluation score.

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask
- Scikit-Learn
- XGBoost
- CatBoost
- Pandas
- NumPy

### Deployment
- Render
- GitHub

---

## 📂 Project Structure

```bash
Student_performance_prediction/
│
├── artifacts/
├── notebooks/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── utils.py
│   └── exception.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── static/
├── app.py
├── requirements.txt
└── setup.py
