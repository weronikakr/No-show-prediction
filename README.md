# No-show-prediction

## 📋 Overview
This project uses machine learning to predict whether a patient will miss (no-show) a scheduled healthcare appointment. Patient no-shows can lead to inefficient resource use, increased wait times, and higher healthcare costs. By identifying patients at risk of missing their appointments, hospitals and clinics can take proactive steps such as sending reminders or offering flexible scheduling.

## 📂 Dataset
Dataset used for this project is available here: https://data.mendeley.com/datasets/wm6w2fvkfj/1

## ⚙️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-6f42c1?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C77A8?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

## 🚀 How to navigate

- **data/** – contains the raw and preprocessed datasets  
- **models/** – contains model scripts (Logistic Regression, Neural Network, XGBoost)  
- **results/** – saved metrics and evaluation outputs  
- **main.py** – script to run the project end-to-end  
- **data_preprocessing.ipynb** – notebook for data cleaning and preprocessing  
- **requirements.txt** – Python dependencies

## 📊 Results

NOTE: Current best results (so far)
| Model      | F1 (%) | Precision (%) | Recall (%) |
|------------|--------|---------------|------------|
| LogReg    | 21.9   | 14.0          | 50.7       |
| NeuralNetwork        | 24.9   | 16.6          | 49.6       |
| XGBoost   | 45.0   | 40.1          | 51.2       |


## 📜 Sources
Salazar, Luiz Henrique (2023), “Medical Appointments No-Show”, Mendeley Data, V1, doi: 10.17632/wm6w2fvkfj.1
