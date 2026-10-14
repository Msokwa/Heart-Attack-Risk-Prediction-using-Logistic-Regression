# Heart-Attack-Risk-Prediction-using-Logistic-Regression
A supervised machine learning pipeline built in Python to predict heart patient risk using clinical indicators, demographics, and lifestyle data. Developed as part of the Machine Learning and Smart Systems course.
📌 Project Overview
Cardiovascular diseases remain a leading global health challenge. Early computational screening empowers preventive medicine and clinical decision-making. This project implements an end-to-end binary classification pipeline using Logistic Regression to analyze patient records, identify key health risk factors, and evaluate performance beyond simple accuracy.  

📊 Dataset Overview

Source: Kaggle Heart Attack Prediction Dataset  

Scale: 8,763 patient records with 26 distinct features  

Features Included:
Demographics: Age, Sex, Country, Continent
Lifestyle Factors: Smoking, Alcohol Consumption, Exercise Hours, Sleep Hours   
Medical Indicators: Cholesterol, Blood Pressure, Heart Rate, BMI, Diabetes status   
Target Variable: Binary Classification (0 = No Risk/Safe, 1 = Risk)   

⚙️ Data Preprocessing & Pipeline Steps

1.Data Cleaning: Removed unique identifiers (Patient ID, Name) that carry no predictive value.   

2.Categorical Encoding: Converted text categories into numeric binary columns using one-hot encoding (pd.get_dummies).  

3.Feature Scaling: Standardized numerical variables using StandardScaler to ensure stable gradient descent convergence and prevent large ranges (e.g., cholesterol) from overpowering smaller ones (e.g., age).   

4.Stratified Splitting: Divided data into an 80% training set and 20% testing set using stratify=y to preserve outcome distributions.  

5.Model Training: Fitted a LogisticRegression classifier (solver="lbfgs", C=1.0, max_iter=1000).   

6.Evaluation: Assessed model behavior using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.   

📈 Key Results & Metrics
Overall Accuracy: ~58%  
Confusion Matrix Breakdown:
True Negatives (Safe): 899  
False Positives (Alarm): 226  
True Positives (Caught): 117   
False Negatives (Missed): 511   

Key Takeaway: While raw accuracy provides a baseline, medical datasets require close analysis of False Negatives and Recall. Missing an at-risk patient carries severe clinical implications, highlighting the need for threshold tuning or advanced models (like Random Forests or SVMs) in future iterations.

🚀 How to Run the Code1.
1.Clone the RepositoryBashgit 
clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
2.cd "prediction (heart attack)"
3. Install Required Libraries
Make sure you have Python installed, then run:

pip install pandas scikit-learn matplotlib numpy
3. Execute the Python Script
python heart_model.py
