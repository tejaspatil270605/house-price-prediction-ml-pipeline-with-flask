## HOUSING PRICE PREDICTION (EDA + ML PIPELINE + SHAP + FLASK)

### Project Overview:-
This is an end-to-end Data Science project that predicts housing prices using an optimized
XGBoost model.
It includes EDA in Jupyter Notebook, machine learning pipeline, SHAP explainability, and Flask
deployment.

### Project Workflow:-
EDA (Jupyter Notebook)
        ↓
Data Preprocessing Pipeline
        ↓
Model Training (XGBoost)
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Flask Deployment

### Project Structure:-
- data/ (dataset and outputs)
- notebook/EDA.ipynb (Exploratory Data Analysis)
- logs/ (pipeline logs)
- pkl_file/ (saved model & metrics)
- shap/ (SHAP output plots)
- templates/ (HTML)
- static/ (CSS/images)
- src/main.py
- src/shap_analysis.py
- app.py (Flask app)

### Technologies Used:-
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Flask, Joblib, Matplotlib
ML Pipeline Features:
- Missing value handling
- IQR outlier clipping
- Skewness handling
- Standard scaling
- One-hot encoding
- RandomizedSearchCV tuning

### Evaluation Metrics:-
R2 Score, MAE, MSE, RMSE, Cross-validation RMSE

### How to Run:-
1. Install libraries:
pip install pandas numpy scikit-learn xgboost flask shap matplotlib joblib scipy jupyter
2. Run EDA:
jupyter notebook → open EDA.ipynb
3. Train model:
python main.py
4. SHAP analysis:
python shap_analysis.py
5. Run Flask app:
python app.py
Open: http://127.0.0.1:5000

### For Prediction :-
If model is already trained and saved:
Run:
python main.py

The system will:
- Load saved model automatically
- Take input from data/test_data.csv
- Generate predictions
- Save output in data/Output_File.csv

Note:
Model will retrain only if model_pipeline.pkl is not found.
Otherwise it runs in prediction mode.

### Resume Value:-
This project demonstrates end-to-end ML pipeline, feature engineering, hyperparameter tuning,model explainability (SHAP), Flask deployment, and production-level structure.

### Author:-
Tejas Patil
   - Aspiring Data Scientist & ML Engineer