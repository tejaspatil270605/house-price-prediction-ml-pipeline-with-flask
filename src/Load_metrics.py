import joblib
import os

Joblib_Directory = "pkl_file"
os.makedirs(Joblib_Directory,exist_ok=True)

metrics = joblib.load(os.path.join(Joblib_Directory,'metrics.pkl'))
print(metrics)