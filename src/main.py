import pandas as pd 
import numpy as np 
import joblib 
import os 
import logging
from scipy.stats import uniform,loguniform,randint
from sklearn.model_selection import (StratifiedShuffleSplit,
                                     RandomizedSearchCV,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import (r2_score,mean_absolute_error,
                             mean_squared_error)
from sklearn.base import BaseEstimator,TransformerMixin
DATA_DIRECTORY = 'data'
LOG_DIRECTORY = 'logs'
JOBLIB_DIRECTORY = 'pkl_file'
DATA_FILE = "housing.csv"

os.makedirs(DATA_DIRECTORY,exist_ok=True)
os.makedirs(LOG_DIRECTORY,exist_ok=True)
os.makedirs(JOBLIB_DIRECTORY,exist_ok=True)



logging.basicConfig(
    filename = os.path.join(LOG_DIRECTORY,'housing_price_pipeline.log'),
    level = logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s',
    filemode= 'a'
)


METRICS_FILE = 'metrics.pkl'
FULL_PIPELINE_FILE = 'model_pipeline.pkl'


# Load The Data :-
def load_data(path):
    try:
        logging.info(f"File Loaded Successfully : {path}")
        data =  pd.read_csv(path)
        logging.info(f"Data Shape is : {data.shape}")
        return data
    except FileNotFoundError as e:
        logging.error(f"File is Not Found : {path}")
        raise e 
    except pd.errors.EmptyDataError as e:
        logging.error(f"Data is Not Found : {path}")
        raise e 
    except Exception as e :
        logging.error(f"Error in Loading : {e}")
        raise e
# Split the data :-
def splitting_data(data):
    try:
        data = data.copy()
        data['income_category'] = pd.cut(data['median_income'],
                                        bins=[0,1.5,3,4.5,6,np.inf],
                                        labels=[1,2,3,4,5])
        split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
        for train_set,test_set in split.split(data,data['income_category']):
            train_data = data.loc[train_set].drop('income_category',axis=1)
            test_data = data.loc[test_set].drop('income_category',axis=1)
            logging.info("Splitting Train and Test Data Successfully.")
            logging.info(f"Train Data : {train_data.shape}") 
            logging.info(f"Test Data : {test_data.shape}")
            return train_data,test_data 
    except Exception as e :
        logging.error(f"Error during splitting the data : {e}")
        raise e 
class IQR_clipping(BaseEstimator,TransformerMixin):
    def __init__(self):
            self.bounds = {}

    def fit(self,x,y=None):
        x = pd.DataFrame(x)
        for col in x.columns:
            Q1 = x[col].quantile(0.25)
            Q3 = x[col].quantile(0.75)
            IQR = Q3-Q1
                
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            self.bounds[col] = (lower_limit,upper_limit)

        return self
        
    def transform(self,x):
        x = pd.DataFrame(x)
        for col,(lower,upper) in self.bounds.items():
            x[col] = x[col].clip(lower,upper)
        return x
    
# Seperate Label and Features:-
def separate_features_label(train_data,test_data):
    try:
        train_features = train_data.drop('median_house_value',axis=1)
        train_label = train_data['median_house_value']
        test_features = test_data.drop('median_house_value',axis=1)
        test_label = test_data['median_house_value']
        train_data.to_csv(os.path.join(DATA_DIRECTORY,"train_data.csv"),index=False)
        test_data.to_csv(os.path.join(DATA_DIRECTORY,"test_data.csv"),index=False)
        logging.info("Features And Labels Seperate Successfully.")
        return train_features,train_label,test_features,test_label
    except Exception as e :
        logging.error(f"Error during the seperate features and labels : {e}")
        raise e
       
# Seperate Numerical And Categorical Attributes :-
def sep_num_cat(train_features):
    try:
        num_attr = train_features.drop('ocean_proximity',axis=1).columns.tolist()
        cat_attr = ['ocean_proximity']
        logging.info("Seperate Numerical And Categorical Attributes Successfully.")
        logging.info(f"Numerical Attributes : {num_attr}")
        logging.info(f"Categorical Attributes : {cat_attr}")
        return num_attr,cat_attr
    except Exception as e :
        logging.error(f"Error during seperate categorical and numerical attribute : {e}")
        raise e
# Build A Transformation Data Pipeline 
def build_pipeline(num_attr,cat_attr):
    try:
        num_pip = Pipeline([
            ('simpleimputer',SimpleImputer(strategy='median')),
            ('iqr',IQR_clipping()),
            ('skewness',PowerTransformer(method='yeo-johnson')),
            ('standardscaler',StandardScaler())
        ])

        cat_pip = Pipeline([
            ('onehotencoder',OneHotEncoder(handle_unknown='ignore'))
        ])

        full_pipeline = ColumnTransformer([
            ('num',num_pip,num_attr),
            ('cat',cat_pip,cat_attr)
        ])
        return full_pipeline
    except Exception as e:
        logging.error(f"Error during transformed the data : {e}")
        raise e 

# Build The Model Pipeline
def model_pipeline(num_attr,cat_attr):
    try :
        logging.info("Model Pipeline Loading...")
        return Pipeline([
            ('transformation',build_pipeline(num_attr,cat_attr)),
            ('model',XGBRegressor(objective = 'reg:squarederror',
                                tree_method='hist'
                                ,random_state=42,
                                n_jobs=-1,
                                eval_metric = 'rmse'))
        ])
        
    except Exception as e:
        logging.error(f"Error during model building : {e}")
        raise e
# Use Random Search For Getting Best Parameter 
def random_search_cv(num_attr,cat_attr,train_features,train_label):
    try:
        logging.info("Random Search CV Started...")
        para_distr = {
            'model__n_estimators' : randint(400,900),
            'model__max_depth' : randint(3,6),
            'model__reg_alpha' : loguniform(0.1,5),
            'model__subsample': uniform(0.6, 0.4),
            'model__reg_lambda' : loguniform(1,10),
            'model__colsample_bytree' : uniform(0.2,0.8),
            'model__learning_rate' : loguniform(0.01,0.2),
            'model__gamma' : uniform(2,10),
            'model__min_child_weight' : randint(10,30)

        }

        model_pipe = model_pipeline(num_attr,cat_attr)

        rscv = RandomizedSearchCV(
            estimator= model_pipe,
            param_distributions=para_distr,
            n_iter = 50,
            cv = 5,
            verbose = 2,
            n_jobs=-1,
            scoring = 'neg_root_mean_squared_error',
            random_state=42
        )
        rscv.fit(train_features,train_label)
        logging.info(f"Best Parameter : {rscv.best_params_}")
        logging.info(f"Best cv Rmse : {-rscv.best_score_}")
        return rscv.best_estimator_
    except Exception as e:
        logging.error(f"Error during Random Search : {e}")
        raise e

# Getting Metrics To Check Model Evaluation
def evaluate_model(best_model,train_features,test_features,train_label,test_label):
    try:
        logging.info("Metrics Calculation Started...")
        train_pred = best_model.predict(train_features)
        test_pred = best_model.predict(test_features)
        train_r2 = r2_score(train_label,train_pred)
        train_mse = mean_squared_error(train_label,train_pred)
        train_mae = mean_absolute_error(train_label,train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_label,train_pred))
        test_r2 = r2_score(test_label,test_pred)
        test_mse = mean_squared_error(test_label,test_pred)
        test_mae = mean_absolute_error(test_label,test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_label,test_pred))
        cvs = -cross_val_score(best_model,train_features,train_label,
                               scoring='neg_root_mean_squared_error',
                               cv=5,n_jobs=-1)

        metric = {
            'train_r2_score' : train_r2,
            'train_mse' : train_mse,
            'train_mae' : train_mae,
            'train_rmse' : train_rmse,
            'test_r2_score' : test_r2,
            'test_mse' : test_mse,
            'test_mae' : test_mae,
            'test_rmse' : test_rmse,
            'cv_rmse_mean' : cvs.mean(),
            'cv_rmse_std' :cvs.std()
        }
        return metric
    except Exception as e:
        logging.error(f"Error during the calculating metrics : {e}")
        raise e

# Run All Function to Train the Model
def train_and_save_model():
    try:
        logging.info("=== Training Started ===")
        housing = load_data(os.path.join(DATA_DIRECTORY,DATA_FILE))
        train_data,test_data = splitting_data(housing)
        train_features,train_label,test_features,test_label = separate_features_label(train_data,test_data)
        num_attr,cat_attr = sep_num_cat(train_features)
        best_model = random_search_cv(num_attr,cat_attr,train_features,train_label)
        metric =  evaluate_model(best_model,train_features,test_features,train_label,test_label)
        logging.info("=== Training Completed ===")
    
    # Save The Model and Pipeline in Given File
    
        joblib.dump(metric,os.path.join(JOBLIB_DIRECTORY,METRICS_FILE))
        joblib.dump(best_model,os.path.join(JOBLIB_DIRECTORY,FULL_PIPELINE_FILE))

        logging.info(f"metrics : {metric}")
        logging.info("Model Trained Successfully")
        logging.info("Model Saved Succesfully.")
        return best_model
    except Exception as e:
        logging.error(f"Error during model training : {e}")
        raise e
 
# Load The Model For Testing
def load_model():
    logging.info("Model Loading For Prediction Successfully")
    try:
        return joblib.load(os.path.join(JOBLIB_DIRECTORY,FULL_PIPELINE_FILE))
    except Exception as e:
        logging.error(f"Error during load model : {e}")
        raise e

# Load Testing Data
def load_test_data(path):
    logging.info("Testing Data Loading Successfully")
    try:
         return pd.read_csv(path).drop('median_house_value',axis=1,errors='ignore')
    except KeyError as e:
        logging.error(f"Required file is not found: {path}")
        raise e
    except Exception as e:
        logging.error(f"Error during load data : {path}")
        raise e

# Make The Prediction  
def prediction(best_model,input_data):
    try:
        prediction = best_model.predict(input_data)
        input_data['Final_Prediction'] = prediction
        input_data.to_csv(os.path.join(DATA_DIRECTORY,'Output_File.csv'),index=False)
        logging.info("Prediction Is Done")
        return input_data
    except Exception as e:
        logging.error(f"Error during predicting output : {e}")
        raise e

# Main File 
if __name__ == "__main__":
    try:
        if not os.path.exists(os.path.join(JOBLIB_DIRECTORY,FULL_PIPELINE_FILE)):
            best_model = train_and_save_model()
        else:
            best_model = load_model()
            input_data = load_test_data(os.path.join(DATA_DIRECTORY,'test_data.csv'))
            housing_prediction = prediction(best_model,input_data)
    except Exception as e:
        logging.error(f"Pipeline Failed : {e}")
        logging.info("Model Is Fully Ready..!")
        print("If Error Occurs Check In Log File For Details....")
        