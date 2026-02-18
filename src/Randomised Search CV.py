import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint,uniform,loguniform
from sklearn.metrics import (r2_score,root_mean_squared_error,
                             mean_squared_error,
                             mean_absolute_error)

Data_Directory = "data"
os.makedirs(Data_Directory,exist_ok=True)

# 1. Read The Data :-
housing = pd.read_csv(os.path.join(Data_Directory,"housing.csv"))

# 2.Seperate Train Data And Test Data :- 
housing['income_category']= pd.cut(housing['median_income'],
                                   bins=[0,1.5,3.0,4.5,6.0,np.inf],
                                   labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_set,test_set in split.split(housing , housing['income_category']):
    train_data = housing.loc[train_set].drop('income_category',axis=1)
    test_data = housing.loc[test_set].drop('income_category',axis=1)

# 3.Seperate Train Label And Features
train_features = train_data.drop('median_house_value',axis=1)
train_label = train_data['median_house_value']

test_features = test_data.drop('median_house_value',axis=1)
test_label = test_data['median_house_value']

# 4.Seperate Num Attributes and Cat Attributes
num_attr = train_features.drop('ocean_proximity',axis=1).columns.tolist()
cat_attr = ['ocean_proximity']

# Make The Pipeline :-
# Num Pipeline :-
num_pip = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('skew',PowerTransformer(method='yeo-johnson')),
    ('standardscaler',StandardScaler())
])

# Cat Pipeline :-
cat_pip = Pipeline([
    ('onehotencoder',OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num',num_pip,num_attr),
    ('cat',cat_pip,cat_attr)
])

# 5.Prepared The Data :-
train_prepared_data = full_pipeline.fit_transform(train_features)
test_prepared_data = full_pipeline.transform(test_features)



model = XGBRegressor()

para_dist = {
    'n_estimators' : randint(200,800),
    'max_depth' : randint(3,10),
    'min_child_weight' : randint(1,10),
    'gamma' : uniform(0,5),
    'learning_rate' : loguniform(1e-5,0.3),
    'reg_alpha' : loguniform(1e-4,10),
    'reg_lambda': loguniform(1e-3,10),
    'colsample_bytree' : uniform(0.6,0.3)

}

random_search = RandomizedSearchCV(
    estimator= model,
    param_distributions=para_dist,
    cv = 5,
    verbose=2,
    n_jobs=-1,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

random_search.fit(train_prepared_data,train_label)
print("Best_parameter",random_search.best_params_)
best_model = random_search.best_estimator_
print(f"The Best Model is :  {best_model}")

pred = best_model.predict(test_prepared_data)
r2score = r2_score(test_label,pred)
print(f"The R2 Score is : {r2score}")