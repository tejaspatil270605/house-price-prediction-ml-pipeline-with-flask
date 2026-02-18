import os 
import pandas as pd 
import numpy as np 
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

# 6.Train The Model :-
models = {
    'Linear Regression' : LinearRegression(),
    'Random Forest Regressor' : RandomForestRegressor(random_state=42),
    'Decision Tree Regressor' : DecisionTreeRegressor(),
    'Support Vector Machine' : LinearSVR(),
    'Ridge Regressor' : Ridge(),
    'Lasso Regressor' : Lasso(),
    'KNeighours Regressor' : KNeighborsRegressor(),
    'XG Boost Regressor' : XGBRegressor(),
    'Light GBM Regressor' : LGBMRegressor()    
}

model_score = {}

for name,model in models.items():
    model.fit(train_prepared_data,train_label)
    train_pred = model.predict(train_prepared_data)
    test_pred =  model.predict(test_prepared_data)
    train_r2_score = r2_score(train_label,train_pred)
    test_r2_score = r2_score(test_label,test_pred)
    train_mae = mean_absolute_error(train_label,train_pred)
    test_mae = mean_absolute_error(test_label,test_pred)
    train_mse = mean_squared_error(train_label,train_pred)
    test_mse = mean_squared_error(test_label,test_pred)
    train_cvs = -cross_val_score(model,train_prepared_data,train_label,
                                 scoring='neg_root_mean_squared_error',
                                 cv=10)
    
    
    model_score[name] = {
          "train_r2_score" :  train_r2_score,
          "train_mae" : train_mae,
          "train_mse" : train_mse,
          "test_r2_score" : test_r2_score,
          "test_mae" : test_mae,
          "tset_mse" : test_mse
     }

    print(f"{name} = mae:{train_mae} , mse:{train_mse} , r2_score:{train_r2_score}")
    print(f"{name} = mae:{test_mae} , mse:{test_mse} , r2_score:{test_r2_score}")
    print(pd.Series(train_cvs).describe())
    

best_model_name = max(model_score,key=lambda x: model_score[x]["test_r2_score"])
best_model = models[best_model_name]
print(best_model)


