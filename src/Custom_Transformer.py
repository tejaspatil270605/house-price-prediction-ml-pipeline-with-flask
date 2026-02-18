from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd 

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
    
    def get_feature_names_out(self, input_features=None):
        return input_features