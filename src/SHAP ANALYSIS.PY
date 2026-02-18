import os 
import shap
import joblib 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from Custom_Transformer import IQR_clipping

Joblib_Directory = "pkl_file"
Data_Directory = "data"
Save_Fig = "src"

os.makedirs(Joblib_Directory,exist_ok=True)
os.makedirs(Data_Directory,exist_ok=True)
os.makedirs(Save_Fig,exist_ok=True)

best_model = joblib.load(os.path.join(Joblib_Directory,'model_pipeline.pkl'))

x_test = pd.read_csv(os.path.join(Data_Directory,'test_data.csv')).drop('median_house_value',axis=1,errors='ignore')

pipeline = best_model.named_steps['transformation']
model = best_model.named_steps['model']

x_test_transformed = pipeline.transform(x_test)
features_names = pipeline.get_feature_names_out()

explainer = shap.TreeExplainer(model)

values = explainer.shap_values(x_test_transformed)

print("plotting are started...wait..")
shap.summary_plot(values, x_test_transformed, feature_names=features_names,show=False)
plt.savefig(os.path.join(Save_Fig,'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
plt.close()

#  SHAP bar plot
shap.summary_plot(values, x_test_transformed, feature_names=features_names, plot_type='bar',show=False)
plt.savefig(os.path.join(Save_Fig,'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# SHAP force plot (matplotlib version)
shap.force_plot(
    explainer.expected_value,
    values[0],
    x_test_transformed[0],
    feature_names=features_names,
    matplotlib=True,
    show=False
)
plt.savefig(os.path.join(Save_Fig,'shap_force_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Plotting is finished!")

