from flask import Flask,render_template,request
from src.Custom_Transformer import IQR_clipping
import os 
import joblib
import pandas as pd 


app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

JOBLIB_DIRECTORY = "pkl_file"
os.makedirs(JOBLIB_DIRECTORY , exist_ok=True)

model = joblib.load(os.path.join(JOBLIB_DIRECTORY,"model_pipeline.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    
    longitude = float(request.form["longitude"])
    latitude = float(request.form["latitude"])
    housing_median_age = float(request.form["housing_median_age"])
    total_rooms = int(request.form["total_rooms"])
    total_bedrooms = int(request.form["total_bedrooms"])
    population = float(request.form["population"])
    households = float(request.form["households"])
    median_income = float(request.form["median_income"])
    ocean_proximity = request.form["ocean_proximity"]
    
    data = pd.DataFrame([{
        "longitude" : longitude,
        "latitude" : latitude,
        "housing_median_age" : housing_median_age,
        "total_rooms" : total_rooms,
        "total_bedrooms" : total_bedrooms,
        "population" : population,
        "households" : households,
        "median_income" : median_income,
        "ocean_proximity" : ocean_proximity
    }])
    
    prediction = model.predict(data)[0]
    price = round(prediction,2)
    return render_template("index.html",price=price)

if __name__ == "__main__":
     app.run(debug=True)