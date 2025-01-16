# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import seaborn as sns
import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import plotly.express as px
from models_details import multiple_models

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ['5','6', '7', '8','9']



#from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------
from import_analyse import basic_info,preprocess_data,eda_plots

# Assuming df is already loaded somewhere globally or within a function

from recamandation_code import recondation_fn

app = Flask(__name__)


@ app.route('/')
def home():
    title = 'IOT Devices foot print Detection Using Machine learning'
    return render_template('index.html', title=title)  



@app.route('/test_application')
def test_application():
    return render_template('recommendation.html')





@app.route('/upload_page')
def upload_page():
    return render_template('rust.html')

# render disease prediction result page
import joblib

# Load the saved model
loaded_model = joblib.load("xgboost_model.pkl")

# Now you can use the model for predictions or any other task
# For example:
# predictions = model.predict(X_test)


# Now you can use `loaded_model` to make predictions

@app.route('/predict1', methods=['POST'])
def predict1():
    if request.method == 'POST':
        # Collect the form data for the required inputs
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        proximity = float(request.form['proximity'])
        energy_consumption = float(request.form['energy_consumption'])
        transmission_rate = float(request.form['transmission_rate'])
        pollution_levels = float(request.form['pollution_levels'])
        redundancy_score = float(request.form['redundancy_score'])
        carbon_footprint = float(request.form['carbon_footprint'])
        cluster_id = int(request.form['cluster_id'])

        # Combine all inputs into a list
        input_features = [latitude, longitude, proximity, energy_consumption, 
                          transmission_rate, pollution_levels, redundancy_score, 
                          carbon_footprint, cluster_id]

        # Convert the list to a numpy array and reshape it to match the model's input shape
        final_features = np.array(input_features).reshape(1, -1)

        # Make the prediction using the loaded model
        prediction = loaded_model.predict(final_features)

        if prediction==0:
            result="let the sensor be on status"
        else:

            result="please turn of the sensor"



        # For simplicity, returning the prediction as a string (you can customize this further)
        return render_template('recommendation.html', 
                               prediction1_text='The predicted value is: {}'.format(result))

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
