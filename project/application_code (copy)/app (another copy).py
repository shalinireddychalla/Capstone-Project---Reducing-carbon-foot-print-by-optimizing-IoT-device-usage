# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ["Fake","Real"]



from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Pan Card DEtection / Verification'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'PanCard Detection'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)
            file = request.files.get('file')

            print(file)
        #if not file:
         #   return render_template('disease.html', title=title)
        #try:
            img1 = file.read()

            #print(img)

            prediction =pred_leaf_disease(img1)

            prediction = (str(disease_dic[prediction]))

            print(prediction)

            if prediction=="Fake":
                precaution="The PAN Card You Are Providing is Fake"
 
            else:

                precaution="Thanks For Using Our Service --Your Data is Real"




            return render_template('disease-result.html', prediction=prediction,precaution=precaution,title=title)
        #except:
         #   pass
    return render_template('disease.html', title=title)


# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
