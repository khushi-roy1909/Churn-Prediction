from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('Churn_Prediction_Model.pkl')  # Ensure your model is saved as random_forest_titanic.pkl

# Load the scaler and encoder from saved files
scaler = joblib.load('scaler.pkl')  # Assume scaler was saved as scaler.pkl
le = joblib.load('label_encoder.pkl')  # Assume label encoder was saved as label_encoder.pkl

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input values from form
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        gender = request.form['gender'].capitalize()  # Capitalize input to make it case-insensitive
        senior_citizen = int(request.form['senior_citizen'])
        partner = request.form['partner'].capitalize()
        dependents = request.form['dependents'].capitalize()
        phone_service = request.form['phone_service'].capitalize()
        multiple_lines = request.form['multiple_lines'].capitalize()

        # Encode categorical inputs
        if 'Gender' in le:
          gender = le['Gender'].transform([gender])[0]
        else:
          return "Error: 'Gender' encoder not found", 500

        partner = le['Partner'].transform([partner])[0]
        dependents = le['Dependents'].transform([dependents])[0]
        phone_service = le['PhoneService'].transform([phone_service])[0]
        multiple_lines = le['MultipleLines'].transform([multiple_lines])[0]

        # Prepare the input data
        input_data = np.array([[tenure, monthly_charges, total_charges, gender, senior_citizen, partner, dependents, phone_service, multiple_lines]])
        input_data_scaled = scaler.transform(input_data)

        # Predict using the trained model
        churn_prediction = model.predict(input_data_scaled)[0]
        
        result = "No Churn 😄" if churn_prediction == 0 else "Churn 😢"
        return render_template("index.html", result=result)

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
