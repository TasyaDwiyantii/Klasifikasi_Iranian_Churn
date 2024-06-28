from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle
import os

app = Flask(__name__)

MODEL_PATH = 'gnb_model.pkl'

if not os.path.exists(MODEL_PATH):
    data_iranian = pd.read_csv("iranian_churn_clean.csv")
    X = data_iranian.drop(columns=['Churn'])
    y = data_iranian['Churn']

    # Initialize Gaussian Naive Bayes model and train it
    gnb_model = GaussianNB()
    gnb_model.fit(X, y)

    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(gnb_model, f)

    # Output to confirm training completion
    print("Model saved.")
else:
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        gnb_model = pickle.load(f)
    print("Model has been loaded from disk.")

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    input_data = {}
    error_message = None

    if request.method == 'POST':
        try:
            # Get user input from form and validate
            input_data = {
                'call_failure': request.form.get('call_failure', '0').replace(',', '.'),
                'complains': request.form.get('complains', '0').replace(',', '.'),
                'subscription_length': request.form.get('subscription_length', '0').replace(',', '.'),
                'charge_amount': request.form.get('charge_amount', '0').replace(',', '.'),
                'seconds_of_use': request.form.get('seconds_of_use', '0').replace(',', '.'),
                'frequency_of_use': request.form.get('frequency_of_use', '0').replace(',', '.'),
                'frequency_of_sms': request.form.get('frequency_of_sms', '0').replace(',', '.'),
                'distinct_called_number': request.form.get('distinct_called_number', '0').replace(',', '.'),
                'age_group': request.form.get('age_group', '0').replace(',', '.'),
                'tariff_plan': request.form.get('tariff_plan', '0').replace(',', '.'),
                'status': request.form.get('status', '0').replace(',', '.'),
                'age': request.form.get('age', '0').replace(',', '.'),
                'customer_value': request.form.get('customer_value', '0').replace(',', '.')
            }

            # Convert input values to floats
            for key, value in input_data.items():
                input_data[key] = float(value)

            # Make prediction for the user input
            new_data_point = [[
                input_data['call_failure'],
                input_data['complains'],
                input_data['subscription_length'],
                input_data['charge_amount'],
                input_data['seconds_of_use'],
                input_data['frequency_of_use'],
                input_data['frequency_of_sms'],
                input_data['distinct_called_number'],
                input_data['age_group'],
                input_data['tariff_plan'],
                input_data['status'],
                input_data['age'],
                input_data['customer_value']
            ]]
            predicted_class = gnb_model.predict(new_data_point)[0]
        except ValueError:
            error_message = "Invalid input: Please enter valid numbers."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
        if predicted_class == 0:
                predicted_class = "0 (Churn)"
        else:
            predicted_class = "1 (Tidak Churn)"


    return render_template('index.html', predicted_class=predicted_class, input_data=input_data, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
