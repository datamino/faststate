from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

current_dir = os.getcwd()  # Gets the current working directory
model_path = os.path.join(current_dir, 'models', 'model.pkl')
dataframe_path = os.path.join(current_dir, 'models', 'df.pkl')

# Load the model and dataset
try:
    with open(model_path, 'rb') as file:
        pipe = pickle.load(file)
    with open(dataframe_path, 'rb') as f:
        df = pickle.load(f)
    print("Model and dataset loaded successfully")
except Exception as e:
    print(f"Error loading model or dataset: {str(e)}")

@app.route('/')
def home():
    try:
        locations = sorted(df['location'].unique().tolist())
        bedrooms = sorted(df['bedrooms'].unique().tolist())
        bathrooms = sorted(df['bathrooms'].unique().tolist())
        kitchens = sorted(df['kitchens'].unique().tolist())
        parking_spaces = sorted(df['parking_spaces'].unique().tolist())
        luxury_categories = sorted(df['Luxury_catogory'].unique().tolist())
        built_in_types = sorted(df['built_in_type'].unique().tolist())
        
        return render_template('index.html', 
                             locations=locations,
                             bedrooms=bedrooms,
                             bathrooms=bathrooms,
                             kitchens=kitchens,
                             parking_spaces=parking_spaces,
                             luxury_categories=luxury_categories,
                             built_in_types=built_in_types)
    except Exception as e:
        app.logger.error(f"Error in home route: {str(e)}")
        return "Error loading page", 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log received data
        app.logger.info(f"Received form data: {request.form}")

        # Get values from the form
        form_data = {
            'type': request.form.get('type'),
            'location': request.form.get('location'),
            'area': request.form.get('area'),
            'bedrooms': request.form.get('bedrooms'),
            'bathrooms': request.form.get('bathrooms'),
            'kitchens': request.form.get('kitchens'),
            'parking_spaces': request.form.get('parking_spaces'),
            'luxury_category': request.form.get('luxury_category'),
            'built_in_type': request.form.get('built_in_type')
        }

        # Validate inputs
        for key, value in form_data.items():
            if value is None or value == '':
                return jsonify({
                    'success': False,
                    'error': f'Missing value for {key}'
                })

        # Convert numeric values
        form_data['area'] = float(form_data['area'])
        form_data['bedrooms'] = float(form_data['bedrooms'])
        form_data['bathrooms'] = float(form_data['bathrooms'])
        form_data['kitchens'] = float(form_data['kitchens'])
        form_data['parking_spaces'] = float(form_data['parking_spaces'])

        # Create DataFrame for prediction
        data = [[
            form_data['type'],
            form_data['location'],
            form_data['area'],
            form_data['bedrooms'],
            form_data['bathrooms'],
            form_data['kitchens'],
            form_data['parking_spaces'],
            form_data['luxury_category'],
            form_data['built_in_type']
        ]]
        
        columns = ['type', 'location', 'area', 'bedrooms', 'bathrooms', 'kitchens',
                  'parking_spaces', 'Luxury_catogory', 'built_in_type']
        
        new_df = pd.DataFrame(data, columns=columns)
        
        # Log DataFrame before prediction
        app.logger.info(f"Prediction DataFrame: {new_df}")

        # Make prediction
        price = round(float(np.expm1(pipe.predict(new_df))[0]), 1)  # Convert to float
        
        # Calculate price range
        lower_estimate = round(float(price - 0.58), 2)  # Convert to float
        if lower_estimate < 0:
            lower_estimate = str("-")
        upper_estimate = round(float(price + 0.58), 2)  # Convert to float

        # Log results
        app.logger.info(f"Prediction results: {price}, {lower_estimate}, {upper_estimate}")

        return jsonify({
            'success': True,
            'predicted_price': price,
            'lower_estimate': lower_estimate,
            'upper_estimate': upper_estimate
        })

    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run()