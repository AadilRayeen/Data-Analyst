from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd  # Import pandas

app = Flask(__name__)

# Load the best model, preprocessor, and feature selector
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from request (coming from the user input via dropdowns)
    data = request.json
    traffic_condition = data['TrafficCondition']
    road_condition = data['RoadCondition']
    weather_condition = data['WeatherCondition']
    bus_capacity = float(data['BusCapacity'])
    avg_speed = float(data['AvgSpeed'])
    distance_to_destination = float(data['DistanceToDestination'])
    time_of_day = data['TimeOfDay']
    day_type = data['DayType']

    # Prepare data for prediction - Convert to pandas DataFrame
    input_data = pd.DataFrame([[traffic_condition, road_condition, weather_condition, 
                                bus_capacity, avg_speed, distance_to_destination, time_of_day, day_type]],
                                columns=['TrafficCondition', 'RoadCondition', 'WeatherCondition', 
                                         'BusCapacity', 'AvgSpeed', 'DistanceToDestination', 'TimeOfDay', 'DayType'])

    # Apply preprocessing and feature selection
    input_data_preprocessed = preprocessor.transform(input_data)
    input_data_selected = selector.transform(input_data_preprocessed)

    # Make prediction using the trained model
    predicted_delay = model.predict(input_data_selected)[0]

    return jsonify({'predicted_delay': round(predicted_delay, 2)})

if __name__ == '__main__':
    app.run(debug=True)
