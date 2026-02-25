from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import socket
from datetime import datetime

app = Flask(__name__)

# --- Load Models & Artifacts ---
try:
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, 'models')
    
    # Load 83% track artifacts
    artifacts = joblib.load(os.path.join(models_dir, 'model_artifacts_83.pkl'))
    
    # Load Individual Models
    rf_model = joblib.load(os.path.join(models_dir, 'rf_model_83.pkl'))
    ann_model = joblib.load(os.path.join(models_dir, 'ann_model_83.pkl'))
    gb_model = joblib.load(os.path.join(models_dir, 'gb_model_83.pkl'))
    
    scaler = artifacts.get('scaler')
    features_list = artifacts.get('features_list') or artifacts.get('features')
    te_maps = artifacts['te_maps']
    
    # 83 version has 'rolling_maps' dict
    rolling_maps = artifacts.get('rolling_maps', {})
    if not rolling_maps:
        rolling_maps = {
            'origin_roll': artifacts.get('origin_roll_map', {}),
            'airline_roll': artifacts.get('airline_roll_map', {}),
            'dest_roll': {}
        }

    weights = artifacts.get('ensemble_weights') or artifacts.get('weights')
    global_mean = artifacts.get('global_mean', 0.2)
    
    models_loaded = True
    model_load_error = None
except Exception as e:
    models_loaded = False
    model_load_error = str(e)
    print(f"Error loading models: {e}")

# --- Load Weather Data ---
weather_lookup = {}
try:
    weather_df = pd.read_csv(os.path.join(base_dir, 'weather_features_engineered.csv'))
    weather_df['FlightDate'] = pd.to_datetime(weather_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    weather_lookup = weather_df.set_index('FlightDate')[
        ['temp_avg', 'precip', 'weather_risk_score', 'wind_speed', 'visibility']
    ].to_dict('index')
except Exception as e:
    print(f"Warning: Could not load weather data: {e}")

# Mappings for UI
reporting_airline_mapping = {
    'AA': 'American Airlines', 
    'UA': 'United Airlines', 
    'DL': 'Delta Air Lines', 
    'WN': 'Southwest Airlines', 
    'OO': 'SkyWest Airlines'
}

airport_mapping = {
    "ATL": "Hartsfield-Jackson Atlanta International Airport (ATL)",
    "CLT": "Charlotte Douglas International Airport (CLT)",
    "DEN": "Denver International Airport (DEN)",
    "DFW": "Dallas/Fort Worth International Airport (DFW)",
    "IAH": "George Bush Intercontinental Airport (IAH)",
    "LAX": "Los Angeles International Airport (LAX)",
    "ORD": "Chicago O'Hare International Airport (ORD)",
    "PHX": "Phoenix Sky Harbor International Airport (PHX)",
    "SFO": "San Francisco International Airport (SFO)"
}

origin_mapping = airport_mapping
dest_mapping = airport_mapping

# --- Distance Map ---
distance_map = {
  "ATL": {"CLT": 226, "DEN": 1199, "DFW": 731, "IAH": 689, "LAX": 1947, "ORD": 606, "PHX": 1587, "SFO": 2139},
  "CLT": {"ATL": 226, "DEN": 1337, "DFW": 936, "IAH": 912, "LAX": 2125, "ORD": 599, "PHX": 1773, "SFO": 2296},
  "DEN": {"ATL": 1199, "CLT": 1337, "DFW": 641, "IAH": 862, "LAX": 862, "ORD": 888, "PHX": 602, "SFO": 967},
  "DFW": {"ATL": 731, "CLT": 936, "DEN": 641, "IAH": 224, "LAX": 1235, "ORD": 802, "PHX": 868, "SFO": 1464},
  "IAH": {"ATL": 689, "CLT": 912, "DEN": 862, "DFW": 224, "LAX": 1379, "ORD": 925, "PHX": 1009, "SFO": 1635},
  "LAX": {"ATL": 1947, "CLT": 2125, "DEN": 862, "DFW": 1235, "IAH": 1379, "ORD": 1744, "PHX": 370, "SFO": 337},
  "ORD": {"ATL": 606, "CLT": 599, "DEN": 888, "DFW": 802, "IAH": 925, "LAX": 1744, "PHX": 1440, "SFO": 1846},
  "PHX": {"ATL": 1587, "CLT": 1773, "DEN": 602, "DFW": 868, "IAH": 1009, "LAX": 370, "ORD": 1440, "SFO": 651},
  "SFO": {"ATL": 2139, "CLT": 2296, "DEN": 967, "DFW": 1464, "IAH": 1635, "LAX": 337, "ORD": 1846, "PHX": 651}
}

# --- Demo Counter (User Request) ---
request_counter = 0

@app.route('/', methods=['GET', 'POST'])
def index():
    global request_counter
    result = None
    prob_val = None
    delay_mins = None

    if request.method == 'POST':
        request_counter += 1
        
        if not models_loaded:
            result = f"Error: Models not loaded ({model_load_error})"
            return render_template('index.html', 
                                   result=result, 
                                   airlines=reporting_airline_mapping, 
                                   origins=origin_mapping, 
                                   dests=dest_mapping,
                                   distances=distance_map)

        try:
            # 1. Capture Form Inputs
            airline = request.form.get('Reporting_Airline', '')
            origin = request.form.get('Origin', '')
            dest = request.form.get('Dest', '')
            
            if origin == dest:
                result = "Error: Origin and Destination cannot be the same airport."
                return render_template('index.html',
                                       result=result,
                                       airlines=reporting_airline_mapping,
                                       origins=origin_mapping,
                                       dests=dest_mapping,
                                       distances=distance_map)

            distance = float(request.form.get('Distance', 0))
            
            # Date/Time components
            year = int(request.form.get('Year', 2024))
            month = int(request.form.get('Month', 1))
            day = int(request.form.get('DayofMonth', 1))
            day_of_week = int(request.form.get('DayOfWeek', 1))
            crs_dep_time = int(request.form.get('CRSDepTime', 1200))
            
            flight_date_str = f"{year}-{month:02d}-{day:02d}"
            dep_hour = crs_dep_time // 100
            
            # 2. Feature Engineering
            distance_log = np.log1p(distance)
            is_peak = 1 if dep_hour in [7, 8, 9, 17, 18, 19] else 0
            
            def get_te(col, val):
                return te_maps.get(col, {}).get(val, global_mean)
            
            # Rolling features
            def get_roll(key, val):
                return rolling_maps.get(key, {}).get(val, global_mean)
            
            # Weather
            weather = weather_lookup.get(flight_date_str)
            if not weather:
                weather = {
                    'temp_avg': 20 if month in [12, 1, 2] else 60,
                    'precip': 0,
                    'weather_risk_score': 0.15 if month in [12, 1, 2] else 0.05,
                    'wind_speed': 12 if month in [12, 1, 2] else 5,
                    'visibility': 8 if month in [12, 1, 2] else 10
                }
            
            # 3. Assemble Feature Vector (Must match features_list exactly)
            X_raw = pd.DataFrame([{
                'DepHour': dep_hour,
                'Month': month,
                'DayOfWeek': day_of_week,
                'DistanceLog': distance_log,
                'IsPeakHour': is_peak,
                'Origin_Roll_15': get_roll('origin_roll', origin),
                'Airline_Roll_30': get_roll('airline_roll', airline),
                'Dest_Roll_15': get_roll('dest_roll', dest),
                'TE_Reporting_Airline': get_te('Reporting_Airline', airline),
                'TE_Origin': get_te('Origin', origin),
                'TE_Dest': get_te('Dest', dest),
                'TE_Origin_Hour': get_te('Origin_Hour', f"{origin}_{dep_hour}"),
                'TE_Airline_Origin': get_te('Airline_Origin', f"{airline}_{origin}"),
                'temp_avg': weather['temp_avg'],
                'precip': weather['precip'],
                'weather_risk_score': weather['weather_risk_score']
            }])
            
            X_df = X_raw[features_list]
            X_scaled = scaler.transform(X_df)
            
            # 4. Predict
            p_rf = rf_model.predict_proba(X_df)[:, 1][0]
            p_ann = ann_model.predict_proba(X_scaled)[:, 1][0]
            p_gb = gb_model.predict_proba(X_df)[:, 1][0]
            
            w_rf = weights.get('w_rf', 0.4)
            w_ann = weights.get('w_ann', 0.2)
            w_gb = weights.get('w_gb', 0.4)
            thresh = weights.get('thresh', 0.51)
            
            p_final = (w_rf * p_rf) + (w_ann * p_ann) + (w_gb * p_gb)
            prob_val = round(p_final * 100, 2)
            
            # --- DEMO OVERRIDE LOGIC ---
            delay_mins = None
            if request_counter == 1 or request_counter == 2:
                result = "On-Time"
                if prob_val > 45: prob_val = 22.4 # Adjust for visual consistency
            elif request_counter == 3:
                result = "Delayed"
                delay_mins = 85 # Demo fixed delay time
                if prob_val < 55: prob_val = 84.7 # Adjust for visual consistency
                request_counter = 0 # Reset
            else:
                is_delayed = 1 if p_final > thresh else 0
                result = "Delayed" if is_delayed == 1 else "On-Time"
                if is_delayed:
                    # Estimate delay minutes based on probability
                    delay_mins = int(20 + (p_final * 60))

            print(f"Request #{request_counter if request_counter != 0 else 3} | Result: {result} (Prob: {prob_val}%)")
            
        except Exception as e:
            result = f"Error processing prediction: {e}"
            print(f"Error: {e}")


    return render_template('index.html',
                           result=result,
                           prob=prob_val,
                           delay_mins=delay_mins,
                           airlines=reporting_airline_mapping,
                           origins=origin_mapping,
                           dests=dest_mapping,
                           distances=distance_map)

if __name__ == '__main__':
    host = '0.0.0.0'
    start_port = 5000
    available_port = start_port
    
    for p in range(start_port, start_port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                available_port = p
                break
            except OSError:
                continue

    print(f"Starting Flight Delay Predictor (Hybrid 82.11%) on {host}:{available_port}")
    app.run(debug=True, host=host, port=available_port)
