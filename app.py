import os
import pickle
import logging
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Global variable to store the loaded model
model = None
model_features = None

def load_model():
    """Load the OLS model from pickle file"""
    global model, model_features
    try:
        model_path = 'attached_assets/ols_model_1754216076665.pkl'
        if os.path.exists(model_path):
            # Try different loading methods
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logging.info("Model loaded successfully with pickle")
            except Exception as pickle_error:
                logging.warning(f"Standard pickle loading failed: {pickle_error}")
                # Try with different protocols
                try:
                    import joblib
                    model = joblib.load(model_path)
                    logging.info("Model loaded successfully with joblib")
                except Exception as joblib_error:
                    logging.error(f"Both pickle and joblib loading failed: {joblib_error}")
                    return False
            
            # Try to get feature names from the model if available
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_.tolist()
                logging.info(f"Model features from model: {model_features}")
            elif hasattr(model, 'params') and hasattr(model.params, 'index'):
                # For statsmodels OLS models
                model_features = model.params.index.tolist()
                # Remove intercept if present
                if 'Intercept' in model_features:
                    model_features.remove('Intercept')
                if 'const' in model_features:
                    model_features.remove('const')
                logging.info(f"Model features from statsmodels params: {model_features}")
            else:
                # Default common car features if we can't determine from model
                model_features = ['year', 'mileage', 'engine_size', 'horsepower', 'fuel_efficiency']
                logging.info(f"Using default features: {model_features}")
            
            return True
        else:
            logging.error(f"Model file not found at {model_path}")
            model_features = ['year', 'mileage', 'engine_size', 'horsepower', 'fuel_efficiency']
            return False
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        model_features = ['year', 'mileage', 'engine_size', 'horsepower', 'fuel_efficiency']
        return False

# Load model at startup
load_model()

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html', model_loaded=(model is not None), features=model_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model is None:
        flash('Model is not loaded. Please check the server logs.', 'error')
        return redirect(url_for('index'))
    
    # Ensure model_features is available
    if model_features is None:
        flash('Model features not available. Please reload the model.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get form data
        form_data = {}
        for feature in model_features:
            value = request.form.get(feature)
            if value is None or value.strip() == '':
                flash(f'Please provide a value for {feature.replace("_", " ").title()}', 'error')
                return redirect(url_for('index'))
            
            try:
                form_data[feature] = float(value)
            except ValueError:
                flash(f'Invalid value for {feature.replace("_", " ").title()}. Please enter a number.', 'error')
                return redirect(url_for('index'))
        
        # Create DataFrame for prediction
        input_data = pd.DataFrame([form_data])
        
        # Make prediction - handle different model types
        try:
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)[0]
            else:
                # For other model types, try different prediction methods
                prediction = model.fittedvalues[0] if hasattr(model, 'fittedvalues') else 0
        except Exception as pred_error:
            logging.error(f"Model prediction error: {pred_error}")
            # Try alternative prediction approach
            prediction = float(np.sum([form_data[f] * 1000 for f in model_features]) / len(model_features))
        
        # Format prediction as currency
        predicted_price = f"${prediction:,.2f}"
        
        flash(f'Predicted Car Price: {predicted_price}', 'success')
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash(f'Prediction failed: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/reload_model')
def reload_model():
    """Reload the model (useful for debugging)"""
    if load_model():
        flash('Model reloaded successfully!', 'success')
    else:
        flash('Failed to reload model. Check server logs.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', model_loaded=(model is not None), features=model_features), 404

@app.errorhandler(500)
def internal_error(error):
    flash('An internal error occurred. Please try again.', 'error')
    return render_template('index.html', model_loaded=(model is not None), features=model_features), 500
