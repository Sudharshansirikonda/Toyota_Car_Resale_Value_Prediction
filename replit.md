# Car Price Predictor

## Overview

A Flask-based web application that provides car price predictions using a pre-trained machine learning model. The application loads an OLS (Ordinary Least Squares) regression model from a pickle file and offers a web interface for users to input car features and receive price estimates. The system is designed as a simple prediction service with a clean, responsive UI built using Bootstrap and Flask templating.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Template Engine**: Flask's Jinja2 templating system for server-side rendering
- **UI Framework**: Bootstrap with dark theme for responsive design
- **Styling**: Custom CSS with gradient backgrounds and modern card-based layouts
- **Icons**: Font Awesome for visual enhancement
- **Form Handling**: HTML forms with Flask form processing and validation

### Backend Architecture
- **Web Framework**: Flask for lightweight web application development
- **Model Loading**: Pickle-based model serialization for ML model persistence
- **Feature Detection**: Dynamic feature extraction from loaded models with fallback defaults
- **Error Handling**: Comprehensive logging and user feedback for model loading issues
- **Session Management**: Flask sessions with configurable secret keys

### Data Processing
- **ML Framework**: Uses pandas and numpy for data manipulation
- **Model Type**: OLS regression model for price prediction
- **Feature Handling**: Dynamic feature detection with common car attributes as defaults (year, mileage, engine_size, horsepower, fuel_efficiency)
- **Input Validation**: Server-side validation through Flask form processing

### Application Structure
- **Modular Design**: Separation of concerns with app.py containing core logic and main.py as entry point
- **Static Assets**: Organized CSS and potential future JavaScript files
- **Template Organization**: HTML templates in dedicated templates directory
- **Model Storage**: External model files in attached_assets directory

## External Dependencies

### Python Libraries
- **Flask**: Web framework for application routing and templating
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing support
- **pickle**: Model serialization and deserialization

### Frontend Dependencies
- **Bootstrap**: CSS framework loaded via CDN for responsive design
- **Font Awesome**: Icon library via CDN for UI enhancement

### Development Environment
- **Replit**: Hosting platform with built-in development environment
- **Environment Variables**: SESSION_SECRET for secure session management

### Model Dependencies
- **Pre-trained Model**: OLS regression model stored as pickle file
- **Model File**: `attached_assets/ols_model_1754216076665.pkl`
- **Feature Requirements**: Dynamic feature detection with car-specific attributes