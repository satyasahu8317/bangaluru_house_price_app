# Bengaluru House Price Prediction System

An end-to-end Machine Learning web application that predicts house prices in Bengaluru based on location, square footage, number of bathrooms, and BHK configuration.

This project demonstrates the complete lifecycle of a real-world ML system — from data preprocessing and model training to API development, frontend integration, and cloud deployment.

---

## Project Overview

The goal of this project is to build a production-style machine learning application that allows users to:

- Predict house prices using a trained ML model
- Access predictions through a web interface
- Use a REST API for programmatic access
- Experience a responsive frontend with light and dark themes
- Interact with a secured and rate-limited backend

---

## Features

### Machine Learning
- Data cleaning and feature engineering
- Outlier removal and preprocessing
- OneHotEncoding for categorical variables
- StandardScaler for numerical features
- Ridge Regression model
- Pipeline-based architecture

### Backend (Flask)
- Web UI prediction route
- REST API endpoint
- API key authentication
- Rate limiting for API protection
- Request validation and structured responses

### Frontend
- Responsive design
- Light and Dark mode support
- Interactive prediction interface

### Deployment
- Cloud-hosted Flask application
- Production-ready structure
- Gunicorn support

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML / CSS
- Git & GitHub
- Cloud Deployment Platform

---

## Project Structure

bengaluru-house-price-app/
bengaluru-house-price-app/
│
├── app.py
├── RidgeModel.pkl
├── requirements.txt
├── Procfile
├── README.md
└── templates/
    └── index.html
