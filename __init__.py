from flask import Flask
import joblib


# Initialize App
app = Flask(__name__)

# Load models
model = joblib.load('model/differential_sticking.dat.gz')
