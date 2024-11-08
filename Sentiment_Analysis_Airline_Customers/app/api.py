from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Define the data structure for the incoming request
class TextIn(BaseModel):
    text: str
    model_name: str  # Specify which model to use for prediction

# Load your trained models
print("Loading models...")
try:
    models = {
        'Naive Bayes': joblib.load('../models/naive_bayes_model.pkl'),
        'Logistic Regression': joblib.load('../models/logistic_regression_model.pkl'),
        'Support Vector Machine': joblib.load('../models/support_vector_machine_model.pkl'),
        'Random Forest': joblib.load('../models/random_forest_model.pkl'),
        'XGBoost': joblib.load('../models/xgboost_model.pkl')
    }
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Load the TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    print("TF-IDF Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF vectorizer: {e}")

# Function to map numerical predictions to sentiment labels
def map_sentiment(prediction):
    # Assuming the models return 0 for 'negative', 1 for 'neutral', and 2 for 'positive'
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_mapping.get(prediction, "unknown")

# Endpoint for predicting sentiment
@app.post("/predict/")
async def predict_sentiment(input_data: TextIn):
    try:
        # Debugging: Print the input text and model name
        print(f"Received text: {input_data.text}")
        print(f"Using model: {input_data.model_name}")
        
        # Preprocess input text
        cleaned_text = input_data.text
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        
        # Check if the model name is valid
        if input_data.model_name not in models:
            return {"error": f"Model {input_data.model_name} not found. Available models: {list(models.keys())}"}

        # Get the selected model and predict sentiment
        model = models[input_data.model_name]
        prediction = model.predict(vectorized_text)

        # Convert the result to a sentiment label (positive, negative, neutral)
        sentiment_label = map_sentiment(int(prediction[0]))

        return {
            "model": input_data.model_name,
            "sentiment": sentiment_label
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"An error occurred during prediction: {e}"}
