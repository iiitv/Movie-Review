import pandas as pd
import joblib
from keras.models import load_model
from pipe import *
from model import *
# Load the trained model, preprocessing pipeline, and embedding generator
model_file = 'model.pkl' 
pipeline_file = 'pre_pipeline.pkl'  

# Load the model
model = joblib.load(model_file)

# Load the preprocessing pipeline
preprocessor_pipeline = joblib.load(pipeline_file)



# Get user input for the review
review = input("Enter the review to predict sentiment: ")

# Preprocess the review using the pipeline
processed_review = preprocessor_pipeline.transform([review])

# Generate embeddings using the embedding generator
embeddings = NLPPreprocessor().single_review_embedding(processed_review)

# Make the prediction
prediction = model.predict(embeddings)
predicted_class = prediction.argmax(axis=-1)  # Get the index of the class with the highest probability

# Interpret the prediction (adjust based on your labeling)
sentiment = "Positive" if predicted_class[0] == 1 else "Negative"

print(f"The predicted sentiment is: {sentiment}")