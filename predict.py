import pandas as pd
import joblib
from keras.models import load_model
from pipe import NLPPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, GlobalAveragePooling1D
from keras.utils import to_categorical, pad_sequences
from ast import literal_eval
from pipe import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

    # Load the trained model, preprocessing pipeline, and embedding generator
model_file = 'model.h5' 
preprocessor = joblib.load('pre_pipeline.pkl')


# Load the model
model = load_model(model_file)

# extra axis for sentiment relations
def interpret_sentiment(prob):
    if prob <= 0.20:
        return "Strongly Negative"
    elif prob <= 0.40:
        return "Negative"
    elif prob <= 0.49:
        return "Neutral / Slightly Negative"
    elif prob == 0.50:
        return "Neutral / Ambiguous"
    elif prob <= 0.60:
        return "Neutral / Slightly Positive"
    elif prob <= 0.80:
        return "Positive"
    else:
        return "Strongly Positive"
    
def predict_sentiment(review:str):
    # Preprocess the review using the pipeline
    a = {"a": [review]}
    processed_review = preprocessor.clean(pd.DataFrame(a)['a'])
    # Generate embeddings using the embedding generator
    embeddings = preprocessor.single_review_embedding(processed_review)
    
    # Make the prediction
    prediction = model.predict(embeddings)
    predicted_prob = prediction[0] 
    # Interpret the prediction (adjust based on your labeling)
    sentiment = interpret_sentiment(predicted_prob)
    
    print(f"The predicted sentiment is: {sentiment}")

predict_sentiment("Harry potter is magical movie, just amazing, i liked whole plot.")

# 1/1 [==============================] - ETA: 0s
# 1/1 [==============================] - 1s 910ms/step
# The predicted sentiment is: Strongly Positive