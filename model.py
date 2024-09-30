import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import joblib

# Load your processed DataFrame (Assuming cleaned_reviews is in the DataFrame)
df = pd.read_csv('train.csv')


# Function to create and train the model you can adjust the arguments as per your requirement
def train(X_train, y_train, X_val, y_val, embedding_dim=100, batch_size=64, epochs=5):

    
    return model
# Function to check the accuracy of the model
def check_accuracy(model, test_df):
    
    return accuracy


# Prepare the data
df=pd.read_csv('processed_train.csv')
df.drop(columns=['reviews'], inplace=True)  # Drop the original reviews
    
# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df['embeddings'], df['sentiment'], test_size=0.2, random_state=42)

# Train the model
model = train(X_train, y_train, X_val, y_val)
test_df=pd.read_csv('test.csv')

# Evaluate on Test.csv
print(f"Accuracy on Test set: {check_accuracy(model, test_df)}")
# Save the trained model and tokenizer
joblib.dump(model, 'model.pkl')


