import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from pipe import *
from keras.models import load_model
from keras.models import Sequential
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
    
def check_accuracy(model:Sequential, test_df:pd.DataFrame, preprocessor:NLPPreprocessor):
    
    X_test = np.array(preprocessor.generate_word_embeddings(preprocessor.clean(test_df['review'])))
    y_pred = model.evaluate(X_test, test_df['Sentiment'])
    print("Test Score:", y_pred[0])
    print("Test Accuracy:", y_pred[1])
    return y_pred
test_df=pd.read_csv('test.csv')
# Evaluate on Test.csv
preprocessor = joblib.load('pre_pipeline.pkl')
print(f"Accuracy on Test set: {check_accuracy(load_model('model.h5'), test_df, preprocessor)}")
# Save the trained model and tokenizer
# model.save('model.h5')

# Get the final Dense layer weights
# r = "How? I wondered why I hadn't seen this in theaters, or even a single commercial for it, and then after I saw the movie, I realized I was duped HARDCORE. I am a big Transporter fan, and a big Blade fan, so when I saw this I imagined some killer fight scene between two badasses, lots of gunplay, a whole bunch of stuff. Instead, I got the Ryan Phillippe movie with a brief cameo by Statham and Snipes. The guy that does the audio and video in the crime lab got more screen time than Wesley. It was like renting a Jackie Chan movie expecting a bunch of kung fu and getting Erin Brockavich. I expect bad movies from Hollywood, but actors like Snipes and Statham should treat the fan base better."
# model = load_model('model.h5')

# a = model.predict(preprocessor.generate_word_embeddings(preprocessor.clean(pd.Series([r]))))
# e = model.layers[0]
# weights = e.get_weights()[0]
# print(weights.shape) # shape: (vocab_size, embedding_dim)

# final_layer_weights, final_layer_biases = model.layers[-1].get_weights()

# print("Final Layer Weights:\n", final_layer_weights)
# print("Final Layer Biases:\n", final_layer_biases)

# Example: Multiply embeddings by weights to see contribution
# contributions = np.dot(X_test[0], final_layer_weights)
# print("Input Contribution to Final Prediction:\n", contributions)