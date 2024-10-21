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

def plot_graphs(history:Sequential, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.savefig(string+".png")
  plt.show()
def plot(history):
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


def train(x_train, y_train, x_val, y_val, vocab_size = 70000, embedding_dim=100, epochs=5, max_length=350, plotg=True):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
    if plotg:
        plot(history)
    return model

# Function to check the accuracy of the model

def check_accuracy(model:Sequential, test_df:pd.DataFrame, preprocessor:NLPPreprocessor):
    X_test = np.array(preprocessor.generate_word_embeddings(preprocessor.clean(test_df['review'])))
    scores = model.evaluate(X_test, test_df['Sentiment'])
    print("Test Score:", scores[0])
    print("Test Accuracy:", scores[1])
    return scores


# Load your processed DataFrame (Assuming cleaned_reviews is in the DataFrame)
df = pd.read_csv('processed_train.csv')

x = np.stack(df['embeddings'].apply(literal_eval))
y = np.stack(df['Sentiment'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
model = train(x_train, y_train, x_val, y_val)
model.save('model.h5')


test_df=pd.read_csv('test.csv')
# Evaluate on Test.csv
preprocessor = joblib.load('pre_pipeline.pkl')

# check accuracy
check_accuracy(model, test_df, preprocessor)
# Test Score: 0.44428691267967224
# Test Accuracy: 0.8607199788093567 -> 86.4% test accuracy which is near to our val accuracy 88.76, no model overfeeding :)

################  OUT ####################################
# Epoch 1/5
# 625/625 [==============================] - 86s 135ms/step - loss: 0.4868 - accuracy: 0.7851 - val_loss: 0.3007 - val_accuracy: 0.8812
# Epoch 2/5
# 625/625 [==============================] - 79s 127ms/step - loss: 0.2252 - accuracy: 0.9162 - val_loss: 0.2670 - val_accuracy: 0.8946
# Epoch 3/5
# 625/625 [==============================] - 82s 131ms/step - loss: 0.1450 - accuracy: 0.9500 - val_loss: 0.2712 - val_accuracy: 0.8962
# Epoch 4/5
# 625/625 [==============================] - 83s 133ms/step - loss: 0.0966 - accuracy: 0.9693 - val_loss: 0.2963 - val_accuracy: 0.8908
# Epoch 5/5
# 625/625 [==============================] - 78s 125ms/step - loss: 0.0653 - accuracy: 0.9811 - val_loss: 0.3199 - val_accuracy: 0.8876
# 782/782 [==============================] - 6s 7ms/step - loss: 0.3961 - accuracy: 0.8641
# Test Score: 0.3961312770843506
# Test Accuracy: 0.8641200065612793

