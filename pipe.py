import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
import pandas as pd
import joblib
# Ensure you have the NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')


class RemovePunctuation(BaseEstimator, TransformerMixin):
# write actual logic for removing punctuation
    def transform(self, X):
        pass

    def fit(self, X, y=None):
        return self


class RemoveExtraSpaces(BaseEstimator, TransformerMixin):
    # write actual logic for removing extra spaces
    def transform(self, X):
        pass

    def fit(self, X, y=None):
        return self


class RemoveStopWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
# write actual logic for removing stop words
    def transform(self, X):
        pass
    def fit(self, X, y=None):
        return self


class LemmatizeText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
# write actual logic for lemmatizing text
    def transform(self, X):
        pass
    def fit(self, X, y=None):
        return self


class NLPPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
            here you must define your model as a parameter
            for eg: self.embedding_model = TfidfVectorizer()
        """
        pass
    def transform(self, X):
        return X  # No additional transformation at this level

    def fit(self, X, y=None):
        return self
# write logic to generate word embeddings, you can also change input arguments as per your requirement
    def generate_word_embeddings(self, texts, vector_size=100, window=5, min_count=1, workers=4):
        pass
# write logic to generate embedding for a single review
    def single_review_embedding(self, text):
        pass
    
pipeline = Pipeline(steps=[
    ('remove_punctuation', RemovePunctuation()),
    ('remove_extra_spaces', RemoveExtraSpaces()),
    ('remove_stop_words', RemoveStopWords()),
    ('lemmatize', LemmatizeText()),
])

df = pd.read_csv('train.csv')
reviews = df['reviews']
# getting the cleaned reviews
cleaned_reviews = pipeline.transform(reviews)

# Creating a new list for embeddings
preprocessor = NLPPreprocessor()

# Generate embeddings for all cleaned reviews at once
embeddings = preprocessor.generate_word_embeddings(cleaned_reviews)  # Generate embeddings for the entire list

# If needed, convert the embeddings to a list or DataFrame for easier handling
embeddings_list = embeddings.tolist()  # Convert to list if embeddings are in a numpy array

df['embeddings'] = embeddings_list


# Save the processed DataFrame
pd.to_csv('processed_train.csv')
# Saving the pipeline
joblib.dump(pipeline, 'pre_pipeline.pkl')


