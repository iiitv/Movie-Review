import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import pandas as pd
import joblib
import re
try: 
    stopwords.words("english")
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
TAG_RE = re.compile(r'<[^>]+>') # remove html tags

class RemoveTags(BaseEstimator, TransformerMixin):
    def transform(self, X:pd.DataFrame):
        return X.apply(lambda text:TAG_RE.sub('', text.lower()))  #this is first step of pipline therefore it will also be lowercasing text for later lines.
        

    def fit(self, X:pd.DataFrame, y=None):
        return self

class RemoveSingleChar(BaseEstimator, TransformerMixin):
    def transform(self, X:pd.DataFrame):
        return X.apply(lambda text:re.sub(r"\s+[a-zA-Z]\s+", ' ', text))
        

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
class RemovePunctuation(BaseEstimator, TransformerMixin):
    def transform(self, X:pd.DataFrame):
        return X.apply(lambda text:re.sub('[^a-zA-Z]', ' ', text))
        

    def fit(self, X:pd.DataFrame, y=None):
        return self


class RemoveExtraSpaces(BaseEstimator, TransformerMixin):
    def transform(self, X:pd.DataFrame):
        return X.apply(lambda text: re.sub(r'\s+', ' ', text))

    def fit(self, X:pd.DataFrame, y=None):
        return self


class RemoveStopWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    def transform(self, X:pd.DataFrame):
        return X.apply(lambda text: pattern.sub('', text))
    def fit(self, X:pd.DataFrame, y=None):
        return self


class LemmatizeText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, X:pd.DataFrame):
        return X.apply(lambda text: ''.join([self.lemmatizer.lemmatize(word)+" " for word in text.split()]))

    def fit(self, X:pd.DataFrame, y=None):
        return self


class NLPPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, vocabsize=70000, max_seq_length=350):
        self.max_seq_length = max_seq_length
        self.embedding_model = Tokenizer(num_words = vocabsize, oov_token="<oov>") 
        self.pipeline = Pipeline(steps=[
                ('remove_tags', RemoveTags()),
                ('remove_punctuation', RemovePunctuation()),
                ('remove_single_char', RemoveSingleChar()),
                ('remove_extra_spaces', RemoveExtraSpaces()),
                ('remove_stop_words', RemoveStopWords()),
                ('lemmatize', LemmatizeText()),
            ])
    def fit(self, texts: pd.DataFrame, y=None):
        return self

    def generatetokens(self, texts: pd.DataFrame):
        self.embedding_model.fit_on_texts(texts.tolist())
        return 

    def generate_word_embeddings(self, texts: pd.DataFrame):
        padedsequences = pad_sequences(self.embedding_model.texts_to_sequences(texts), maxlen=self.max_seq_length, padding='post', truncating='post')
        return padedsequences

    def single_review_embedding(self, text):
        padedsequences = pad_sequences(self.embedding_model.texts_to_sequences(text), maxlen=self.max_seq_length, padding='post', truncating='post')
        return padedsequences
    
    def clean(self, texts:pd.DataFrame):
        return self.pipeline.transform(texts)
        
    
    def save(self):
        joblib.dump(self, 'pre_pipeline.pkl')


def preprocess():
    preprocessor = NLPPreprocessor()

    df = pd.read_csv('train.csv')

    # save necessary things for furthur usage.

    df['review'] = preprocessor.clean(df['review'])

    #generate this tokens only for train data review, for test it must not be used
    preprocessor.generatetokens(df['review'])

    df['embeddings'] = preprocessor.generate_word_embeddings(df['review']).tolist()
    
    # print(f"vocab size = ", len(preprocessor.embedding_model.word_counts)+1)
    # 66507 -> total vocab

    df.to_csv("processed_train.csv", index=False)

    preprocessor.save()

# preprocess()
# [Done] exited with code=0 in 59.544 seconds