# Movie Sentiment Classification

This GitHub repository contains code for a movie sentiment classification project.

## Dataset
You can download the dataset from the following link:
[IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

After downloading, extract the file, and the dataset will be structured as follows:

## File Structure

```bash
aclImdb
|----- aclImdb
        |------Train
                |------pos
                        |------12260_10.txt
                                .....
                |-------neg
                        |------12262_3.txt
        |-------Test
                |-------pos
                        |------- ....
                |-------neg
                        |-------- ....
```
### DataSet Information
Dataset contains information about various movie review and their sentiment. 25000 movies are provided for training and 25000 movies are provided for testing. Inside train folder you can find two folders pos and neg.
'pos' contains information about positive sentiment reviews in text format and 'neg' contains information about negative sentiment in text. Similiar is the structure of test folder.

### Steps To execute The Task
**Step1**- Make The Train.csv and Test.csv using make_csv.py . In this csv files two columns should be present first one is 'review' which contains text of the review and 'Sentiment' which contains sentiment of the review where 'Positive' Sentiment is labeled as '1' and negative sentiment is labeled as '0'.

**Step2**- Make a preprocessing pipeline for our model using pipe.py . This pipeline will contain functions for perpreprocessing nlp steps like: removing stop words, removing extra spaces, removing punctuations, lemmatization etc. and finally a class to make word embeddings from preprocessed text. Use the pipeline to process the review column of train and save the resultant csv. Fianlly save the pipeline for future use.

**Step3**- Use the processed dataframe and train a model of your choice on it (eg: RNN, LSTM , NaiveBayes) and after training save the model for future use. Also while training divide your csv in two parts train and val to check the accuracy while training. And after training use Test.csv to futher check accuracy of your model. Choose model which provides best accuracy.

**Step4**- Finally predict.py will be used to make predictions. This file will have a review whose sentiment has to be predicted it will load the model and pipeline and all the classes that it requires and will give predictions accordingly.

### Installing Requirements
Run the following command to install the requirements
```bash
pip install -r requirements.txt
```
