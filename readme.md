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

