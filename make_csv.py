import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
TRAIN = 'train.csv'
TEST = 'test.csv'


def read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read().strip()

def build_reviews(path: str, max_workers=8) -> tuple[list]:
    neg_review = [] 
    pos_review = []  
    neg = os.path.join(path, "neg")
    pos = os.path.join(path, "pos")
    
    neg_files = [os.path.join(neg, x) for x in os.listdir(neg)]
    pos_files = [os.path.join(pos, x) for x in os.listdir(pos)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        neg_review.extend(executor.map(read_file, neg_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pos_review.extend(executor.map(read_file, pos_files))
        
    
    return pos_review, neg_review

# This function creates CSVs from the data in the folder structure
def make_csv(base_path: str, output_path: str, max_workers=8) -> None:
    test_path = os.path.join(base_path, "test")
    train_path = os.path.join(base_path, "train")
    
    ### for test csv
    pos_review, neg_review = build_reviews(test_path, max_workers)

    ### for train csv
    pos_review1, neg_review1 = build_reviews(train_path, max_workers)

    testreview = [*pos_review, *neg_review]
    trainreview = [*pos_review1, *neg_review1]
    # optimistic way of marking review either 0 or 1.
    test_csv = {'review':testreview, 
                'Sentiment': [1]*len(pos_review) + [0]*len(neg_review)} 
    train_csv = {'review':trainreview, 
                 'Sentiment': [1]*len(pos_review1) + [0]*len(neg_review1)}
    
    test_csv = pd.DataFrame(test_csv)
    test_csv.to_csv(os.path.join(output_path, TEST), index=False)

    train_csv = pd.DataFrame(train_csv)
    train_csv.to_csv(os.path.join(output_path, TRAIN), index=False)
    
# Example usage
# make_csv('aclImdb', '', max_workers=10)
# 38.717 SECONDS
