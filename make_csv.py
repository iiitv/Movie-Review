import os
import pandas as pd

def make_csv(base_path, output_path):
    train_data = []
    test_data = []

    # Iterate through train and test directories
    for data_type in ['train', 'test']:
        data_dir = os.path.join(base_path, data_type)
        for sentiment in ['pos', 'neg']:
            sentiment_dir = os.path.join(data_dir, sentiment)

            # Iterate through files in each sentiment directory
            for filename in os.listdir(sentiment_dir):
                file_path = os.path.join(sentiment_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    review = file.read().strip()  # Read the review content

                    # Assign sentiment label (1 for positive, 0 for negative)
                    sentiment_label = 1 if sentiment == 'pos' else 0

                    # Append data to the appropriate list
                    if data_type == 'train':
                        train_data.append({'reviews': review, 'sentiment': sentiment_label})
                    else:
                        test_data.append({'reviews': review, 'sentiment': sentiment_label})

    # Create DataFrames from the collected data
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Save DataFrames to CSV files
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
