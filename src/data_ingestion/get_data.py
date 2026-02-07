
import datasets
from datasets import load_dataset
import pandas as pd

def get_data():
    """
    Downloads and loads the CNN/DailyMail dataset.
    Returns:
        dataset dict containing train, validation, and test splits.
    """
    print("Downloading CNN/DailyMail dataset...")
    # Using version 3.0.0 as standard for summarization
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    print("\nDataset Structure:")
    print(dataset)
    
    return dataset

def show_samples(dataset, num_samples=3):
    """
    Prints a few samples from the training set.
    """
    print(f"\nShowing {num_samples} samples from training set:")
    train_data = dataset['train']
    
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        print("Article:")
        print(train_data[i]['article'][:500] + "...") # Print first 500 chars
        print("\nHighlights (Summary):")
        print(train_data[i]['highlights'])
        print("-" * 80)

if __name__ == "__main__":
    try:
        dataset = get_data()
        show_samples(dataset)
        print("\nData collection completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
