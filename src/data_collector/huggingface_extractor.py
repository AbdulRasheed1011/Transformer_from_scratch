import os
import numpy  # currently unused, but fine to keep
from datasets import load_dataset


class HuggingFaceExtractor:
    def __init__(
        self,
        hglink,
        data_path='data/huggingface',
        train_data_name='train.txt',
        test_data_name='test.txt'
    ):
        self.hglink = hglink
        self.data_path = data_path
        self.train_data_name = train_data_name
        self.test_data_name = test_data_name

    def load_save(self):
        # Load dataset from Hugging Face
        ds = load_dataset(self.hglink)

        # Expecting 'train' and 'test' splits
        train_ds = ds['train']
        test_ds = ds['test']

        # Extract text column
        train_texts = train_ds['text']
        test_texts = test_ds['text']

        # Use the instance attributes here
        file_map = {
            self.train_data_name: train_texts,
            self.test_data_name: test_texts,
        }

        # Save each split to its file
        for file_name, texts in file_map.items():
            file_path = os.path.join(self.data_path, file_name)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if os.path.exists(file_path):
                print("File already exists:", file_path)
            else:
                print("Creating file:", file_path)
                with open(file_path, "w", encoding="utf-8") as fp:
                    for t in texts:
                        t = str(t).replace("\n", " ").strip()
                        if t:
                            fp.write(t + "\n")

hg = HuggingFaceExtractor("agentlans/high-quality-english-sentences")
hg.load_save()