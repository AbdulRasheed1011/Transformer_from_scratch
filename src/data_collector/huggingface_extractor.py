import pandas
import numpy
import os
from pathlib import Path
from datasets import load_dataset
import yaml

def load_yaml(path = "config.yaml"):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config

class HuggingFaceExtractor:
    def __init__(self, hglink, split, text_column):
        self.hglink = hglink
        self.split = split
        self.text_column = text_column
        self.dataset = None
    def load_data(self):
        ds = load_dataset(self.hglink)

        print(ds['train'][0])



