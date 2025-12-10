import os
import numpy as np


class TextVectorizer:
    def __init__(self, max_vocab_size=20000, min_freq=1, max_len=32):
        """
        max_vocab_size: maximum number of tokens in vocab (including special tokens)
        min_freq: minimum frequency for a word to be included
        max_len: fixed sequence length (for padding/truncating)
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_len = max_len

        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"

    def Normalize(self, text):
        return text.strip().lower().split()

    def fit(self, file_path = 'data/huggingface/trai.txt'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        
        freq = {}

        with open(file_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                tokens = self.Normalize(line)
                for tok in tokens:
                    freq[tok] = freq.get(tok, 0) + 1
        filtered_tokens = []
        for w,c in freq.items():
            if c >= self.min_freq:
                filtered_tokens.append(w)
        filtered_tokens.sort(key = lambda w: freq[w], reverse = True)

        vocab, index = {}, 2 # initilizing because 'pad':0, 'unk':1
        vocab[self.PAD_TOKEN] = 0
        vocab[self.UNK_TOKEN] = 1
        
        for token in filtered_tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1
        
        inverse_vocab = {index:token for token, index in vocab.items()}
        
        return vocab, inverse_vocab
        
tex = TextVectorizer()
vocab, inverse_vocab = tex.fit('data/huggingface/train.txt')
print(len(vocab), len(inverse_vocab))