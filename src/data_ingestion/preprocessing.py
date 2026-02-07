
import os
import yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import tensorflow as tf
from src.data_ingestion.get_data import get_data
import numpy as np

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_tokenizer(dataset, vocab_size=8192, save_path="tokenizer.json"):
    """
    Trains a BPE tokenizer on the dataset.
    """
    print("Training tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    
    # Define an iterator to yield texts from the dataset
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset['train']), batch_size):
            yield dataset['train'][i : i + batch_size]['article']
            yield dataset['train'][i : i + batch_size]['highlights']

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Post-processing to add [CLS] and [SEP]
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

def load_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)

def create_tf_dataset(dataset, tokenizer, max_input_length, max_output_length, batch_size):
    """
    Creates a tf.data.Dataset from the Hugging Face dataset.
    """
    
    def encode(article, summary):
        # Convert to python string if they are bytes (TensorFlow passes bytes)
        if isinstance(article, bytes):
            article = article.decode('utf-8')
        if isinstance(summary, bytes):
            summary = summary.decode('utf-8')
            
        # Tokenize article
        # encoding.ids given automatically adds [CLS]...[SEP] due to post_processor
        # We need to handle padding manually or via tokenizer.enable_padding()
        # But enabling padding in tokenizer might conflict with batching if not done carefully.
        # Let's do manual padding here for simplicity and explicit control.
        
        enc_check = tokenizer.encode(article)
        dec_check = tokenizer.encode(summary)
        
        enc_ids = enc_check.ids
        dec_ids = dec_check.ids
        
        pad_id = tokenizer.token_to_id("[PAD]")
        
        # Truncate
        if len(enc_ids) > max_input_length:
            enc_ids = enc_ids[:max_input_length]
        if len(dec_ids) > max_output_length:
            dec_ids = dec_ids[:max_output_length]
            
        # Pad
        enc_ids = enc_ids + [pad_id] * (max_input_length - len(enc_ids))
        dec_ids = dec_ids + [pad_id] * (max_output_length - len(dec_ids))
        
        return enc_ids, dec_ids

    def tf_encode(article, summary):
        # numpy function wrapper
        enc_ids, dec_ids = tf.numpy_function(encode, [article, summary], [tf.int64, tf.int64])
        
        # Set shape
        enc_ids.set_shape([max_input_length])
        dec_ids.set_shape([max_output_length])
        
        return (enc_ids, dec_ids), dec_ids 
        
    def tf_encode_map(article, summary):
        # We need to wrap the python function
        # The python function returns lists, which TF converts to tensors
        enc, dec = tf.numpy_function(encode, [article, summary], [tf.int64, tf.int64])
        enc.set_shape([max_input_length])
        dec.set_shape([max_output_length])
        return enc, dec
        
    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        if 'train' in dataset:
            train_dataset = dataset['train']
        else:
            # Fallback or assume it's a dict of splits but we only want one? 
            # For this specific function, historically it was just for train.
            # But now we use it for val/test too.
            # If passed a dict without train, raising error is fine OR 
            # we should update usage.
            # But here we want to support passing just a Dataset object.
            train_dataset = dataset
    else:
        # It's likely a Dataset object
        train_dataset = dataset

    # If it was a dict but didn't have train, and we assigned it to train_dataset,
    # the next steps might fail if it doesn't behave like a Dataset.
    # But let's assume usage is correct: either full dict or specific split. 

    # Convert to tf.data.Dataset
    # dataset['train'] is a Dataset object, we can iterate over it or use from_generator
    # from_tensor_slices might fail if data is too big to fit in memory
    # Use from_generator for large datasets
    
    def gen():
        for i in range(len(train_dataset)):
            yield train_dataset[i]['article'], train_dataset[i]['highlights']
            
    train_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    
    train_ds = train_ds.map(tf_encode_map, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache() # Caching might be too heavy for full dataset
    # Let's remove cache for full dataset or cache to file
    # train_ds = train_ds.cache() 
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True) # Drop remainder for consistent batch shapes
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, tokenizer

def get_train_val_datasets(dataset, tokenizer, max_length, batch_size):
    # Split handled by get_data (dataset['train'], dataset['validation'])
    # Implement similarly for validation
    pass 

if __name__ == "__main__":
    config = load_config()
    dataset = get_data()
    
    # Setup tokenizer
    tokenizer_path = config['paths']['tokenizer_file']
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer = train_tokenizer(dataset, vocab_size=config['data']['vocab_size'], save_path=tokenizer_path)
    
    # Test Pipeline
    print("Creating Dataset pipeline...")
    train_ds, _ = create_tf_dataset(dataset, tokenizer, config['data']['max_input_length'], config['data']['max_output_length'], config['training']['batch_size'])
    
    print("\nInspecting one batch...")
    for (enc, dec) in train_ds.take(1):
        print("Encoder Input Shape:", enc.shape)
        print("Decoder Input Shape:", dec.shape)
        print("Encoder Input Sample (IDs):", enc[0].numpy())
        print("Decoder Input Sample (IDs):", dec[0].numpy())
        
        # Decode back to text
        print("\nDecoded Article:", tokenizer.decode(enc[0].numpy()))
        print("Decoded Summary:", tokenizer.decode(dec[0].numpy()))
        break
