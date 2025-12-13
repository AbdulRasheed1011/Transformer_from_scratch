import tensorflow as tf
from src.data.dataset import load_cnn_dailymail
from src.data.tokenizer import tokenizer_batch


def preprocess_datasets(config, tokenizer):
    train_ds, val_ds, test_ds = load_cnn_dailymail(config)

    def _tokenize_fn(batch):
        return tokenizer_batch(tokenizer, batch, config)

    train_enc = train_ds.map(
        _tokenize_fn,
        batched = True,
        remove_columns = train_ds.column_names,
    )

    val_enc = val_ds.map(
        _tokenize_fn,
        batched = True,
        remove_columns = val_ds.column_names,
    )

    test_enc = test_ds.map(
        _tokenize_fn,
        batched = True,
        remove_columns = test_ds.column_names,
    )

    return train_enc, val_enc, test_enc

def create_tf_datasets(train_enc, val_enc, test_enc, config):
    
    batch_size = config['training']['batch_size']
    input_cols = ['input_ids', 'attention_mask']
    label_cols = ['labels']

    train_tf = train_enc.to_tf_dataset(
        columns = input_cols,
        label_cols = label_cols,
        shuffle = True,
        batch_size= batch_size,
    )

    val_tf = val_enc.to_tf_dataset(
        columns = input_cols,
        label_cols = label_cols,
        shuffle = False,
        batch_size = batch_size,
    )

    test_tf = test_enc.to_tf_dataset(
        columns = input_cols,
        label_cols = label_cols,
        shuffle = False,
        batch_size = batch_size,
    )

    return train_tf, val_tf, test_tf