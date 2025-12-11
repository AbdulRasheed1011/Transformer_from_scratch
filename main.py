import tensorflow as tf
from src.utils.config import load_config
from src.data.tokenizer import load_tokenizer
from src.data.tf_dataloader import preprocess_datasets, create_tf_datasets
from src.models.embedding import TokenAndPositionEmbedding
from src.models.masks import create_padding_mask

def main():
    config = load_config()
    
    tokenizer = load_tokenizer(config)
    
    train_enc, val_enc, test_enc = preprocess_datasets(config, tokenizer)
    train_tf, val_tf, test_tf = create_tf_datasets(train_enc, val_enc, test_enc, config)

    batch = next(iter(train_tf))
    x_dict, y = batch
    input_ids = x_dict['input_ids']         #(batch, seq_len)
    attention_mask = x_dict['attention_mask']

    print('input_ids shape:', input_ids.shape)

    d_model = config['model']['d_model']
    max_len = config['dataset']['max_input_len']
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    embed_layer = TokenAndPositionEmbedding(vocab_size, d_model, max_len)
    x = embed_layer(input_ids)

    pad_mask = create_padding_mask(input_ids, pad_token_id)         #(batch, 1, 1, seq_len)

    print('x shape for encoder :', x.shape)
    print('pad_mask shape:', pad_mask.shape)


if __name__ == '__main__':
    main()